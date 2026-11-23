"""Reframing helpers for the video pipeline."""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import copy
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from .common import (
    FACE_PRESENCE_MIN_AREA,
    FACE_PRESENCE_RATE,
    FACE_RATE_WINDOW,
    KENBURNS_FOCUS_Y,
    FIT_DETAIL_MARGIN_FRAC,
    FIT_MAX_FOREGROUND_H,
    FIT_MAX_ZOOM_LEVELS,
    FIT_VISIBLE_LEVELS,
    FIT_ZOOM_HOLD_SECONDS,
    KENBURNS_MIN_SECONDS,
    KENBURNS_SUPERSAMPLE_H,
    KENBURNS_SUPERSAMPLE_W,
    KENBURNS_ZOOM_DELTA,
    LAYOUT_SNAP_WINDOW,
    MIN_LAYOUT_SECONDS,
    OUTPUT_FPS,
    UPSCALE_SHARPEN_FILTER,
    VERTICAL_SCALE_CHAIN,
    logger,
)
from .face_detection import open_haar_cascade, open_yunet_detector
from .scene_context import (
    ContextAnalyzer,
    SceneEvidence,
    build_context_layout_plan,
    context_scene_cuts,
)
from .ffmpeg import (
    build_audio_output_args,
    build_final_video_encode_args,
    clamp_even,
    ffmpeg_escape_filter_path,
    ffprobe_duration,
    ffprobe_has_audio,
    ffprobe_video_size,
    round_to_even,
    run_ffmpeg_command,
    subtitles_filter_fragment,
)
from .broll_overlay import (
    broll_input_args,
    broll_overlay_graph,
    normalize_broll_overlays,
)
from .source_limits import (
    FPS_EPSILON,
    MAX_OUTPUT_FPS,
    fit_within_1080p,
    output_rate_token,
    probe_video_geometry,
)


def detect_optimal_crop_region(
    video_path: Path,
    start_time: float,
    end_time: float,
    target_ratio: float = 9 / 16,
) -> Tuple[int, int, int, int]:
    """Detect optimal crop region using improved face detection."""
    try:
        original_width, original_height = ffprobe_video_size(video_path)

        # Calculate target dimensions and ensure they're even
        if original_width / original_height > target_ratio:
            new_width = round_to_even(int(original_height * target_ratio))
            new_height = round_to_even(original_height)
        else:
            new_width = round_to_even(original_width)
            new_height = round_to_even(int(original_width / target_ratio))

        # Try improved face detection
        face_centers = detect_faces_in_clip(video_path, start_time, end_time)

        # Calculate crop position
        if face_centers:
            # Use weighted average of face centers with temporal consistency
            total_weight = sum(
                area * confidence for _, _, area, confidence in face_centers
            )
            if total_weight > 0:
                weighted_x = (
                    sum(
                        x * area * confidence for x, y, area, confidence in face_centers
                    )
                    / total_weight
                )
                weighted_y = (
                    sum(
                        y * area * confidence for x, y, area, confidence in face_centers
                    )
                    / total_weight
                )

                # Add slight bias towards upper portion for better face framing
                weighted_y = max(0, weighted_y - new_height * 0.1)

                x_offset = max(
                    0, min(int(weighted_x - new_width // 2), original_width - new_width)
                )
                y_offset = max(
                    0,
                    min(
                        int(weighted_y - new_height // 2), original_height - new_height
                    ),
                )

                logger.info(
                    f"Face-centered crop: {len(face_centers)} faces detected with improved algorithm"
                )
            else:
                # Center crop
                x_offset = (
                    (original_width - new_width) // 2
                    if original_width > new_width
                    else 0
                )
                y_offset = (
                    (original_height - new_height) // 2
                    if original_height > new_height
                    else 0
                )
        else:
            # Center crop
            x_offset = (
                (original_width - new_width) // 2 if original_width > new_width else 0
            )
            y_offset = (
                (original_height - new_height) // 2
                if original_height > new_height
                else 0
            )
            logger.info("Using center crop (no faces detected)")

        # Ensure offsets are even too
        x_offset = round_to_even(x_offset)
        y_offset = round_to_even(y_offset)

        logger.info(
            f"Crop dimensions: {new_width}x{new_height} at offset ({x_offset}, {y_offset})"
        )
        return (x_offset, y_offset, new_width, new_height)

    except Exception as e:
        logger.error(f"Error in crop detection: {e}")
        # Fallback to center crop
        original_width, original_height = ffprobe_video_size(video_path)
        if original_width / original_height > target_ratio:
            new_width = round_to_even(int(original_height * target_ratio))
            new_height = round_to_even(original_height)
        else:
            new_width = round_to_even(original_width)
            new_height = round_to_even(int(original_width / target_ratio))

        x_offset = (
            round_to_even((original_width - new_width) // 2)
            if original_width > new_width
            else 0
        )
        y_offset = (
            round_to_even((original_height - new_height) // 2)
            if original_height > new_height
            else 0
        )

        return (x_offset, y_offset, new_width, new_height)


def detect_faces_in_clip(
    video_path: Path, start_time: float, end_time: float
) -> List[Tuple[int, int, int, float]]:
    """
    Improved face detection using multiple methods and temporal consistency.
    Returns list of (x, y, area, confidence) tuples.
    """
    face_centers = []

    try:
        # YuNet (most accurate) with Haar as the last-resort fallback
        yunet, haar_cascade = _open_face_detectors()

        # Try to load DNN face detector (more accurate than Haar)
        dnn_net = None
        try:
            # Load OpenCV's DNN face detector
            prototxt_path = cv2.data.haarcascades.replace(
                "haarcascades", "opencv_face_detector.pbtxt"
            )
            model_path = cv2.data.haarcascades.replace(
                "haarcascades", "opencv_face_detector_uint8.pb"
            )

            # If DNN model files don't exist, we'll fall back to Haar cascade
            import os

            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                dnn_net = cv2.dnn.readNetFromTensorflow(model_path, prototxt_path)
                logger.info("OpenCV DNN face detector loaded as backup")
            else:
                logger.info("OpenCV DNN face detector not available")
        except Exception:
            logger.info("OpenCV DNN face detector failed to load")

        # Sample more frames for better face detection (every 0.5 seconds)
        duration = end_time - start_time
        sample_interval = min(0.5, duration / 10)  # At least 10 samples, max every 0.5s
        sample_times = []

        current_time = start_time
        while current_time < end_time:
            sample_times.append(current_time)
            current_time += sample_interval

        # Ensure we always sample the middle and end
        if duration > 1.0:
            middle_time = start_time + duration / 2
            if middle_time not in sample_times:
                sample_times.append(middle_time)

        sample_times = [t for t in sample_times if t < end_time]
        logger.info(f"Sampling {len(sample_times)} frames for face detection")

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            logger.warning("Unable to open video for face detection: %s", video_path)
            return []

        for sample_time in sample_times:
            try:
                capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, sample_time) * 1000.0)
                ok, frame_bgr = capture.read()
                if not ok or frame_bgr is None:
                    continue
                height, width = frame_bgr.shape[:2]
                detected_faces = []

                # Try YuNet first (most accurate)
                if yunet is not None:
                    try:
                        for fx, fy, fw, fh, confidence in yunet.detect(frame_bgr):
                            x, y, w, h = int(fx), int(fy), int(fw), int(fh)
                            if w > 30 and h > 30:  # Minimum face size
                                detected_faces.append((x, y, w, h, confidence))
                    except Exception as e:
                        logger.warning(
                            f"YuNet detection failed for frame at {sample_time}s: {e}"
                        )

                # If YuNet didn't find faces, try DNN detector
                if not detected_faces and dnn_net is not None:
                    try:
                        blob = cv2.dnn.blobFromImage(
                            frame_bgr, 1.0, (300, 300), [104, 117, 123]
                        )
                        dnn_net.setInput(blob)
                        detections = dnn_net.forward()

                        for i in range(detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            if confidence > 0.5:  # Confidence threshold
                                x1 = int(detections[0, 0, i, 3] * width)
                                y1 = int(detections[0, 0, i, 4] * height)
                                x2 = int(detections[0, 0, i, 5] * width)
                                y2 = int(detections[0, 0, i, 6] * height)

                                w = x2 - x1
                                h = y2 - y1

                                if w > 30 and h > 30:  # Minimum face size
                                    detected_faces.append((x1, y1, w, h, confidence))
                    except Exception as e:
                        logger.warning(
                            f"DNN detection failed for frame at {sample_time}s: {e}"
                        )

                # If still no faces found, use Haar cascade
                if not detected_faces:
                    try:
                        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                        faces = haar_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.05,  # More sensitive
                            minNeighbors=3,  # Less strict
                            minSize=(40, 40),  # Smaller minimum size
                            maxSize=(
                                int(width * 0.7),
                                int(height * 0.7),
                            ),  # Maximum size limit
                        )

                        for x, y, w, h in faces:
                            # Estimate confidence based on face size and position
                            face_area = w * h
                            relative_size = face_area / (width * height)
                            confidence = min(
                                0.9, 0.3 + relative_size * 2
                            )  # Rough confidence estimate
                            detected_faces.append((x, y, w, h, confidence))
                    except Exception as e:
                        logger.warning(
                            f"Haar cascade detection failed for frame at {sample_time}s: {e}"
                        )

                # Process detected faces
                for x, y, w, h, confidence in detected_faces:
                    face_center_x = x + w // 2
                    face_center_y = y + h // 2
                    face_area = w * h

                    # Filter out very small or very large faces
                    frame_area = width * height
                    relative_area = face_area / frame_area

                    if (
                        0.005 < relative_area < 0.3
                    ):  # Face should be 0.5% to 30% of frame
                        face_centers.append(
                            (face_center_x, face_center_y, face_area, confidence)
                        )

            except Exception as e:
                logger.warning(f"Error detecting faces in frame at {sample_time}s: {e}")
                continue

        capture.release()

        # Remove outliers (faces that are very far from the median position)
        if len(face_centers) > 2:
            face_centers = filter_face_outliers(face_centers)

        logger.info(f"Detected {len(face_centers)} reliable face centers")
        return face_centers

    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        return []


def filter_face_outliers(
    face_centers: List[Tuple[int, int, int, float]],
) -> List[Tuple[int, int, int, float]]:
    """Remove face detections that are outliers (likely false positives)."""
    if len(face_centers) < 3:
        return face_centers

    try:
        # Calculate median position
        x_positions = [x for x, y, area, conf in face_centers]
        y_positions = [y for x, y, area, conf in face_centers]

        median_x = np.median(x_positions)
        median_y = np.median(y_positions)

        # Calculate standard deviation
        std_x = np.std(x_positions)
        std_y = np.std(y_positions)

        # Filter out faces that are more than 2 standard deviations away
        filtered_faces = []
        for face in face_centers:
            x, y, area, conf = face
            if abs(x - median_x) <= 2 * std_x and abs(y - median_y) <= 2 * std_y:
                filtered_faces.append(face)

        logger.info(
            f"Filtered {len(face_centers)} -> {len(filtered_faces)} faces (removed outliers)"
        )
        return (
            filtered_faces if filtered_faces else face_centers
        )  # Return original if all filtered

    except Exception as e:
        logger.warning(f"Error filtering face outliers: {e}")
        return face_centers


def count_scene_cuts(video_path: Path, threshold: float = 0.35) -> int:
    """Count likely scene cuts in a clip using ffmpeg's scene score."""
    result = run_ffmpeg_command(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-filter:v",
            f"select='gt(scene,{threshold})',showinfo",
            "-f",
            "null",
            "-",
        ],
        timeout=300,
    )
    if result.returncode != 0:
        return 0
    return len(re.findall(r"pts_time:", result.stderr))


def parse_motion_metadata(path: Path) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    values: List[float] = []
    current_time: Optional[float] = None
    for line in path.read_text(errors="ignore").splitlines():
        time_match = re.search(r"pts_time:([0-9.]+)", line)
        if time_match:
            current_time = float(time_match.group(1))
            continue
        value_match = re.search(r"lavfi\.signalstats\.YAVG=([0-9.]+)", line)
        if value_match and current_time is not None:
            times.append(current_time)
            values.append(float(value_match.group(1)))
            current_time = None
    return times, values


def smooth_values(values: List[float], window: int = 15) -> List[float]:
    if not values:
        return []
    smoothed: List[float] = []
    half = window // 2
    for idx in range(len(values)):
        start = max(0, idx - half)
        end = min(len(values), idx + half + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def build_speaker_timeline_from_motion(
    times: List[float],
    left_values: List[float],
    right_values: List[float],
    min_duration: float = 1.0,
) -> List[Dict[str, Any]]:
    if not times or len(left_values) != len(right_values):
        return []

    def normalize(values: List[float]) -> List[float]:
        mean_value = sum(values) / max(len(values), 1)
        return [value / mean_value if mean_value > 0 else 0.0 for value in values]

    left = smooth_values(normalize(left_values))
    right = smooth_values(normalize(right_values))
    if not left or not right:
        return []

    margin = 1.15
    current = 0 if left[0] >= right[0] else 1
    speakers: List[int] = []
    for left_value, right_value in zip(left, right):
        if current == 0 and right_value > left_value * margin:
            current = 1
        elif current == 1 and left_value > right_value * margin:
            current = 0
        speakers.append(current)

    segments: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(speakers):
        end_idx = idx
        while end_idx + 1 < len(speakers) and speakers[end_idx + 1] == speakers[idx]:
            end_idx += 1
        seg_start = times[idx]
        seg_end = times[min(end_idx + 1, len(times) - 1)]
        if seg_end <= seg_start:
            seg_end = seg_start + 0.05
        segments.append(
            {
                "start": seg_start,
                "end": seg_end,
                "speaker": "left" if speakers[idx] == 0 else "right",
            }
        )
        idx = end_idx + 1

    merged: List[Dict[str, Any]] = []
    for segment in segments:
        if merged and segment["end"] - segment["start"] < min_duration:
            merged[-1]["end"] = segment["end"]
            continue
        if merged and merged[-1]["speaker"] == segment["speaker"]:
            merged[-1]["end"] = segment["end"]
            continue
        merged.append(segment)
    return merged


def cluster_two_face_regions(
    face_centers: List[Tuple[int, int, int, float]],
    width: int,
    height: int,
) -> Optional[Dict[str, Dict[str, int]]]:
    """Approximate left/right face regions from sampled face centers."""
    if len(face_centers) < 2:
        return None

    median_x = float(np.median([face[0] for face in face_centers]))
    left_faces = [face for face in face_centers if face[0] <= median_x]
    right_faces = [face for face in face_centers if face[0] > median_x]
    if not left_faces or not right_faces:
        return None

    def region(faces: List[Tuple[int, int, int, float]]) -> Dict[str, int]:
        center_x = int(np.median([face[0] for face in faces]))
        center_y = int(np.median([face[1] for face in faces]))
        face_size = int(np.sqrt(max(1, float(np.median([face[2] for face in faces])))))
        roi_w = max(80, int(face_size * 1.4))
        roi_h = max(70, int(face_size * 0.9))
        roi_x = clamp_even(center_x - roi_w // 2, 0, max(0, width - roi_w))
        roi_y = clamp_even(center_y, 0, max(0, height - roi_h))
        tile_w = min(width, max(160, int(face_size * 2.8)))
        tile_h = min(height, max(160, int(face_size * 2.4)))
        tile_x = clamp_even(center_x - tile_w // 2, 0, max(0, width - tile_w))
        tile_y = clamp_even(center_y - int(tile_h * 0.42), 0, max(0, height - tile_h))
        return {
            "center_x": center_x,
            "center_y": center_y,
            "roi_x": roi_x,
            "roi_y": roi_y,
            "roi_w": round_to_even(min(roi_w, width - roi_x)),
            "roi_h": round_to_even(min(roi_h, height - roi_y)),
            "tile_x": tile_x,
            "tile_y": tile_y,
            "tile_w": round_to_even(min(tile_w, width - tile_x)),
            "tile_h": round_to_even(min(tile_h, height - tile_y)),
        }

    left = region(left_faces)
    right = region(right_faces)
    if abs(right["center_x"] - left["center_x"]) < width * 0.15:
        return None
    return {"left": left, "right": right}


def build_pan_expression(
    timeline: List[Dict[str, Any]], left_x: int, right_x: int, ramp: float = 0.45
) -> str:
    """Eased crop-x expression that glides between two speaker framings.

    Instead of snapping the crop instantly at each speaker change, this ramps
    smoothly over ``ramp`` seconds, giving a natural camera-pan feel.
    """
    if not timeline:
        return str(left_x)

    def x_for(speaker: str) -> int:
        return left_x if speaker == "left" else right_x

    keys: List[Tuple[float, float]] = [(0.0, float(x_for(timeline[0]["speaker"])))]
    for segment in timeline:
        switch_t = max(0.0, float(segment["start"]))
        target = float(x_for(segment["speaker"]))
        if abs(target - keys[-1][1]) < 1.0:
            continue
        keys.append((switch_t, keys[-1][1]))  # hold previous framing until switch
        keys.append((switch_t + ramp, target))  # then ease into the new framing

    cleaned: List[Tuple[float, int]] = []
    for t, x in keys:
        if cleaned and t <= cleaned[-1][0]:
            t = cleaned[-1][0] + 0.01
        cleaned.append((t, int(round(x))))

    if len(cleaned) < 2:
        return str(int(cleaned[0][1]) if cleaned else left_x)
    return build_smooth_pan_expression(cleaned)


# ---------------------------------------------------------------------------
# Facecam + gameplay stack ("vertical_facecam")
# ---------------------------------------------------------------------------
# Streams usually show the streamer's webcam as a small static overlay. The
# webcam goes on top (kept close to its own 16:9 shape so it is not over-zoomed)
# and the centre of the gameplay fills the rest of the 1080x1920 frame.
FACECAM_SAMPLE_FRAMES = 24
# Webcam box width as a multiple of the detected face size, used when the
# exact overlay is not configured. Kept small so the crop stays inside the
# overlay instead of pulling in gameplay pixels.
FACECAM_FACE_SCALE = float(os.getenv("FACECAM_FACE_SCALE", "3.8"))
FACECAM_MAX_FACE_FRACTION = float(os.getenv("FACECAM_MAX_FACE_FRACTION", "0.16"))
FACECAM_TOP_MIN_H = 480
FACECAM_TOP_MAX_H = 960
# Dynamic camera: cut between split, reaction (big webcam) and full gameplay
# shots following the streamer's voice/motion and the on-screen action.
FACECAM_DYNAMIC = os.getenv("FACECAM_DYNAMIC", "true").strip().lower() not in {"0", "false", "no", "off"}
FACECAM_SFX = os.getenv("FACECAM_SFX", "true").strip().lower() not in {"0", "false", "no", "off"}
FACECAM_SFX_VOLUME = float(os.getenv("FACECAM_SFX_VOLUME", "0.5"))
FACECAM_REACT_TOP_H = 1152
FACECAM_REACT_ZOOM = 1.15
FACECAM_BIN_SECONDS = 0.5
FACECAM_REACT_SECONDS = 2.5
FACECAM_REACT_RISE_DB = float(os.getenv("FACECAM_REACT_RISE_DB", "8"))
FACECAM_GAME_MIN_MOTION = float(os.getenv("FACECAM_GAME_MIN_MOTION", "8"))
FACECAM_GAME_MAX_SECONDS = 5.0
# The opening holds the split while the hook banner sits on its seam.
FACECAM_INTRO_SECONDS = float(os.getenv("FACECAM_INTRO_SECONDS", "4.0"))
HOOK_BALLOON_IMAGE = Path(os.getenv(
    "HOOK_BALLOON_IMAGE", str(Path(__file__).resolve().parents[2] / "transitions" / "hook" / "balao_hq.png")
))
SFX_DIR = Path(os.getenv("SFX_DIR", str(Path(__file__).resolve().parents[2] / "transitions" / "sfx")))
# Last webcam box found per source resolution. Clips of one task are cut from
# the same stream, so a clip where detection misses reuses its neighbour's box.
_LAST_FACECAM_BOX: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}


def _configured_facecam_box(width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    """FACECAM_REGION="x,y,w,h" as fractions of the frame (e.g. 0.78,0.09,0.21,0.22)."""
    raw = os.getenv("FACECAM_REGION", "").strip()
    if not raw or raw.lower() == "auto":
        return None
    try:
        x, y, w, h = (float(part) for part in raw.split(","))
    except ValueError:
        logger.warning("Ignoring FACECAM_REGION=%r: expected four fractions x,y,w,h", raw)
        return None
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
        logger.warning("Ignoring FACECAM_REGION=%r: values must be fractions of the frame", raw)
        return None
    bx, by = int(x * width), int(y * height)
    return bx, by, min(int(w * width), width - bx), min(int(h * height), height - by)


def find_static_facecam_box(
    clip_path: Path,
    width: int,
    height: int,
    samples: int = FACECAM_SAMPLE_FRAMES,
    min_share: float = 0.25,
) -> Optional[Tuple[int, int, int, int]]:
    """Locate a webcam overlay as the face that stays put across the clip.

    Gameplay characters move around; the streamer's face in the webcam stays
    in the same spot. Faces are sampled across the clip, grouped by position
    and size, and the group seen in the most frames wins. Groups centred in
    the middle of the frame are ignored: that is where the player character is.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None
    yunet, haar = _open_face_detectors()
    hits: List[Tuple[int, float, float, float]] = []
    frames_read = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = total / fps if total > 0 else ffprobe_duration(clip_path)
        for index in range(samples):
            cap.set(cv2.CAP_PROP_POS_MSEC, duration * (index + 0.5) / samples * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames_read += 1
            for fx, fy, fw, fh in _detect_face_boxes(frame, yunet, haar):
                hits.append((index, fx + fw / 2, fy + fh / 2, (fw + fh) / 2))
    finally:
        cap.release()
        if yunet is not None and hasattr(yunet, "close"):
            try:
                yunet.close()
            except Exception:
                pass

    def same_spot(a: Tuple[int, float, float, float], b: Tuple[int, float, float, float]) -> bool:
        return (
            abs(a[1] - b[1]) <= 0.35 * b[3]
            and abs(a[2] - b[2]) <= 0.35 * b[3]
            and 0.7 <= a[3] / max(b[3], 1.0) <= 1.4
        )

    best: List[Tuple[int, float, float, float]] = []
    best_frames = 0
    for anchor in hits:
        if abs(anchor[1] - width / 2) < 0.2 * width and abs(anchor[2] - height / 2) < 0.2 * height:
            continue
        # A webcam overlay is a small corner box. A face this big is a
        # character in a cutscene, which can hold still just as long.
        if anchor[3] > FACECAM_MAX_FACE_FRACTION * height:
            continue
        group = [hit for hit in hits if same_spot(hit, anchor)]
        frames = len({hit[0] for hit in group})
        if frames > best_frames:
            best, best_frames = group, frames
    needed = max(3, int(frames_read * min_share))
    if best_frames < needed:
        logger.info("Facecam not found: best static face in %d/%d frames", best_frames, frames_read)
        return None

    cx = float(np.median([hit[1] for hit in best]))
    cy = float(np.median([hit[2] for hit in best]))
    size = float(np.median([hit[3] for hit in best]))
    box_w = min(width, size * FACECAM_FACE_SCALE)
    box_h = min(height, box_w * 9 / 16)
    bx = int(max(0, min(width - box_w, cx - box_w / 2)))
    # Face slightly below centre leaves headroom (hair, headphones).
    by = int(max(0, min(height - box_h, cy - box_h * 0.56)))
    logger.info(
        "Facecam found at (%d,%d) %dx%d from a face seen in %d/%d frames",
        bx, by, int(box_w), int(box_h), best_frames, frames_read,
    )
    return bx, by, int(box_w), int(box_h)


_FACECAM_PLAN_CACHE: Dict[Tuple[str, int, float], Optional[Dict[str, Any]]] = {}
# Webcam box of the stream being rendered, found once over the whole source.
_STREAM_FACECAM_BOX: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
FACECAM_STREAM_SAMPLES = 72


def prime_stream_facecam(source_path: Path) -> Optional[Tuple[int, int, int, int]]:
    """Find the webcam once over the whole source and use it for every clip.

    A cutscene can hold a character's face still for a whole clip, but only the
    webcam stays in one place across hours of stream. Call before rendering a
    task's clips; the box replaces per-clip detection for that resolution.
    """
    _STREAM_FACECAM_BOX.clear()
    try:
        width, height = ffprobe_video_size(source_path)
    except Exception:
        return None
    if width / max(height, 1) <= 1.2:
        return None
    box = _configured_facecam_box(width, height) or find_static_facecam_box(
        source_path, width, height, samples=FACECAM_STREAM_SAMPLES, min_share=0.35
    )
    if box:
        _STREAM_FACECAM_BOX[(width, height)] = box
        _FACECAM_PLAN_CACHE.clear()
        logger.info("Stream webcam box: %s", box)
    else:
        logger.info("No stream-wide webcam found; clips detect their own")
    return box


def detect_facecam_plan(clip_path: Path) -> Optional[Dict[str, Any]]:
    """Webcam-on-top, gameplay-below plan for a landscape stream clip.

    Cached per file version: the caption probe and the final render ask for
    the same clip, and the activity analysis is a full decode.
    """
    try:
        stat = Path(clip_path).stat()
        key = (str(Path(clip_path).resolve()), stat.st_size, stat.st_mtime)
    except OSError:
        key = None
    if key is not None and key in _FACECAM_PLAN_CACHE:
        return copy.deepcopy(_FACECAM_PLAN_CACHE[key])
    plan = _detect_facecam_plan(clip_path)
    if key is not None:
        if len(_FACECAM_PLAN_CACHE) > 64:
            _FACECAM_PLAN_CACHE.clear()
        _FACECAM_PLAN_CACHE[key] = copy.deepcopy(plan)
    return plan


def _detect_facecam_plan(clip_path: Path) -> Optional[Dict[str, Any]]:
    width, height = ffprobe_video_size(clip_path)
    if width / max(height, 1) <= 1.2:
        return None
    key = (width, height)
    box = (
        _configured_facecam_box(width, height)
        or _STREAM_FACECAM_BOX.get(key)
        or find_static_facecam_box(clip_path, width, height)
    )
    if box:
        _LAST_FACECAM_BOX[key] = box
    else:
        box = _LAST_FACECAM_BOX.get(key)
        if box:
            logger.info("Facecam reused from a previous clip: %s", box)
    if not box:
        return None

    top_h = round_to_even(int(max(FACECAM_TOP_MIN_H, min(FACECAM_TOP_MAX_H, 1080 * box[3] / max(box[2], 1)))))
    bottom_h = 1920 - top_h
    plan: Dict[str, Any] = {
        "mode": "facecam",
        "width": width,
        "height": height,
        "box": list(box),
        "cam": _fit_crop(box, 1080 / top_h, width, height),
        "game": _game_crop(1080 / bottom_h, width, height, avoid=box),
        "top_h": top_h,
        "bottom_h": bottom_h,
        # "Reaction" shot: the webcam takes most of the frame, punched in on the face.
        "react_cam": _fit_crop(box, 1080 / FACECAM_REACT_TOP_H, width, height, zoom=FACECAM_REACT_ZOOM),
        "react_game": _game_crop(1080 / (1920 - FACECAM_REACT_TOP_H), width, height, avoid=box),
        "react_top_h": FACECAM_REACT_TOP_H,
        # "Game" shot: gameplay alone, full screen.
        "full_game": _game_crop(1080 / 1920, width, height, avoid=box),
        "segments": [],
    }
    if FACECAM_DYNAMIC:
        try:
            plan["segments"] = plan_facecam_shots(clip_path, plan)
        except Exception as exc:
            logger.warning("Dynamic facecam analysis failed (%s); keeping the static split", exc)
    return plan


def _fit_crop(
    box: Tuple[int, int, int, int] | List[int], aspect: float, width: int, height: int, zoom: float = 1.0
) -> Dict[str, int]:
    """Crop of ``box`` at ``aspect`` (w/h), centred, optionally punched in by ``zoom``."""
    bx, by, bw, bh = box
    cx, cy = bx + bw / 2, by + bh / 2
    bw, bh = bw / zoom, bh / zoom
    if bw / max(bh, 1) > aspect:
        bw = bh * aspect
    else:
        bh = bw / aspect
    w, h = round_to_even(max(2, int(bw))), round_to_even(max(2, int(bh)))
    w, h = min(w, round_to_even(width)), min(h, round_to_even(height))
    x = clamp_even(int(cx - w / 2), 0, max(0, width - w))
    y = clamp_even(int(cy - h / 2), 0, max(0, height - h))
    return {"x": x, "y": y, "w": w, "h": h}


def _game_crop(
    aspect: float, width: int, height: int, avoid: Optional[Tuple[int, int, int, int] | List[int]] = None
) -> Dict[str, int]:
    """Largest crop at ``aspect`` (w/h) near the centre that leaves out ``avoid``.

    ``avoid`` is the webcam overlay: a wide gameplay crop would otherwise show a
    second, tiny copy of the streamer in its corner.
    """

    def fit(x0: int, y0: int, x1: int, y1: int) -> Optional[Dict[str, int]]:
        aw, ah = x1 - x0, y1 - y0
        if aw <= 0 or ah <= 0:
            return None
        h = ah
        w = int(h * aspect)
        if w > aw:
            w = aw
            h = int(w / aspect)
        w, h = round_to_even(w), round_to_even(h)
        if w < 64 or h < 64:
            return None
        # As close to the frame centre as the region allows.
        x = min(max(x0, (width - w) // 2), x1 - w)
        y = min(max(y0, (height - h) // 2), y1 - h)
        return {"x": clamp_even(x, 0, max(0, width - w)), "y": clamp_even(y, 0, max(0, height - h)), "w": w, "h": h}

    full = fit(0, 0, width, height)
    if not avoid or full is None:
        return full or {"x": 0, "y": 0, "w": round_to_even(width), "h": round_to_even(height)}
    bx, by, bw, bh = avoid
    # The box is estimated from the face, so the real overlay can be a bit larger.
    pad = int(max(12, 0.12 * max(bw, bh)))
    ax0, ay0, ax1, ay1 = bx - pad, by - pad, bx + bw + pad, by + bh + pad

    def clear(c: Dict[str, int]) -> bool:
        return c["x"] + c["w"] <= ax0 or c["x"] >= ax1 or c["y"] + c["h"] <= ay0 or c["y"] >= ay1

    if clear(full):
        return full
    candidates = [
        fit(0, max(0, ay1), width, height),        # below the webcam
        fit(0, 0, width, min(height, ay0)),        # above it
        fit(0, 0, min(width, ax0), height),        # left of it
        fit(max(0, ax1), 0, width, height),        # right of it
    ]
    candidates = [c for c in candidates if c and clear(c)]
    if not candidates:
        return full
    best = max(candidates, key=lambda c: c["w"] * c["h"])
    # Only worth it while most of the picture survives; otherwise accept the overlay.
    return best if best["w"] * best["h"] >= 0.45 * full["w"] * full["h"] else full


def _parse_metadata_series(path: Path, key: str) -> Tuple[List[float], List[float]]:
    """(times, values) for one ``metadata=mode=print`` key."""
    times: List[float] = []
    values: List[float] = []
    current: Optional[float] = None
    pattern = re.compile(re.escape(key) + r"=(-?[0-9.]+|-?inf)")
    try:
        text = path.read_text(errors="ignore")
    except FileNotFoundError:
        return times, values
    for line in text.splitlines():
        found = re.search(r"pts_time:([0-9.]+)", line)
        if found:
            current = float(found.group(1))
            continue
        found = pattern.search(line)
        if found and current is not None:
            raw = found.group(1)
            times.append(current)
            values.append(-70.0 if "inf" in raw else float(raw))
            current = None
    return times, values


def _bin_series(times: List[float], values: List[float], duration: float, step: float) -> List[float]:
    bins = max(1, int(math.ceil(duration / step)))
    sums = [0.0] * bins
    counts = [0] * bins
    for t, v in zip(times, values):
        i = min(bins - 1, max(0, int(t / step)))
        sums[i] += v
        counts[i] += 1
    out: List[float] = []
    last = 0.0
    for total, count in zip(sums, counts):
        last = total / count if count else last
        out.append(last)
    return out


def _zscores(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    spread = float(np.median(np.abs(arr - med))) * 1.4826 or float(np.std(arr)) or 1.0
    return [float(v) for v in (arr - med) / spread]


def measure_facecam_activity(clip_path: Path, plan: Dict[str, Any]) -> Dict[str, List[float]]:
    """Loudness, webcam motion and gameplay motion in FACECAM_BIN_SECONDS bins."""
    duration = ffprobe_duration(clip_path)
    scale = 480 / plan["width"]

    def scaled(crop: Dict[str, int]) -> str:
        w = max(8, round_to_even(int(crop["w"] * scale)))
        h = max(8, round_to_even(int(crop["h"] * scale)))
        x = round_to_even(int(crop["x"] * scale))
        y = round_to_even(int(crop["y"] * scale))
        return f"{w}:{h}:{x}:{y}"

    with tempfile.TemporaryDirectory(prefix="supoclip_facecam_") as tmp:
        cam_file, game_file, loud_file = (Path(tmp) / n for n in ("cam.txt", "game.txt", "loud.txt"))
        graph = (
            "[0:v]fps=10,scale=480:-2,split=2[a][b];"
            f"[a]crop={scaled(plan['cam'])},format=gray,tblend=all_mode=difference,signalstats,"
            f"metadata=mode=print:key=lavfi.signalstats.YAVG:file={ffmpeg_escape_filter_path(cam_file)}[ao];"
            f"[b]crop={scaled(plan['full_game'])},format=gray,tblend=all_mode=difference,signalstats,"
            f"metadata=mode=print:key=lavfi.signalstats.YAVG:file={ffmpeg_escape_filter_path(game_file)}[bo]"
        )
        maps = ["-map", "[ao]", "-f", "null", "-", "-map", "[bo]", "-f", "null", "-"]
        has_audio = ffprobe_has_audio(clip_path)
        if has_audio:
            graph += (
                ";[0:a]ebur128=metadata=1,"
                f"ametadata=mode=print:key=lavfi.r128.M:file={ffmpeg_escape_filter_path(loud_file)}[lo]"
            )
            maps += ["-map", "[lo]", "-f", "null", "-"]
        result = run_ffmpeg_command(["ffmpeg", "-y", "-i", str(clip_path), "-filter_complex", graph, *maps], timeout=600)
        if result.returncode != 0:
            raise RuntimeError("activity analysis ffmpeg pass failed")
        step = FACECAM_BIN_SECONDS
        cam = _bin_series(*_parse_metadata_series(cam_file, "lavfi.signalstats.YAVG"), duration, step)
        game = _bin_series(*_parse_metadata_series(game_file, "lavfi.signalstats.YAVG"), duration, step)
        loud = (
            _bin_series(*_parse_metadata_series(loud_file, "lavfi.r128.M"), duration, step)
            if has_audio
            else [-70.0] * len(cam)
        )
    n = min(len(cam), len(game), len(loud))
    return {"duration": [duration], "cam": cam[:n], "game": game[:n], "loud": loud[:n]}


def plan_facecam_shots(clip_path: Path, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Shot list (start, end, kind) cutting between split, reaction and game shots.

    * reaction: the streamer gets loud or moves a lot (scream, laugh, jump scare)
    * game: busy gameplay while the streamer is fairly quiet
    * split: everything else
    Short shots are merged away so the cuts stay readable.
    """
    activity = measure_facecam_activity(clip_path, plan)
    duration = activity["duration"][0]
    loud, cam, game = activity["loud"], activity["cam"], activity["game"]
    if len(loud) < 4:
        return [{"start": 0.0, "end": duration, "kind": "split"}]
    step = FACECAM_BIN_SECONDS
    n = len(loud)
    loud_med = float(np.median(loud))
    z_cam = _zscores(cam)
    labels = ["split"] * n

    # Reaction = a sudden jump in the streamer's voice (scream, laugh, scare),
    # not loud talking in general: compare with the second before it.
    react_len = int(round(FACECAM_REACT_SECONDS / step))
    lead = int(round(0.5 / step))
    i = 2
    while i < n:
        before = float(np.mean(loud[max(0, i - 3):i - 1])) if i >= 2 else loud_med
        rise = loud[i] - before
        if (rise >= FACECAM_REACT_RISE_DB and loud[i] >= loud_med + 8.0) or (
            rise >= FACECAM_REACT_RISE_DB - 3 and loud[i] >= loud_med + 6.0 and z_cam[i] >= 2.0
        ):
            for j in range(max(0, i - lead), min(n, i - lead + react_len)):
                labels[j] = "react"
            i += react_len + int(round(2.0 / step))   # breathing room before the next one
            continue
        i += 1

    # Game = intense, sustained action on screen outside reactions.
    window = int(round(2.0 / step))
    smooth = [float(np.mean(game[max(0, k - window // 2):k + window // 2 + 1])) for k in range(n)]
    floor = max(FACECAM_GAME_MIN_MOTION, float(np.percentile(smooth, 65)))
    max_game = int(round(FACECAM_GAME_MAX_SECONDS / step))
    run = cooldown = 0
    for k in range(n):
        if cooldown:
            # After a capped game shot, show the split again for a while.
            cooldown -= 1
            run = 0
            continue
        if labels[k] == "split" and smooth[k] >= floor:
            labels[k] = "game"
            run += 1
            if run >= max_game:
                run, cooldown = 0, int(round(2.5 / step))
        else:
            run = 0

    # Variety on calm clips: give the strongest gameplay moment its own shot.
    if duration >= 12 and all(label == "split" for label in labels):
        width = int(round(3.0 / step))
        best = max(range(n - width + 1), key=lambda k: sum(game[k:k + width]), default=None)
        if best is not None and sum(game[best:best + width]) / width >= FACECAM_GAME_MIN_MOTION / 2:
            for j in range(best, best + width):
                labels[j] = "game"

    for j in range(min(n, int(round(FACECAM_INTRO_SECONDS / step)))):
        labels[j] = "split"

    runs: List[List[Any]] = []
    for i, label in enumerate(labels):
        if runs and runs[-1][0] == label:
            runs[-1][2] = i + 1
        else:
            runs.append([label, i, i + 1])
    min_bins = {"react": int(round(1.5 / step)), "game": int(round(2.0 / step)), "split": int(round(2.0 / step))}
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for idx, (label, a, b) in enumerate(runs):
            if b - a >= min_bins[label]:
                continue
            if idx == 0:
                target = 1
            elif idx == len(runs) - 1:
                target = idx - 1
            else:
                target = idx - 1 if runs[idx - 1][2] - runs[idx - 1][1] >= runs[idx + 1][2] - runs[idx + 1][1] else idx + 1
            if target < idx:
                runs[target][2] = b
            else:
                runs[target][1] = a
            del runs[idx]
            merged: List[List[Any]] = []
            for run in runs:
                if merged and merged[-1][0] == run[0]:
                    merged[-1][2] = run[2]
                else:
                    merged.append(run)
            runs = merged
            changed = True
            break

    shots = [
        {"start": round(a * step, 3), "end": round(min(duration, b * step), 3), "kind": label}
        for label, a, b in runs
    ]
    shots[0]["start"] = 0.0
    shots[-1]["end"] = round(duration, 3)
    logger.info("Facecam shots: %s", [(s["kind"], s["start"], s["end"]) for s in shots])
    return shots


def facecam_hook_center_y(plan: Dict[str, Any]) -> float:
    """Output fraction of the webcam/gameplay seam in the opening shot."""
    shots = plan.get("segments") or []
    kind = shots[0]["kind"] if shots else "split"
    if kind == "react":
        return plan["react_top_h"] / 1920
    if kind == "game":
        return 0.32
    return plan["top_h"] / 1920


def build_hook_balloon_filter(base: str, geometry: Dict[str, Any], out: str) -> str:
    """Overlay the comic banner image behind the hook (``base`` -> ``out`` labels).

    The banner pops in (0 -> 112% -> 97% -> 100%, like the text) and fades out
    with it. The image stream is trimmed at the end so it costs nothing after.
    """
    w, h = round_to_even(int(geometry["w"])), round_to_even(int(geometry["h"]))
    cx, cy = int(geometry["x"]), int(geometry["y"])
    start, end = float(geometry["start"]), float(geometry["end"])
    rel = f"(t-{start:.3f})"
    pop = (
        f"if(lt({rel},0.13),max(0.02,1.12*{rel}/0.13),"
        f"if(lt({rel},0.23),1.12-0.15*({rel}-0.13)/0.10,"
        f"if(lt({rel},0.30),0.97+0.03*({rel}-0.23)/0.07,1)))"
    )
    image = ffmpeg_escape_filter_path(HOOK_BALLOON_IMAGE)
    return (
        f"movie={image}:loop=0,setpts=N/30/TB,trim=end={end + 0.05:.3f},format=rgba,"
        f"scale=w='max(2,trunc({w}*{pop}/2)*2)':h='max(2,trunc({h}*{pop}/2)*2)':eval=frame,"
        f"fade=t=out:st={max(start, end - 0.2):.3f}:d=0.2:alpha=1[hookbal];"
        f"[{base}][hookbal]overlay=x='{cx}-w/2':y='{cy}-h/2':eval=frame:eof_action=pass:"
        f"enable='between(t,{start:.3f},{end:.3f})'[{out}]"
    )


def _load_hook_balloon(ass_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Banner geometry the caption builder left beside the .ass, if any."""
    if not ass_path or not HOOK_BALLOON_IMAGE.is_file():
        return None
    from .captions import hook_balloon_geometry_path
    try:
        geometry = json.loads(hook_balloon_geometry_path(Path(ass_path)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    required = {"x", "y", "w", "h", "start", "end"}
    return geometry if isinstance(geometry, dict) and required <= geometry.keys() else None


def _crop(c: Dict[str, int]) -> str:
    return f"crop={c['w']}:{c['h']}:{c['x']}:{c['y']}"


def _facecam_shot_chain(plan: Dict[str, Any], kind: str, src: str, out: str, i: int) -> str:
    """One shot of the dynamic layout, from ``src`` to a 1080x1920 ``out``."""
    tail = "format=yuv420p,setsar=1"
    if kind == "game":
        g = plan["full_game"]
        return f"[{src}]{_crop(g)},scale=1080:1920:flags=lanczos,{tail}[{out}]"
    if kind == "react":
        cam, game, top = plan["react_cam"], plan["react_game"], plan["react_top_h"]
    else:
        cam, game, top = plan["cam"], plan["game"], plan["top_h"]
    return (
        f"[{src}]split=2[c{i}][g{i}];"
        f"[c{i}]{_crop(cam)},scale=1080:{top}:flags=lanczos,{UPSCALE_SHARPEN_FILTER},setsar=1[ct{i}];"
        f"[g{i}]{_crop(game)},scale=1080:{1920 - top}:flags=lanczos,setsar=1[gt{i}];"
        f"[ct{i}][gt{i}]vstack,{tail}[{out}]"
    )


def build_facecam_stack_filter(plan: Dict[str, Any]) -> str:
    """Filter graph (no output label): webcam over gameplay, cutting between shots."""
    shots = plan.get("segments") or []
    if len(shots) <= 1:
        kind = shots[0]["kind"] if shots else "split"
        return _facecam_shot_chain(plan, kind, "0:v", "fc", 0).rsplit("[fc]", 1)[0]
    labels = "".join(f"[s{i}]" for i in range(len(shots)))
    parts = [f"[0:v]split={len(shots)}{labels}"]
    for i, shot in enumerate(shots):
        end = "" if i == len(shots) - 1 else f":end={shot['end']:.3f}"
        parts.append(f"[s{i}]trim=start={shot['start']:.3f}{end},setpts=PTS-STARTPTS[t{i}]")
        parts.append(_facecam_shot_chain(plan, shot["kind"], f"t{i}", f"v{i}", i))
    parts.append("".join(f"[v{i}]" for i in range(len(shots))) + f"concat=n={len(shots)}:v=1:a=0")
    return ";".join(parts)


def mix_transition_sfx(clip_path: Path, shots: List[Dict[str, Any]]) -> Optional[Path]:
    """Copy of the clip with a whoosh/impact under each cut (video stream copied).

    Returns a temporary .mkv (lossless PCM audio) or None when there is nothing
    to add or the mix fails. The caller deletes its parent directory.
    """
    if not FACECAM_SFX or len(shots) <= 1:
        return None
    events: List[Tuple[Path, float]] = []
    for prev, shot in zip(shots, shots[1:]):
        cut = shot["start"]
        if shot["kind"] == "react":
            events.append((SFX_DIR / "impact.wav", max(0.0, cut - 0.03)))
            events.append((SFX_DIR / "swish.wav", max(0.0, cut - 0.18)))
        else:
            # The whoosh peaks ~0.34 s in: line the peak up with the cut.
            events.append((SFX_DIR / "whoosh.wav", max(0.0, cut - 0.34)))
    events = [(path, at) for path, at in events if path.is_file()]
    if not events:
        logger.info("No SFX files in %s; cutting without sound effects", SFX_DIR)
        return None
    tmp = Path(tempfile.mkdtemp(prefix="supoclip_sfx_"))
    out = tmp / "with_sfx.mkv"
    inputs: List[str] = []
    chains: List[str] = []
    for k, (path, at) in enumerate(events, start=1):
        inputs += ["-i", str(path)]
        ms = int(at * 1000)
        chains.append(f"[{k}:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay={ms}|{ms},volume={FACECAM_SFX_VOLUME}[x{k}]")
    mix = "".join(f"[x{k}]" for k in range(1, len(events) + 1))
    graph = (
        "[0:a]aformat=sample_rates=48000:channel_layouts=stereo[main];"
        + ";".join(chains)
        + f";[main]{mix}amix=inputs={len(events) + 1}:duration=first:normalize=0[a]"
    )
    command = [
        "ffmpeg", "-y", "-i", str(clip_path), *inputs,
        "-filter_complex", graph,
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "pcm_s16le",
        str(out),
    ]
    if run_ffmpeg_command(command, timeout=600).returncode != 0 or not out.is_file():
        shutil.rmtree(tmp, ignore_errors=True)
        logger.warning("SFX mix failed; rendering without sound effects")
        return None
    logger.info("Added %d transition SFX", len(events))
    return out


def detect_speaker_reframe_plan(
    clip_path: Path,
    output_format: str,
) -> Optional[Dict[str, Any]]:
    """Build a speaker-aware pan or split-screen plan for a trimmed clip."""
    if output_format == "vertical_facecam":
        try:
            return detect_facecam_plan(clip_path)
        except Exception as exc:
            logger.warning("Facecam layout failed (%s); using the default layout", exc)
            return None
    try:
        width, height = ffprobe_video_size(clip_path)
        if width / max(height, 1) <= 1.2:
            return None

        scene_cuts = count_scene_cuts(clip_path)
        if scene_cuts > 2:
            logger.info("Skipping speaker reframe: %d scene cuts detected", scene_cuts)
            return None

        duration = ffprobe_duration(clip_path)
        face_centers = detect_faces_in_clip(clip_path, 0, min(duration, 12.0))
        regions = cluster_two_face_regions(face_centers, width, height)
        if not regions:
            return None

        crop_w = round_to_even(min(width, int(height * 9 / 16)))
        left_x = clamp_even(
            regions["left"]["center_x"] - crop_w // 2,
            0,
            max(0, width - crop_w),
        )
        right_x = clamp_even(
            regions["right"]["center_x"] - crop_w // 2,
            0,
            max(0, width - crop_w),
        )

        if output_format == "vertical_split":
            return {
                "mode": "split",
                "width": width,
                "height": height,
                "regions": regions,
            }

        with tempfile.TemporaryDirectory(prefix="supoclip_motion_") as motion_dir:
            left_motion = Path(motion_dir) / "left.txt"
            right_motion = Path(motion_dir) / "right.txt"
            left = regions["left"]
            right = regions["right"]
            filter_complex = (
                f"[0:v]split=2[l][r];"
                f"[l]crop={left['roi_w']}:{left['roi_h']}:{left['roi_x']}:{left['roi_y']},"
                f"format=gray,tblend=all_mode=difference,signalstats,"
                f"metadata=mode=print:key=lavfi.signalstats.YAVG:file={ffmpeg_escape_filter_path(left_motion)}[lo];"
                f"[r]crop={right['roi_w']}:{right['roi_h']}:{right['roi_x']}:{right['roi_y']},"
                f"format=gray,tblend=all_mode=difference,signalstats,"
                f"metadata=mode=print:key=lavfi.signalstats.YAVG:file={ffmpeg_escape_filter_path(right_motion)}[ro]"
            )
            result = run_ffmpeg_command(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(clip_path),
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[lo]",
                    "-f",
                    "null",
                    "-",
                    "-map",
                    "[ro]",
                    "-f",
                    "null",
                    "-",
                ],
                timeout=300,
            )
            if result.returncode != 0:
                return None
            times, left_values = parse_motion_metadata(left_motion)
            _, right_values = parse_motion_metadata(right_motion)
            timeline = build_speaker_timeline_from_motion(
                times,
                left_values,
                right_values,
            )
            if len(timeline) < 2:
                return None

        return {
            "mode": "pan",
            "width": width,
            "height": height,
            "crop_w": crop_w,
            "crop_h": height,
            "x_expression": build_pan_expression(timeline, left_x, right_x),
            "timeline": timeline,
        }
    except Exception as exc:
        logger.warning("Speaker reframe planning failed: %s", exc)
        return None


def compute_vertical_crop_dims(
    width: int, height: int, target_ratio: float = 9 / 16
) -> Tuple[int, int]:
    """Even-dimensioned 9:16 crop box that fits inside a source frame."""
    if width <= 0 or height <= 0:
        return width, height
    if width / height > target_ratio:
        crop_w = round_to_even(int(height * target_ratio))
        crop_h = round_to_even(height)
    else:
        crop_w = round_to_even(width)
        crop_h = round_to_even(int(width / target_ratio))
    return (
        max(2, min(crop_w, round_to_even(width))),
        max(2, min(crop_h, round_to_even(height))),
    )


def _open_face_detectors():
    """Initialise the YuNet (preferred) + Haar (fallback) face detectors.

    Returns (face_detector_or_None, haar). The YuNet wrapper also exposes a
    MediaPipe-style ``process``/``close`` so older call sites keep working.
    """
    return open_yunet_detector(), open_haar_cascade()


def _detect_face_boxes(frame_bgr, face_detector, haar) -> List[Tuple[float, float, float, float]]:
    """All face boxes (x, y, w, h) in a frame, largest first. Frame-pixel coordinates."""
    h, w = frame_bgr.shape[:2]
    frame_area = float(max(1, w * h))
    boxes: List[Tuple[float, float, float, float, float]] = []

    if face_detector is not None:
        try:
            for fx, fy, fw, fh, conf in face_detector.detect(frame_bgr):
                if fw > 10 and fh > 10:
                    boxes.append((float(fw * fh * conf), float(fx), float(fy), float(fw), float(fh)))
        except Exception as exc:
            logger.debug("YuNet detection failed on frame: %s", exc)

    if not boxes:
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            min_side = max(14, int(w * 0.04))
            for (fx, fy, fw, fh) in haar.detectMultiScale(
                gray,
                scaleFactor=1.2,  # coarser scale steps -> ~2x faster
                minNeighbors=3,
                minSize=(min_side, min_side),
                maxSize=(int(w * 0.7), int(h * 0.7)),
            ):
                boxes.append((float(fw * fh), float(fx), float(fy), float(fw), float(fh)))
        except Exception:
            pass

    boxes.sort(key=lambda item: item[0], reverse=True)
    kept: List[Tuple[float, float, float, float]] = []
    for _, fx, fy, fw, fh in boxes:
        if (fw * fh) / frame_area < FACE_PRESENCE_MIN_AREA:
            continue
        cx, cy = fx + fw / 2, fy + fh / 2
        if any(k[0] <= cx <= k[0] + k[2] and k[1] <= cy <= k[1] + k[3] for k in kept):
            continue
        kept.append((fx, fy, fw, fh))
        if len(kept) >= 6:
            break
    return kept


def _detect_dominant_face(frame_bgr, face_detector, haar) -> Optional[Tuple[float, float]]:
    """Return (center_x_fraction, area_fraction) of the dominant face, or None."""
    h, w = frame_bgr.shape[:2]
    frame_area = float(max(1, w * h))
    boxes = _detect_face_boxes(frame_bgr, face_detector, haar)
    if not boxes:
        return None
    fx, fy, fw, fh = max(boxes, key=lambda box: box[2] * box[3])
    return (fx + fw / 2.0) / w, (fw * fh) / frame_area


def _scene_cuts_from_diffs(diffs: List[Tuple[float, float]]) -> List[float]:
    """Derive scene-cut timestamps from per-frame difference spikes."""
    if len(diffs) < 3:
        return []
    vals = [d for _, d in diffs]
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
    threshold = max(14.0, mean + 3.5 * std)
    cuts: List[float] = []
    for i, (t, d) in enumerate(diffs):
        if d <= threshold:
            continue
        if (i == 0 or d >= diffs[i - 1][1]) and (
            i == len(diffs) - 1 or d >= diffs[i + 1][1]
        ):
            cuts.append(t)
    return cuts


def analyze_vertical_clip(
    input_path: Path,
    *,
    sample_fps: float = 3.0,
    proc_width: int = 480,
    analyze_context: bool = True,
) -> Tuple[
    List[Tuple[float, Optional[float], float]],
    List[float],
    List[Tuple[float, List[Tuple[float, float, float, float]]]],
    List[SceneEvidence],
]:
    """Fast single-pass clip analysis: face track + scene cuts in one decode.

    Replaces slow per-sample random seeks (and a separate scene-detect pass) with
    a single sequential ffmpeg decode at low fps/resolution, piped straight into
    lightweight face detection. Scene cuts come from frame differences computed
    in the same pass. Returns (track, scene_cuts, face_samples, context) where track
    entries are (t, center_x_in_source_px or None, area_frac) and face_samples
    are (t, [(x, y, w, h), ...]) in source pixels.

    ``analyze_context=False`` skips optical flow, text components, and histograms
    and returns an empty context list. The reaction detail pass only needs the
    finer face boxes, so it uses that mode.
    """
    if not np.isfinite(sample_fps) or sample_fps <= 0 or proc_width < 2:
        raise ValueError("Analysis requires positive sample_fps and proc_width >= 2")
    width, height = ffprobe_video_size(input_path)
    if width <= 0 or height <= 0:
        return [], [], [], []
    proc_w = round_to_even(min(proc_width, width))
    proc_h = round_to_even(max(2, int(round(proc_w * height / width))))
    frame_bytes = proc_w * proc_h * 3

    command = [
        "ffmpeg", "-v", "error", "-an", "-sn",
        "-i", str(input_path),
        "-vf", f"fps={sample_fps:.3f},scale={proc_w}:{proc_h}",
        "-pix_fmt", "bgr24", "-f", "rawvideo",
        "-threads", "0", "-",
    ]
    try:
        proc = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
    except Exception as exc:
        logger.warning("analyze_vertical_clip: ffmpeg spawn failed (%s)", exc)
        return [], [], [], []

    face_detector = None
    track: List[Tuple[float, Optional[float], float]] = []
    face_samples: List[Tuple[float, List[Tuple[float, float, float, float]]]] = []
    diffs: List[Tuple[float, float]] = []
    context_analyzer = ContextAnalyzer() if analyze_context else None
    context: List[SceneEvidence] = []
    prev_small = None
    idx = 0
    scale_x = width / proc_w
    scale_y = height / proc_h
    try:
        face_detector, haar = _open_face_detectors()
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) < frame_bytes:
                raise RuntimeError("Video analysis decoder returned an incomplete frame")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(proc_h, proc_w, 3)
            t = idx / sample_fps
            boxes = _detect_face_boxes(frame, face_detector, haar)
            if context_analyzer is not None:
                context.append(context_analyzer.measure(frame, boxes, t))
            if not boxes:
                track.append((t, None, 0.0))
                face_samples.append((t, []))
            else:
                fx, fy, fw, fh = max(boxes, key=lambda box: box[2] * box[3])
                area = (fw * fh) / float(proc_w * proc_h)
                track.append((t, (fx + fw / 2.0) * scale_x, area))
                face_samples.append(
                    (t, [(bx * scale_x, by * scale_y, bw * scale_x, bh * scale_y) for bx, by, bw, bh in boxes])
                )
            small = cv2.resize(frame, (32, 18)).astype(np.int16)
            if prev_small is not None:
                diffs.append((t, float(np.mean(np.abs(small - prev_small)))))
            prev_small = small
            idx += 1
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
        if face_detector is not None:
            face_detector.close()

    if proc.returncode != 0:
        raise RuntimeError("Video analysis decoder failed; partial analysis discarded")
    cuts = sorted(set(_scene_cuts_from_diffs(diffs)) | set(context_scene_cuts(context)))
    return track, cuts, face_samples, context


def _median_filter(values: List[float], window: int = 3) -> List[float]:
    """Small median filter to remove single-frame detection spikes."""
    if window <= 1 or len(values) < window:
        return list(values)
    half = window // 2
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        seg = sorted(values[lo:hi])
        out.append(seg[len(seg) // 2])
    return out


def build_crop_trajectory(
    track: List[Tuple[float, Optional[float], float]],
    width: int,
    crop_w: int,
    *,
    deadzone_frac: float = 0.05,
    smooth_time: float = 0.9,
    max_pan_speed_frac: float = 0.4,
) -> List[Tuple[float, int]]:
    """Turn a raw face-centre track into a smooth, eased crop-x trajectory.

    Returns keyframes [(t, x)] for the crop's left edge. The motion is produced
    by a critically-damped spring (Unity-style SmoothDamp) easing toward a
    comfort-zone target, which gives natural ease-in/ease-out with no overshoot
    and no mechanical ramp-then-stop feel. A deadzone keeps the frame still for
    small head movements; a median pre-filter removes detection spikes. Returns
    [] when there isn't enough signal to track.
    """
    if not track:
        return []
    max_x = max(0, width - crop_w)
    if max_x <= 0:
        return []

    centers: List[Optional[float]] = [c for _, c, _ in track]
    times = [t for t, _, _ in track]
    detected = sum(1 for c in centers if c is not None)
    if detected < max(3, len(centers) // 5):
        return []  # too sparse to trust — caller falls back to a static crop

    # Gap-fill missing detections: forward fill, then back fill.
    last: Optional[float] = None
    for i in range(len(centers)):
        if centers[i] is None:
            centers[i] = last
        else:
            last = centers[i]
    last = None
    for i in range(len(centers) - 1, -1, -1):
        if centers[i] is None:
            centers[i] = last
        else:
            last = centers[i]
    if any(c is None for c in centers):
        return []

    desired = [min(max(c - crop_w / 2.0, 0.0), float(max_x)) for c in centers]
    desired = _median_filter(desired, window=3)

    deadzone = max(2.0, crop_w * deadzone_frac)
    max_speed = max(1.0, width * max_pan_speed_frac)
    smooth_time = max(0.05, smooth_time)
    omega = 2.0 / smooth_time

    # Comfort-zone target: a stable goal that only moves once the subject drifts
    # past the deadzone, so the spring isn't chasing sub-deadzone jitter.
    targets: List[float] = []
    anchor = desired[0]
    for d in desired:
        if d - anchor > deadzone:
            anchor = d - deadzone
        elif anchor - d > deadzone:
            anchor = d + deadzone
        targets.append(anchor)

    # Critically-damped spring toward the comfort-zone target.
    eased: List[float] = []
    cur = float(targets[0])
    vel = 0.0
    for i, tgt in enumerate(targets):
        dt = (times[i] - times[i - 1]) if i > 0 else 0.0
        if dt <= 0:
            eased.append(cur)
            continue
        x = omega * dt
        exp_factor = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        change = cur - tgt
        max_change = max_speed * smooth_time
        change = max(-max_change, min(change, max_change))
        adj_target = cur - change
        temp = (vel + omega * change) * dt
        vel = (vel - omega * temp) * exp_factor
        out = adj_target + (change + temp) * exp_factor
        # Prevent overshoot past the target.
        if (tgt - cur > 0) == (out > tgt):
            out = tgt
            vel = (out - tgt) / dt
        cur = min(max(out, 0.0), float(max_x))
        eased.append(cur)

    # Final low-pass pass: removes residual velocity steps so the piecewise-
    # linear keyframes read as continuous, fluid motion.
    eased = smooth_values(eased, window=5)

    # Keep keyframes fine enough that linear interpolation tracks the smooth
    # curve without visible faceting.
    def simplify(tol: float) -> List[Tuple[float, int]]:
        keys: List[Tuple[float, int]] = [(0.0, int(round(eased[0])))]
        for i in range(1, len(eased)):
            if abs(eased[i] - keys[-1][1]) >= tol:
                keys.append((times[i], int(round(eased[i]))))
        if keys[-1][0] < times[-1]:
            keys.append((times[-1], int(round(eased[-1]))))
        return keys

    tol = max(1.5, crop_w * 0.006)
    keys = simplify(tol)
    while len(keys) > 90:
        tol *= 1.5
        keys = simplify(tol)

    if keys and keys[0][0] > 0.0:
        keys[0] = (0.0, keys[0][1])
    return keys


def trajectory_has_movement(keys: List[Tuple[float, int]], crop_w: int) -> bool:
    """Whether a trajectory pans enough to be worth a moving crop."""
    if len(keys) < 2:
        return False
    xs = [x for _, x in keys]
    return (max(xs) - min(xs)) >= max(8, crop_w * 0.04)


def build_smooth_pan_expression(keys: List[Tuple[float, int]]) -> str:
    """Piecewise-linear ffmpeg crop-x expression interpolating the keyframes.

    Commas are escaped for use inside a quoted filtergraph expression. The
    result is rounded to an even integer for clean chroma subsampling.
    """
    if not keys:
        return "0"
    if len(keys) == 1:
        return str(int(keys[0][1]))

    expr = str(int(keys[-1][1]))
    for i in range(len(keys) - 2, -1, -1):
        t0, x0 = keys[i]
        t1, x1 = keys[i + 1]
        span = max(1e-3, t1 - t0)
        lerp = f"({int(x0)}+({int(x1) - int(x0)})*(t-{t0:.3f})/{span:.3f})"
        expr = f"if(lt(t\\,{t1:.3f})\\,{lerp}\\,{expr})"
    return f"trunc(({expr})/2)*2"


def build_layout_plan(
    track: List[Tuple[float, Optional[float], float]],
    scene_cuts: List[float],
    duration: float,
) -> List[Dict[str, Any]]:
    """Classify a clip into 'face' (tracked crop) and 'fit' (full-frame) shots.

    Talking-head shots become a tracked crop; content shots (tweets, graphs,
    code — no real face) become a full-frame blurred-background fit so nothing
    is cropped off. Boundaries are debounced and snapped to scene cuts.
    """
    if duration <= 0 or not track:
        return [{"start": 0.0, "end": max(0.0, duration), "kind": "fit"}]

    times = [t for t, _, _ in track]
    present = [
        1 if (c is not None and a >= FACE_PRESENCE_MIN_AREA) else 0
        for _, c, a in track
    ]

    # Classify by face-presence RATE over a window: a real talking shot has a
    # face in many frames (even if small/spotty); a content shot has ~none.
    diffs = [times[i] - times[i - 1] for i in range(1, len(times))]
    dt = sorted(diffs)[len(diffs) // 2] if diffs else 0.25
    half = max(1, int(round(FACE_RATE_WINDOW / 2.0 / max(dt, 0.05))))
    smoothed: List[int] = []
    for i in range(len(present)):
        seg = present[max(0, i - half) : min(len(present), i + half + 1)]
        rate = sum(seg) / len(seg)
        smoothed.append(1 if rate >= FACE_PRESENCE_RATE else 0)

    # Build runs of constant layout value.
    runs: List[List[float]] = []
    run_start, run_val = 0.0, smoothed[0]
    for i in range(1, len(times)):
        if smoothed[i] != run_val:
            runs.append([run_start, times[i], run_val])
            run_start, run_val = times[i], smoothed[i]
    runs.append([run_start, duration, run_val])

    def coalesce(rs: List[List[float]]) -> List[List[float]]:
        out = [rs[0][:]]
        for r in rs[1:]:
            if r[2] == out[-1][2]:
                out[-1][1] = r[1]
            else:
                out.append(r[:])
        return out

    # Merge any run shorter than the minimum layout duration (flip + coalesce).
    runs = coalesce(runs)
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for r in runs:
            if r[1] - r[0] < MIN_LAYOUT_SECONDS:
                r[2] = 1 - r[2]
                changed = True
                break
        if changed:
            runs = coalesce(runs)

    # Snap internal boundaries to nearby scene cuts for clean switches.
    cuts = sorted(c for c in (scene_cuts or []) if 0.05 < c < duration - 0.05)
    for i in range(len(runs) - 1):
        boundary = runs[i][1]
        near = [c for c in cuts if abs(c - boundary) <= LAYOUT_SNAP_WINDOW]
        if not near:
            continue
        snapped = min(near, key=lambda c: abs(c - boundary))
        if runs[i][0] + 0.1 < snapped < runs[i + 1][1] - 0.1:
            runs[i][1] = snapped
            runs[i + 1][0] = snapped

    return [
        {"start": r[0], "end": r[1], "kind": "face" if r[2] == 1 else "fit"}
        for r in runs
    ]


def build_scene_crop_expression(
    track: List[Tuple[float, Optional[float], float]],
    scene_cuts: List[float],
    duration: float,
    width: int,
    crop_w: int,
) -> str:
    """Track each camera shot independently, with a hard reset at every cut."""
    boundaries = [0.0, *sorted({c for c in scene_cuts if 0 < c < duration}), duration]
    shots = []
    for start, end in zip(boundaries, boundaries[1:]):
        shot_track = [(t, x, area) for t, x, area in track if start <= t < end]
        keys = build_crop_trajectory(shot_track, width, crop_w)
        if keys:
            # Hold the first/last detected positions to the shot boundaries;
            # never extrapolate an initial movement backwards over a cut.
            keys = [(start, keys[0][1]), *[(t, x) for t, x in keys if start < t < end], (end, keys[-1][1])]
            expression = build_smooth_pan_expression(keys)
        else:
            centers = [x for _, x, _ in shot_track if x is not None]
            center = float(np.median(centers)) if centers else width / 2
            expression = str(clamp_even(int(center - crop_w / 2), 0, width - crop_w))
        shots.append((end, expression))
    result = shots[-1][1]
    for end, expression in reversed(shots[:-1]):
        result = f"if(lt(t\\,{end:.3f})\\,{expression}\\,{result})"
    return result


def kenburns_zoom_fragment(duration: float) -> Optional[str]:
    """Slow linear punch-in fragment producing a 1080x1920 [setsar-ed] stream.

    Zooms through a 1.5x supersampled frame so zoompan's integer crop offsets
    stay sub-pixel in the output (no visible stepping). The frame rate is
    normalised to OUTPUT_FPS *before* zoompan and re-declared on it, because
    zoompan re-times its output at its own fps — matching the two keeps the
    video duration identical and the audio in sync.
    """
    if duration < KENBURNS_MIN_SECONDS:
        return None
    z_expr = (
        f"if(isnan(it)\\,1\\,1+{KENBURNS_ZOOM_DELTA}*min(it/{duration:.3f}\\,1))"
    )
    return (
        f"scale={KENBURNS_SUPERSAMPLE_W}:{KENBURNS_SUPERSAMPLE_H}:flags=lanczos,"
        f"fps={OUTPUT_FPS},"
        f"zoompan=z='{z_expr}':x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*{KENBURNS_FOCUS_Y}'"
        f":d=1:s=1080x1920:fps={OUTPUT_FPS},{UPSCALE_SHARPEN_FILTER},setsar=1"
    )


FIT_CHAIN = (
    "scale=1080:1920:force_original_aspect_ratio=decrease:flags=lanczos,"
    "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"
)


FitWindow = Tuple[float, float, Optional[int], int]


def plan_fit_windows(
    plan: List[Dict[str, Any]],
    samples: List[SceneEvidence],
    width: int,
    height: int,
) -> List[FitWindow]:
    """Readable crops for the fitted (full-frame) shots of a layout plan.

    Returns [(start, end, crop_w, x)], with crop_w None where the whole frame
    stays visible, or [] when nothing zooms. Each window contains all detail
    measured during it (for text cut by a face crop: the faces and the text
    lines), so text is never pushed off the edge. Zoom sizes come from
    FIT_VISIBLE_LEVELS. A zoom shorter than FIT_ZOOM_HOLD_SECONDS joins a
    neighbour that already shows its detail, otherwise it shows the full frame;
    a fitted shot with no fitted neighbour keeps its zoom.
    """
    if width <= 0 or height <= 0:
        return []
    min_visible = height * 1080 / FIT_MAX_FOREGROUND_H / width
    levels = [level for level in FIT_VISIBLE_LEVELS if level >= min_visible]

    def needed(left: float, right: float) -> float:
        return next((level for level in levels if right - left <= level), 1.0)

    runs: List[Dict[str, Any]] = []

    def add(start: float, end: float, left: float, right: float) -> None:
        if end <= start:
            return
        if right <= left:
            left = right = (left + right) / 2
        left = max(0.0, left - FIT_DETAIL_MARGIN_FRAC)
        right = min(1.0, right + FIT_DETAIL_MARGIN_FRAC)
        if runs and runs[-1]["end"] == start:
            run = runs[-1]
            union = (min(run["left"], left), max(run["right"], right))
            if needed(left, right) == run["level"] and needed(*union) == run["level"]:
                run.update(end=end, left=union[0], right=union[1])
                return
        runs.append({"start": start, "end": end, "left": left, "right": right,
                     "level": needed(left, right), "settled": False})

    for shot in plan:
        if shot["kind"] != "fit":
            continue
        cut_text = shot.get("reason") == "text_cut_by_crop"
        inside = [s for s in samples if shot["start"] <= s.time < shot["end"]]
        if not inside:
            add(shot["start"], shot["end"], 0.0, 1.0)
            continue
        for index, sample in enumerate(inside):
            start = shot["start"] if index == 0 else sample.time
            end = inside[index + 1].time if index + 1 < len(inside) else shot["end"]
            if cut_text:
                add(start, end, sample.protect_left, sample.protect_right)
            else:
                add(start, end, sample.detail_left, sample.detail_right)

    def touching(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        return abs(a["end"] - b["start"]) < 1e-6 or abs(b["end"] - a["start"]) < 1e-6

    while True:
        short = [r for r in runs if not r["settled"] and r["end"] - r["start"] < FIT_ZOOM_HOLD_SECONDS]
        if not short:
            break
        run = min(short, key=lambda r: r["end"] - r["start"])
        index = runs.index(run)
        neighbours = [
            runs[j] for j in (index - 1, index + 1)
            if 0 <= j < len(runs) and touching(runs[j], run)
        ]
        neighbours.sort(key=lambda r: r["end"] - r["start"], reverse=True)
        host = next((
            n for n in neighbours
            if max(n["right"], run["right"]) - min(n["left"], run["left"]) <= n["level"]
        ), None)
        if not neighbours:
            # A whole fitted shot between crops (or the whole clip) changes
            # layout at its edges anyway, so its zoom causes no extra jump.
            run["settled"] = True
        elif host is not None:
            host.update(
                start=min(host["start"], run["start"]), end=max(host["end"], run["end"]),
                left=min(host["left"], run["left"]), right=max(host["right"], run["right"]),
            )
            runs.remove(run)
        elif run["level"] < 1.0:
            run["level"] = 1.0
        else:
            run["settled"] = True

    # Each size is its own filter branch, so cap how many a clip may use.
    while len({r["level"] for r in runs}) > FIT_MAX_ZOOM_LEVELS:
        smallest, next_size = sorted({r["level"] for r in runs})[:2]
        for run in runs:
            if run["level"] == smallest:
                run["level"] = next_size

    if all(r["level"] >= 1.0 for r in runs):
        return []
    merged: List[Dict[str, Any]] = []
    for run in runs:
        last = merged[-1] if merged else None
        if (
            last is not None and touching(last, run) and last["level"] == run["level"]
            and max(last["right"], run["right"]) - min(last["left"], run["left"]) <= run["level"]
        ):
            last.update(end=run["end"], left=min(last["left"], run["left"]),
                        right=max(last["right"], run["right"]))
        else:
            merged.append(dict(run))
    windows: List[FitWindow] = []
    for run in merged:
        if run["level"] >= 1.0:
            windows.append((run["start"], run["end"], None, 0))
            continue
        crop_w = round_to_even(int(width * run["level"]))
        centre = (run["left"] + run["right"]) / 2 * width
        x = clamp_even(int(centre - crop_w / 2), 0, width - crop_w)
        windows.append((run["start"], run["end"], crop_w, x))
    return windows


def build_fit_layers(
    fit_intervals: List[Tuple[float, float]],
    windows: Optional[List[FitWindow]] = None,
) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Fitted foreground layers as (filter chain, intervals), one per crop size."""
    if not windows:
        return [(FIT_CHAIN, fit_intervals)]
    layers = []
    for crop_w in sorted({w for _, _, w, _ in windows}, key=lambda w: w or 0):
        mine = [window for window in windows if window[2] == crop_w]
        intervals = [(start, end) for start, end, _, _ in mine]
        if crop_w is None:
            layers.append((FIT_CHAIN, intervals))
            continue
        x_expr = str(mine[-1][3])
        for _, end, _, x in reversed(mine[:-1]):
            x_expr = f"if(lt(t\\,{end:.3f})\\,{x}\\,{x_expr})"
        sharpen = f"{UPSCALE_SHARPEN_FILTER}," if crop_w < 1080 else ""
        layers.append((
            f"crop={crop_w}:ih:x='{x_expr}':y=0,scale=1080:-2:flags=lanczos,{sharpen}setsar=1",
            intervals,
        ))
    return layers


def _enable_expr(intervals: List[Tuple[float, float]]) -> str:
    if not intervals:
        return "0"
    return "+".join(f"between(t\\,{a:.3f}\\,{b:.3f})" for a, b in intervals)


def _fit_overlays(base: str, layers: List[Tuple[str, List[Tuple[float, float]]]], gated: bool) -> str:
    """Chain [ft0].. into [fit0].. and overlay them on ``base`` into [vout]."""
    parts = [f"[ft{i}]{chain}[fit{i}]" for i, (chain, _) in enumerate(layers)]
    current = base
    for i, (_, intervals) in enumerate(layers):
        out = "[vout]" if i == len(layers) - 1 else f"[f{i}]"
        enable = f":enable='{_enable_expr(intervals)}'" if gated else ""
        parts.append(f"{current}[fit{i}]overlay=(W-w)/2:(H-h)/2{enable}{out}")
        current = out
    return ";".join(parts)


def build_full_frame_filter(
    blur_sigma: int = 14, fit_windows: Optional[List[FitWindow]] = None
) -> str:
    """Preserve the complete source on a vertical canvas, without a crop branch.

    ``fit_windows`` (see ``plan_fit_windows``) enlarges content whose detail
    leaves empty margins; without them the whole frame is letterboxed.
    """
    background = (
        "[bgsrc]scale=540:960:force_original_aspect_ratio=increase,crop=540:960,"
        f"gblur=sigma={blur_sigma}:steps=2,scale=1080:1920:flags=bilinear,setsar=1[bg];"
    )
    if not fit_windows:
        return (
            "[0:v]split=2[bgsrc][ftsrc];"
            f"{background}"
            f"[ftsrc]{FIT_CHAIN}[fit];"
            "[bg][fit]overlay=(W-w)/2:(H-h)/2[vout]"
        )
    # Nothing but the fit is shown here, so the last window stays on to the
    # final frame even if the stream runs past the probed duration.
    last = max(range(len(fit_windows)), key=lambda i: fit_windows[i][1])
    fit_windows = [
        (start, max(end, 1e6) if i == last else end, crop_w, x)
        for i, (start, end, crop_w, x) in enumerate(fit_windows)
    ]
    layers = build_fit_layers([(0.0, fit_windows[last][1])], fit_windows)
    sources = "".join(f"[ft{i}]" for i in range(len(layers)))
    return (
        f"[0:v]split={1 + len(layers)}[bgsrc]{sources};"
        f"{background}"
        + _fit_overlays("[bg]", layers, gated=len(layers) > 1)
    )


def build_vertical_compositor_filter(
    crop_chain: str,
    face_intervals: List[Tuple[float, float]],
    fit_intervals: List[Tuple[float, float]],
    blur_sigma: int = 14,
    fit_windows: Optional[List[FitWindow]] = None,
) -> str:
    """filter_complex switching between a tracked face crop and a blurred-
    background full-frame fit over time. Produces a labelled [vout] stream.

    Layers: a blurred fill background (always), the face crop on top during face
    shots (covers the frame), and the centred full-frame fit during content
    shots (background shows around it). ``fit_windows`` splits the fit into
    readable zoomed crops (see ``plan_fit_windows``).
    """
    face_en = _enable_expr(face_intervals)
    # Smooth Gaussian background: blur at half resolution (plenty of detail for a
    # heavy blur) with multiple passes for a true Gaussian falloff, then upscale
    # 2x with bilinear so there's no lanczos ringing/blockiness — a creamy blur.
    head = (
        "[bgsrc]scale=540:960:force_original_aspect_ratio=increase,crop=540:960,"
        f"gblur=sigma={blur_sigma}:steps=2,scale=1080:1920:flags=bilinear,setsar=1[bg];"
        f"[crsrc]{crop_chain}[face];"
    )
    if not fit_windows:
        fit_en = _enable_expr(fit_intervals)
        return (
            "[0:v]split=3[bgsrc][crsrc][ftsrc];"
            f"{head}"
            f"[ftsrc]{FIT_CHAIN}[fit];"
            f"[bg][face]overlay=0:0:enable='{face_en}'[t1];"
            f"[t1][fit]overlay=(W-w)/2:(H-h)/2:enable='{fit_en}'[vout]"
        )
    layers = build_fit_layers(fit_intervals, fit_windows)
    sources = "".join(f"[ft{i}]" for i in range(len(layers)))
    return (
        f"[0:v]split={2 + len(layers)}[bgsrc][crsrc]{sources};"
        f"{head}"
        f"[bg][face]overlay=0:0:enable='{face_en}'[t1];"
        + _fit_overlays("[t1]", layers, gated=True)
    )


def _described(
    kind: str, video_filter: str, mode: str, caption_y: Optional[float]
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "video_filter": video_filter,
        "mode": mode,
        "caption_y": caption_y,
    }


def build_vertical_filter_plan(
    input_path: Path,
    width: int,
    height: int,
    speaker_turns: Optional[List[Dict[str, Any]]] = None,
    speaker_words: Optional[List[Dict[str, Any]]] = None,
    report: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Optional[float]]:
    """Build the 9:16 reframing filter for the default vertical mode.

    Returns (filter, mode, caption_y). mode 'vf' is a simple crop chain; mode
    'complex' is a filter_complex producing [vout]. caption_y is a 0-1 output
    fraction that keeps captions off the detected face, or None to keep the
    template default. ``report["kind"]`` receives the layout kind.
    """
    described = describe_vertical_layout(
        input_path, width, height, speaker_turns, speaker_words
    )
    if report is not None:
        report["kind"] = described["kind"]
        if described.get("hook_center_y") is not None:
            report["hook_center_y"] = described["hook_center_y"]
    return described["video_filter"], described["mode"], described["caption_y"]


def describe_vertical_layout(
    input_path: Path,
    width: int,
    height: int,
    speaker_turns: Optional[List[Dict[str, Any]]] = None,
    speaker_words: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Same plan as ``build_vertical_filter_plan``, plus a layout ``kind``.

    Kinds the editor keeps whole are ``reaction`` (reactor above the source),
    ``stack`` (side-by-side call tiles), ``overlap`` (two speakers at once)
    and ``interview`` (per shot: two people stacked, one cropped tight, or
    the full frame; see ``interview_layout``).
    ``crop``, ``speaker``, ``stage``, ``fit`` and ``compositor`` stay on the
    editor's vertical or original crop.

    All-face clips use a face-aware layout when the subject would otherwise
    sit low in a full-height crop (stacked call tiles, a wider stage for one
    webcam, or the whole gallery). Well-framed closeups keep the tracked crop.
    Clips containing content shots use the scene-aware compositor so tweets /
    graphs / slides are shown in full. With diarized speaker turns, multi-person
    wide shots that are not a video-call grid cut to whoever is talking.
    """
    from .talking_head_layout import (
        caption_y_below_subject,
        choose_talking_head_layout,
        dominant_face_bottom,
    )

    crop_w, crop_h = compute_vertical_crop_dims(width, height)
    duration = ffprobe_duration(input_path)

    def bleed_caption(samples) -> Optional[float]:
        bottom = dominant_face_bottom(samples, height)
        if bottom is None:
            return None
        adjusted = caption_y_below_subject(bottom)
        return adjusted if adjusted > 0.805 else None

    # Narrow/portrait source: no horizontal room to crop — static fit, with a
    # slow punch-in so the frame still breathes.
    if crop_w >= width:
        sx, sy, sw, sh = detect_optimal_crop_region(input_path, 0, min(duration, 12.0))
        tail = kenburns_zoom_fragment(duration) or VERTICAL_SCALE_CHAIN
        return _described("crop", f"crop={sw}:{sh}:{sx}:{sy},{tail}", "vf", None)

    # One fast decode pass yields the face track, face boxes, and scene cuts.
    try:
        analyzed = analyze_vertical_clip(input_path)
    except Exception as exc:
        logger.warning("Clip analysis failed (%s); preserving full frame", exc)
        analyzed = ([], [], [])
    track, scene_cuts = analyzed[0], analyzed[1]
    face_samples = analyzed[2] if len(analyzed) > 2 else []
    context = analyzed[3] if len(analyzed) > 3 else []
    plan = (
        build_context_layout_plan(context, scene_cuts, duration, crop_w / width)
        if context else build_layout_plan(track, scene_cuts, duration)
    )
    logger.info("Vertical scene layout: %s", plan)
    try:
        fit_windows = plan_fit_windows(plan, context, width, height) if context else []
    except Exception as exc:
        logger.warning("Readable fit planning failed (%s); keeping the plain fit", exc)
        fit_windows = []
    if fit_windows:
        logger.info("Readable fit windows (start, end, crop_w, x): %s", fit_windows)
    fit_intervals = [(s["start"], s["end"]) for s in plan if s["kind"] == "fit"]
    # Two people who cannot share a 9:16 crop are not slide or sports content.
    # Call/stage layouts and speaker framing still apply; if both decline, that
    # shot keeps the full frame instead of a dominant-face crop.
    content_fits = [
        shot for shot in plan
        if shot["kind"] == "fit" and shot.get("reason") != "interaction_exceeds_crop"
    ]

    # A persistent reaction inset is independent of fit intervals, camera cuts,
    # and speaker labels. Separate it before dominant-face tracking can confuse
    # the inset with the source, or a full-frame fallback discards it.
    from .reaction_layout import choose_reaction_layout, find_reactor_face

    if find_reactor_face(width, height, face_samples):
        # Tiny faces in the source wide shot need more detail than the normal
        # 480px tracking pass. Keep its detections too: a profile/occluded face
        # can be visible to only one of the two scales. The pass only reads
        # face boxes, so skip optical flow, text, and histograms.
        try:
            detailed = analyze_vertical_clip(
                input_path, proc_width=960, analyze_context=False
            )[2]
            combined = []
            for (t, coarse), (_, fine) in zip(face_samples, detailed):
                boxes = []
                for box in sorted(list(coarse) + list(fine), key=lambda f: f[2]*f[3], reverse=True):
                    cx, cy = box[0]+box[2]/2, box[1]+box[3]/2
                    if not any(x <= cx <= x+w and y <= cy <= y+h for x, y, w, h in boxes):
                        boxes.append(box)
                combined.append((t, boxes))
            if len(combined) == len(face_samples):
                face_samples = combined
        except Exception as exc:
            logger.debug("Reaction detail analysis unavailable: %s", exc)
    reaction = choose_reaction_layout(
        input_path, width, height, duration, face_samples, scene_cuts
    )
    if reaction:
        logger.info("Reaction layout: inset %s, %d source sections", reaction['inset'], len(reaction['source_runs']))
        return _described("reaction", reaction["video_filter"], "complex", reaction["caption_y"])

    # A sit-down interview (a two-shot, usually cut with close-ups) is composed
    # shot by shot: both people stacked, one person cropped tight, graphics
    # and b-roll in full. Without this the two-shot letterboxes the clip.
    from .interview_layout import plan_interview_layout

    try:
        interview = plan_interview_layout(
            width, height, duration, scene_cuts, face_samples, context, input_path=input_path)
    except Exception as exc:
        logger.warning("Interview layout failed (%s); using the generic layout", exc)
        interview = None
    if interview:
        result = _described(
            "interview", interview["video_filter"], "complex",
            caption_y_below_subject(interview["face_bottom"]),
        )
        if interview.get("hook_center_y") is not None:
            result["hook_center_y"] = interview["hook_center_y"]
        return result

    if all(s["kind"] == "fit" for s in plan) and content_fits:
        return _described("fit", build_full_frame_filter(fit_windows=fit_windows), "complex", 0.80)

    # choose_talking_head_layout already refuses cut-heavy clips (more than
    # four cuts). One jump cut must not drop stage, stack, or gallery layouts.
    talking = (
        choose_talking_head_layout(width, height, face_samples, scene_cuts)
        if not content_fits else None
    )
    if not content_fits and speaker_words and (not talking or talking["kind"] == "stack"):
        # Diarized speakers: cut to whoever is talking on multi-person wide
        # shots, including a pair that does not fit in one 9:16 crop. Slides
        # and other content shots keep the compositor below.
        from .speaker_framing import build_speaker_filter_plan

        try:
            speaker_plan = build_speaker_filter_plan(
                input_path, width, height, crop_w, crop_h, duration,
                speaker_turns or [], speaker_words,
            )
        except Exception as exc:
            logger.warning("Speaker framing failed (%s); tracking dominant face", exc)
            speaker_plan = None
        if speaker_plan:
            speaker_filter, speaker_mode = speaker_plan
            if not talking or speaker_mode == "complex":
                # Overlap gets full-bleed panels even if the wide shot looks
                # like a call. Put captions between the two faces.
                caption_y = 0.5 if speaker_mode == "complex" else bleed_caption(face_samples)
                kind = "overlap" if speaker_mode == "complex" else "speaker"
                return _described(kind, speaker_filter, speaker_mode, caption_y)
    if talking:
        logger.info(
            "Face-aware vertical layout: %s (upscale %.2fx, caption_y %.2f)",
            talking["kind"], talking["upscale"], talking["caption_y"],
        )
        return _described(
            talking["kind"], talking["video_filter"], "complex", talking["caption_y"]
        )
    if all(s["kind"] == "fit" for s in plan):
        return _described("fit", build_full_frame_filter(fit_windows=fit_windows), "complex", 0.80)

    keys = build_crop_trajectory(track, width, crop_w) if track else []
    # A cut by itself is not camera motion. Ken Burns stays available unless
    # the face trajectory actually pans. Real pans still reset at each cut.
    moving = bool(keys and trajectory_has_movement(keys, crop_w))
    static_x = 0
    if moving:
        x_expr = (
            build_scene_crop_expression(track, scene_cuts, duration, width, crop_w)
            if scene_cuts else build_smooth_pan_expression(keys)
        )
        crop_chain = f"crop={crop_w}:{crop_h}:x='{x_expr}':y=0,{VERTICAL_SCALE_CHAIN}"
    else:
        if keys:
            static_x = clamp_even(
                int(np.median([x for _, x in keys])), 0, max(0, width - crop_w)
            )
        else:
            sx, _, _, _ = detect_optimal_crop_region(input_path, 0, min(duration, 12.0))
            static_x = clamp_even(sx, 0, max(0, width - crop_w))
        crop_chain = f"crop={crop_w}:{crop_h}:{static_x}:0,{VERTICAL_SCALE_CHAIN}"

    if not fit_intervals:
        # Static, all-face clip: add the slow Ken Burns punch-in. (The moving
        # tracked crop and the compositor path keep the plain chain — they
        # already have motion, and zoompan's re-timing would fight overlays.)
        if not moving:
            zoom = kenburns_zoom_fragment(duration)
            if zoom:
                return _described(
                    "crop",
                    f"crop={crop_w}:{crop_h}:{static_x}:0,{zoom}",
                    "vf",
                    bleed_caption(face_samples),
                )
        return _described("crop", crop_chain, "vf", bleed_caption(face_samples))

    face_intervals = [(s["start"], s["end"]) for s in plan if s["kind"] == "face"]
    logger.info(
        "Scene-aware vertical layout: %d face shot(s), %d content shot(s)",
        len(face_intervals), len(fit_intervals),
    )
    return _described(
        "compositor",
        build_vertical_compositor_filter(
            crop_chain, face_intervals, fit_intervals, fit_windows=fit_windows
        ),
        "complex",
        bleed_caption(face_samples),
    )


def render_reframed_clip_ffmpeg(
    input_path: Path,
    output_path: Path,
    output_format: str,
    subtitle_ass_path: Optional[Path] = None,
    fonts_dir: Optional[Path] = None,
    speaker_turns: Optional[List[Dict[str, Any]]] = None,
    speaker_words: Optional[List[Dict[str, Any]]] = None,
    vertical_plan: Optional[Tuple[str, str]] = None,
    speaker_plan: Optional[Dict[str, Any]] = None,
    broll_overlays: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[bool, int, int]:
    """Render the final framed clip and (optionally) burn subtitles in one pass.

    Collapsing reframing + subtitle burn into a single encode avoids a whole
    generation of re-encode loss. The pass uses the high-quality profile, CFR
    output and loudness-normalised audio. ``broll_overlays`` (path, start and
    duration on the clip timeline) are composited over the framed picture and
    under the captions.
    """
    width, height = ffprobe_video_size(input_path)
    has_audio = ffprobe_has_audio(input_path)
    subs = (
        subtitles_filter_fragment(subtitle_ass_path, fonts_dir)
        if subtitle_ass_path
        else None
    )
    audio_args = build_audio_output_args(has_audio)
    overlays = (
        normalize_broll_overlays(broll_overlays, ffprobe_duration(input_path))
        if broll_overlays
        else []
    )

    def composited(head: str, head_label: str, out_w: int, out_h: int, tail: str = "", **encode) -> bool:
        graph, label = broll_overlay_graph(head_label, overlays, out_w, out_h)
        post = ",".join(part for part in (subs, tail) if part)
        if post:
            graph = f"{graph};{label}{post}[v]"
            label = "[v]"
        command = [
            "ffmpeg", "-y", "-i", str(input_path),
            *broll_input_args(overlays),
            "-filter_complex", f"{head};{graph}",
            "-map", label, "-map", "0:a?",
            *build_final_video_encode_args(**encode),
            *audio_args,
            "-movflags", "+faststart",
            str(output_path),
        ]
        return run_ffmpeg_command(command).returncode == 0

    if output_format == "original":
        geometry = probe_video_geometry(input_path)
        display_width = geometry.display_width if geometry else width
        display_height = geometry.display_height if geometry else height
        source_fps = geometry.fps if geometry else None
        capped_width, capped_height = fit_within_1080p(display_width, display_height)
        needs_scale = (capped_width, capped_height) != (display_width, display_height)
        needs_fps = source_fps is not None and source_fps > MAX_OUTPUT_FPS + FPS_EPSILON
        out_w, out_h = round_to_even(capped_width), round_to_even(capped_height)
        if overlays:
            # Always scale to the display size: on a non-square-pixel source
            # the coded frame is not the size the overlay is built for.
            ok = composited(
                f"[0:v]scale={out_w}:{out_h}:flags=lanczos,setsar=1[base]", "[base]", out_w, out_h,
                tail="setsar=1",
            )
            return ok, out_w, out_h
        # A source already within 1080p/60 can be copied. Anything above that is
        # re-encoded so original-format output cannot exceed the cap.
        if not subs and not needs_scale and not needs_fps:
            shutil.copyfile(input_path, output_path)
            return True, round_to_even(display_width), round_to_even(display_height)
        filters: List[str] = []
        if needs_scale:
            filters.append(f"scale={out_w}:{out_h}:flags=lanczos,setsar=1")
        if needs_fps and not subs:
            filters.append(f"fps={MAX_OUTPUT_FPS}")
        if subs:
            filters.append(subs)
        if not needs_scale:
            filters.append("setsar=1")
        if subs:
            encode_fps: int | str = OUTPUT_FPS
            encode_level = "4.1"
        elif needs_fps or source_fps is None:
            encode_fps = MAX_OUTPUT_FPS if needs_fps else OUTPUT_FPS
            encode_level = "4.2" if needs_fps else "4.1"
        else:
            # 59.94 stays 60000/1001. Rounding it to 60 would drop or duplicate
            # frames on an NTSC master that is already inside the cap.
            encode_fps = output_rate_token(source_fps)
            encode_level = "4.2" if source_fps > 30 + FPS_EPSILON else "4.1"
        command = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", ",".join(filters),
            *build_final_video_encode_args(fps=encode_fps, level=encode_level),
            *audio_args,
            "-movflags", "+faststart",
            str(output_path),
        ]
        return run_ffmpeg_command(command).returncode == 0, out_w, out_h

    # A caller that already built the split or pan plan (the editor) skips a
    # second face pass and renders that exact layout.
    if speaker_plan is not None:
        plan = speaker_plan
    else:
        plan = (
            detect_speaker_reframe_plan(input_path, output_format)
            if output_format in {"vertical_pan", "vertical_split", "vertical_facecam"}
            else None
        )

    if plan and plan["mode"] in {"split", "facecam"}:
        vstack_tail = f",{subs}" if subs else ""
        if plan["mode"] == "facecam":
            stacked = build_facecam_stack_filter(plan)
            balloon = _load_hook_balloon(subtitle_ass_path)
            if balloon is not None:
                # Banner goes over the picture and under the burned subtitles.
                hook = build_hook_balloon_filter("fcbase", balloon, "fchook")
                stacked = f"{stacked}[fcbase];{hook};[fchook]null"
        else:
            left = plan["regions"]["left"]
            right = plan["regions"]["right"]
            stacked = (
                f"[0:v]split=2[l][r];"
                f"[l]crop={left['tile_w']}:{left['tile_h']}:{left['tile_x']}:{left['tile_y']},"
                f"scale=1080:960:flags=lanczos,{UPSCALE_SHARPEN_FILTER},setsar=1[lv];"
                f"[r]crop={right['tile_w']}:{right['tile_h']}:{right['tile_x']}:{right['tile_y']},"
                f"scale=1080:960:flags=lanczos,{UPSCALE_SHARPEN_FILTER},setsar=1[rv];"
                f"[lv][rv]vstack,setsar=1"
            )
        # Dynamic facecam cuts get a whoosh/impact under each cut. The mix is a
        # temporary copy (video stream copied, PCM audio) used as the input.
        sfx_source = (
            mix_transition_sfx(input_path, plan.get("segments") or [])
            if plan["mode"] == "facecam" and has_audio
            else None
        )
        if sfx_source is not None:
            input_path = sfx_source
        try:
            if overlays:
                return composited(f"{stacked}[base]", "[base]", 1080, 1920), 1080, 1920
            video_filter = f"{stacked}{vstack_tail}[v]"
            command = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-filter_complex", video_filter,
                "-map", "[v]", "-map", "0:a?",
                *build_final_video_encode_args(),
                *audio_args,
                "-movflags", "+faststart",
                str(output_path),
            ]
            return run_ffmpeg_command(command).returncode == 0, 1080, 1920
        finally:
            if sfx_source is not None:
                shutil.rmtree(sfx_source.parent, ignore_errors=True)

    if plan and plan["mode"] == "pan":
        video_filter = (
            f"crop={plan['crop_w']}:{plan['crop_h']}:x='{plan['x_expression']}':y=0,"
            f"{VERTICAL_SCALE_CHAIN}"
        )
        if overlays:
            return composited(f"[0:v]{video_filter}[base]", "[base]", 1080, 1920), 1080, 1920
        if subs:
            video_filter = f"{video_filter},{subs}"
        command = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", video_filter,
            *build_final_video_encode_args(),
            *audio_args,
            "-movflags", "+faststart",
            str(output_path),
        ]
        return run_ffmpeg_command(command).returncode == 0, 1080, 1920

    # Default "vertical": face-aware for low webcams and calls, tracked crop
    # for a well-framed closeup, blurred full-frame fit for content shots.
    if output_format == "vertical_fit":
        video_filter, mode = build_full_frame_filter(), "complex"
    elif vertical_plan is not None:
        video_filter, mode = vertical_plan
    else:
        video_filter, mode, _caption_y = build_vertical_filter_plan(
            input_path, width, height, speaker_turns, speaker_words
        )
    if overlays:
        if mode == "complex":
            return composited(video_filter, "[vout]", 1080, 1920), 1080, 1920
        return composited(f"[0:v]{video_filter}[base]", "[base]", 1080, 1920), 1080, 1920
    if mode == "complex":
        if subs:
            graph = f"{video_filter};[vout]{subs}[v]"
            map_label = "[v]"
        else:
            graph = video_filter
            map_label = "[vout]"
        command = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-filter_complex", graph,
            "-map", map_label, "-map", "0:a?",
            *build_final_video_encode_args(),
            *audio_args,
            "-movflags", "+faststart",
            str(output_path),
        ]
        return run_ffmpeg_command(command).returncode == 0, 1080, 1920

    if subs:
        video_filter = f"{video_filter},{subs}"
    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", video_filter,
        *build_final_video_encode_args(),
        *audio_args,
        "-movflags", "+faststart",
        str(output_path),
    ]
    return run_ffmpeg_command(command).returncode == 0, 1080, 1920


def caption_probe_plan(input_path: Path, output_format: str, vertical_plan=None) -> Tuple[str, str]:
    """The same composition as the final render, without subtitles or stock footage."""
    if output_format == "vertical_fit":
        return build_full_frame_filter(), "complex"
    if output_format in {"vertical_split", "vertical_pan", "vertical_facecam"}:
        plan = detect_speaker_reframe_plan(input_path, output_format)
        if plan and plan["mode"] == "facecam":
            return f"{build_facecam_stack_filter(plan)}[vout]", "complex"
        if plan and plan["mode"] == "split":
            left, right = plan["regions"]["left"], plan["regions"]["right"]
            return (f"[0:v]split=2[a][b];[a]crop={left['tile_w']}:{left['tile_h']}:{left['tile_x']}:{left['tile_y']},scale=1080:960:flags=lanczos,{UPSCALE_SHARPEN_FILTER},setsar=1[top];[b]crop={right['tile_w']}:{right['tile_h']}:{right['tile_x']}:{right['tile_y']},scale=1080:960:flags=lanczos,{UPSCALE_SHARPEN_FILTER},setsar=1[bottom];[top][bottom]vstack,setsar=1[vout]", "complex")
        if plan and plan["mode"] == "pan":
            return f"crop={plan['crop_w']}:{plan['crop_h']}:x='{plan['x_expression']}':y=0,{VERTICAL_SCALE_CHAIN}", "vf"
    if vertical_plan is not None:
        return vertical_plan
    width, height = ffprobe_video_size(input_path)
    video_filter, mode, _ = build_vertical_filter_plan(input_path, width, height)
    return video_filter, mode
