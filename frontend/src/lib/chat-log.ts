/** Live chat attached to a new task: uploaded once, then referenced by id. */

export type ChatLogUpload = {
  chat_log_id: string;
  message_count: number;
  duration_seconds: number;
};

/** "1:02:03", "62:03" or "3723" → seconds; null when it is not a time. */
export function parseChatOffset(value: string): number | null {
  const text = value.trim();
  if (!text) return 0;
  const negative = text.startsWith("-");
  const body = negative ? text.slice(1) : text;
  let seconds: number;
  if (/^\d+(\.\d+)?$/.test(body)) {
    seconds = Number(body);
  } else if (/^\d{1,3}:\d{1,2}(:\d{1,2})?$/.test(body)) {
    seconds = body.split(":").map(Number).reduce((total, part) => total * 60 + part, 0);
  } else {
    return null;
  }
  return negative ? -seconds : seconds;
}

export async function uploadChatLog(file: File): Promise<ChatLogUpload> {
  const form = new FormData();
  form.append("chat", file, file.name);
  const response = await fetch("/api/chat-logs", { method: "POST", body: form });
  if (!response.ok) {
    let detail = `Erro ${response.status} ao enviar o chat`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // keep the status message
    }
    throw new Error(detail);
  }
  return response.json();
}
