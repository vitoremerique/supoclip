import { NextResponse } from "next/server";

import { getPrismaClient } from "@/server/prisma";
import { getServerSession } from "@/server/session";

// Local TikTok poster (postador.bat) running on the Windows host. It drives a
// real Chrome, so it cannot live in Docker; this route only forwards to it.
const POSTADOR_URL = process.env.POSTADOR_URL || "http://host.docker.internal:5680";
const ACOES = new Set(["status", "enviar", "cancelar", "login", "checar", "login_youtube", "checar_youtube", "avatar"]);
const AVATAR_RE = /^(yt|tt-[a-z0-9._]{2,40})$/;
// "youtube", "tiktok" (main account) or "tiktok:@account" (any account configured in the poster).
const PLATAFORMA_RE = /^(youtube|tiktok|tiktok:@[A-Za-z0-9._]{2,40})$/;
const CONTA_RE = /^@[A-Za-z0-9._]{2,40}$/;
const MAX_LEGENDA = 2200;

type ClipRow = { id: string; file_path: string; filename: string; hook_title: string | null };

async function postador(path: string, init?: RequestInit) {
  try {
    const res = await fetch(`${POSTADOR_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
      cache: "no-store",
      signal: AbortSignal.timeout(15000),
    });
    const body = await res.json().catch(() => ({}));
    return NextResponse.json(body, { status: res.status });
  } catch {
    return NextResponse.json(
      { offline: true, erro: "Postador desligado: abra o postador.bat na pasta do SupoClip." },
      { status: 503 },
    );
  }
}

async function ownedClips(userId: string, ids: string[]): Promise<ClipRow[]> {
  if (!ids.length) return [];
  return getPrismaClient().$queryRaw<ClipRow[]>`
    SELECT c.id::text AS id, c.file_path, c.filename, c.hook_title
    FROM generated_clips c
    JOIN tasks t ON t.id = c.task_id
    WHERE c.id::text = ANY(${ids}::text[]) AND t.user_id = ${userId}`;
}

async function handle(request: Request, { params }: { params: Promise<{ acao: string }> }) {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const { acao } = await params;
  if (!ACOES.has(acao)) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  if (acao === "avatar") {
    // Profile photo of a connected account, saved by the poster.
    const chave = new URL(request.url).searchParams.get("chave") || "";
    if (!AVATAR_RE.test(chave)) return new NextResponse(null, { status: 404 });
    try {
      const res = await fetch(`${POSTADOR_URL}/avatar?chave=${encodeURIComponent(chave)}`, {
        cache: "no-store",
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) return new NextResponse(null, { status: 404 });
      return new NextResponse(await res.arrayBuffer(), {
        headers: { "Content-Type": "image/jpeg", "Cache-Control": "private, max-age=60" },
      });
    } catch {
      return new NextResponse(null, { status: 404 });
    }
  }

  if (acao === "status") {
    const ids = (new URL(request.url).searchParams.get("ids") || "").split(",").filter(Boolean).slice(0, 200);
    const owned = (await ownedClips(session.user.id, ids)).map((clip) => clip.id);
    const res = await postador(`/status?ids=${encodeURIComponent(owned.join(","))}`);
    const body = await res.json();
    return NextResponse.json(
      { ...body, hashtags: process.env.TIKTOK_HASHTAGS || "" },
      { status: res.status },
    );
  }

  if (acao === "enviar") {
    const payload = (await request.json().catch(() => ({}))) as {
      itens?: { clipId?: string; legenda?: string }[];
      inicio?: string;
      plataformas?: string[];
    };
    const plataformas = (payload.plataformas || ["tiktok"]).filter((p) => typeof p === "string" && PLATAFORMA_RE.test(p)).slice(0, 10);
    if (!plataformas.length) {
      return NextResponse.json({ error: "Escolha TikTok e/ou YouTube." }, { status: 400 });
    }
    // Optional local date/time for the first clip ("2026-11-21T18:00"); the rest follow 5 min apart.
    const inicio = typeof payload.inicio === "string" && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(payload.inicio)
      ? payload.inicio
      : undefined;
    const itens = (payload.itens || []).filter((item) => typeof item.clipId === "string").slice(0, 100);
    const clips = await ownedClips(session.user.id, itens.map((item) => item.clipId as string));
    const byId = new Map(clips.map((clip) => [clip.id, clip]));
    // Keep the order the user picked: the poster schedules them 5 min apart.
    const forward = itens
      .filter((item) => byId.has(item.clipId as string))
      .map((item) => {
        const clip = byId.get(item.clipId as string) as ClipRow;
        return {
          id: clip.id,
          arquivo: clip.file_path,
          nome: clip.filename,
          legenda: String(item.legenda || "").slice(0, MAX_LEGENDA),
          gancho: clip.hook_title || "",
        };
      });
    if (!forward.length) {
      return NextResponse.json({ error: "Nenhum corte válido selecionado." }, { status: 400 });
    }
    return postador("/enviar", { method: "POST", body: JSON.stringify({ itens: forward, inicio, plataformas }) });
  }

  if (request.method !== "POST") {
    return NextResponse.json({ error: "Method not allowed" }, { status: 405 });
  }
  // login/checar can name a TikTok account; the poster checks it against its own list.
  const body = (await request.json().catch(() => ({}))) as { conta?: unknown };
  const conta = typeof body.conta === "string" && CONTA_RE.test(body.conta) ? body.conta : undefined;
  return postador(`/${acao}`, { method: "POST", body: JSON.stringify(conta ? { conta } : {}) });
}

export const GET = handle;
export const POST = handle;
