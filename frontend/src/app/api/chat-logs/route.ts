import { headers } from "next/headers";
import { NextResponse } from "next/server";

import { auth } from "@/lib/auth";
import { buildBackendAuthHeaders } from "@/lib/backend-auth";
import { backendTraceHeaders } from "@/server/backend-api";

export const dynamic = "force-dynamic";
export const maxDuration = 600;

// Live chat export (TwitchDownloader / yt-dlp / chat-downloader JSON or .txt) → POST /chat-logs.
export async function POST(request: Request) {
  const session = await auth.api.getSession({ headers: await headers() });
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  const contentType = request.headers.get("content-type");
  const declaredLength = request.headers.get("content-length");
  if (!contentType || !request.body || !declaredLength) {
    return NextResponse.json({ detail: "A multipart upload is required" }, { status: 400 });
  }
  const apiUrl = (
    process.env.BACKEND_INTERNAL_URL ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8000"
  ).replace(/\/$/, "");
  const init: RequestInit & { duplex: "half" } = {
    method: "POST",
    headers: {
      ...buildBackendAuthHeaders(session.user.id),
      ...(await backendTraceHeaders()),
      "content-type": contentType,
      "content-length": declaredLength,
    },
    body: request.body,
    duplex: "half",
  };
  let upstream: Response;
  try {
    upstream = await fetch(`${apiUrl}/chat-logs`, init);
  } catch (error) {
    console.error("Chat log could not be forwarded to the API", error);
    return NextResponse.json({ detail: "Não foi possível enviar o chat agora." }, { status: 502 });
  }
  return new NextResponse(await upstream.text(), {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("content-type") || "application/json" },
  });
}
