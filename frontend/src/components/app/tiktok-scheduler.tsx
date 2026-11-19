"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

// Panel for the local TikTok poster (postador.bat): pick clips, edit each
// description and schedule them on TikTok 15 minutes ahead, 5 minutes apart.

type SchedulerClip = { id: string; clip_order: number; hook_title: string | null; filename: string };

type Registro = { status: string; para?: string; erro?: string | null };

type PosterStatus = {
  offline?: boolean;
  erro?: string;
  logado?: boolean | null;
  conta?: string | null;
  esperada?: string | null;
  atual?: string | null;
  pendentes?: string[];
  msg?: string;
  hashtags?: string;
  registro?: Record<string, Registro>;
};

const STATUS_LABEL: Record<string, string> = {
  "na fila": "⏳ na fila",
  enviando: "⬆ enviando ao TikTok…",
  agendado: "✓ agendado",
  erro: "⚠ erro",
};

async function call(path: string, init?: RequestInit): Promise<PosterStatus & { adicionados?: number; cancelados?: number }> {
  const res = await fetch(`/api/tiktok/${path}`, {
    ...init,
    headers: { "Content-Type": "application/json" },
    cache: "no-store",
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok && !body.offline) throw new Error(body.error || body.erro || "Falha ao falar com o postador");
  return body;
}

export function TikTokScheduler({ clips }: { clips: SchedulerClip[] }) {
  const [status, setStatus] = useState<PosterStatus | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [descs, setDescs] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const ids = useMemo(() => clips.map((clip) => clip.id).join(","), [clips]);

  const refresh = useCallback(async () => {
    try {
      setStatus(await call(`status?ids=${encodeURIComponent(ids)}`));
    } catch (err) {
      setStatus({ offline: true, erro: err instanceof Error ? err.message : String(err) });
    }
  }, [ids]);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  // Default description: hook title + hashtags from .env (TIKTOK_HASHTAGS).
  const hashtags = status?.hashtags ?? "";
  useEffect(() => {
    if (status === null) return;
    setDescs((current) => {
      const next = { ...current };
      for (const clip of clips) {
        if (!(clip.id in next)) next[clip.id] = `${clip.hook_title ?? ""} ${hashtags}`.trim();
      }
      return next;
    });
  }, [clips, hashtags, status]);

  const online = status && !status.offline;
  const contaErrada = Boolean(
    online && status?.logado && status.esperada && status.conta &&
      status.conta.toLowerCase() !== status.esperada.toLowerCase(),
  );
  const registro = status?.registro ?? {};
  const ordered = [...clips].sort((a, b) => a.clip_order - b.clip_order);
  const selected = ordered.filter((clip) => picked.has(clip.id));
  const inQueue = (status?.pendentes?.length ?? 0) + (status?.atual ? 1 : 0);

  let dot = "bg-muted-foreground";
  let line = "verificando o postador…";
  if (status?.offline) {
    dot = "bg-destructive";
    line = status.erro || "Postador desligado: abra o postador.bat na pasta do SupoClip.";
  } else if (contaErrada) {
    dot = "bg-destructive";
    line = `conta errada logada (${status?.conta}); o esperado é ${status?.esperada}. Nada será enviado.`;
  } else if (status?.logado === false) {
    dot = "bg-amber-500";
    line = "não está logado: clique em Entrar no TikTok";
  } else if (status?.logado) {
    dot = "bg-emerald-500";
    line = `conectado como ${status.conta ?? "?"}`;
  } else if (online) {
    dot = "bg-amber-500";
    line = `postador ligado${status?.esperada ? ` (conta ${status.esperada})` : ""}: clique em Verificar conta`;
  }

  const toggle = (id: string, on: boolean) =>
    setPicked((current) => {
      const next = new Set(current);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });

  async function act(acao: "login" | "checar" | "cancelar") {
    setBusy(true);
    try {
      const r = await call(acao, { method: "POST", body: "{}" });
      if (r.offline) throw new Error(r.erro);
      if (acao === "login") toast("Entre na conta no Chrome que abriu e depois FECHE essa janela. Em seguida clique em Verificar conta.");
      if (acao === "checar") toast("Abrindo o TikTok Studio para verificar a conta…");
      if (acao === "cancelar") toast(`${r.cancelados ?? 0} corte(s) tirados da fila`);
      void refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function schedule() {
    if (!selected.length) return;
    const conta = status?.conta || status?.esperada || "(conta não verificada)";
    if (!window.confirm(`Agendar ${selected.length} corte(s) na conta ${conta}?\nO primeiro sai daqui ~15 min, depois um a cada 5 min.`)) return;
    const again = selected.filter((clip) => registro[clip.id]?.status === "agendado");
    if (again.length && !window.confirm(`${again.length} desses já foram agendados. Agendar de novo mesmo assim?`)) return;
    setBusy(true);
    try {
      const r = await call("enviar", {
        method: "POST",
        body: JSON.stringify({ itens: selected.map((clip) => ({ clipId: clip.id, legenda: descs[clip.id] ?? "" })) }),
      });
      if (r.offline) throw new Error(r.erro);
      toast(`${r.adicionados ?? 0} corte(s) na fila para agendar`);
      setPicked(new Set());
      void refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="mt-10 rounded-xl border p-4 sm:p-5" aria-label="Agendar no TikTok">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-base font-semibold">Agendar no TikTok</h2>
          <p className="mt-1 flex items-center gap-2 text-sm text-muted-foreground">
            <span className={cn("size-2 shrink-0 rounded-full", dot)} aria-hidden />
            <span className="min-w-0 break-words">{line}</span>
          </p>
          {status?.msg ? <p className="mt-1 text-xs text-muted-foreground">{status.msg}</p> : null}
        </div>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant="outline" disabled={!online || busy} onClick={() => void act("login")}>Entrar no TikTok</Button>
          <Button size="sm" variant="outline" disabled={!online || busy} onClick={() => void act("checar")}>Verificar conta</Button>
          {inQueue > 0 ? (
            <Button size="sm" variant="outline" disabled={busy} onClick={() => {
              if (window.confirm("Cancelar todos os cortes da fila? O que já foi agendado no TikTok continua agendado.")) void act("cancelar");
            }}>Cancelar fila ({inQueue})</Button>
          ) : null}
          <Button size="sm" disabled={!online || busy || contaErrada || !selected.length} onClick={() => void schedule()}>
            {selected.length ? `Agendar ${selected.length} (daqui 15 min)` : "Agendar (daqui 15 min)"}
          </Button>
        </div>
      </div>

      <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
        <button type="button" className="underline-offset-2 hover:underline" onClick={() => setPicked(new Set(ordered.map((clip) => clip.id)))}>Selecionar todos</button>
        <button type="button" className="underline-offset-2 hover:underline" onClick={() => setPicked(new Set())}>Limpar seleção</button>
      </div>

      <ul className="mt-3 divide-y">
        {ordered.map((clip) => {
          const reg = registro[clip.id];
          const label = reg ? STATUS_LABEL[reg.status] ?? reg.status : null;
          return (
            <li key={clip.id} className="flex gap-3 py-3">
              <Checkbox
                className="mt-1"
                checked={picked.has(clip.id)}
                onCheckedChange={(value) => toggle(clip.id, value === true)}
                aria-label={`Selecionar corte ${clip.clip_order}`}
              />
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <p className="min-w-0 truncate text-sm font-medium">
                    #{clip.clip_order} · {clip.hook_title || clip.filename}
                  </p>
                  {label ? (
                    <span className={cn("text-xs", reg?.status === "erro" ? "text-destructive" : reg?.status === "agendado" ? "text-emerald-600 dark:text-emerald-400" : "text-muted-foreground")}>
                      {label}{reg?.status === "agendado" && reg.para ? ` para ${reg.para}` : ""}
                    </span>
                  ) : null}
                </div>
                {reg?.status === "erro" && reg.erro ? <p className="mt-0.5 text-xs text-destructive">{reg.erro}</p> : null}
                <Textarea
                  className="mt-2 min-h-16 text-sm"
                  rows={2}
                  value={descs[clip.id] ?? ""}
                  onChange={(event) => setDescs((current) => ({ ...current, [clip.id]: event.target.value }))}
                  aria-label={`Descrição do corte ${clip.clip_order} no TikTok`}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
