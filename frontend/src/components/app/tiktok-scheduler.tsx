"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

// Panel for the local poster (postador.bat): pick clips, edit each description
// and schedule them on one or more TikTok accounts and/or YouTube Shorts from a
// chosen date and time; the next clips follow 5 minutes apart.

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
  registro_yt?: Record<string, Registro>;
  contas_tiktok?: ContaTikTok[];
  registro_tt?: Record<string, Record<string, Registro>>;
  avatares?: Record<string, number>;
  yt_logado?: boolean | null;
  yt_canal?: string | null;
  yt_esperado?: string | null;
};

type ContaTikTok = { conta: string; principal: boolean; logado: boolean | null; lida: string | null };

// Destinations are "tiktok:@account" (one per TikTok account) or "youtube".
const YOUTUBE = "youtube";
const PLATAFORMAS_KEY = "supoclip-postador-destinos";
const PLATAFORMAS_KEY_ANTIGA = "supoclip-postador-plataformas";
const tiktokId = (conta: string) => `tiktok:${conta}`;
const avatarKey = (destino: string) =>
  destino === YOUTUBE ? "yt" : `tt-${destino.replace(/^tiktok:/, "").replace(/^@/, "").toLowerCase()}`;

/** Profile photo saved by the poster (or the account's initial until it has one). */
function Avatar({ chave, versao, nome, className }: { chave: string; versao?: number; nome: string; className?: string }) {
  const [quebrada, setQuebrada] = useState(false);
  const letra = nome.replace(/^@/, "").charAt(0).toUpperCase() || "?";
  if (!versao || quebrada) {
    return (
      <span className={cn("inline-flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-[10px] font-semibold text-muted-foreground", className)} aria-hidden>
        {letra}
      </span>
    );
  }
  return (
    // eslint-disable-next-line @next/next/no-img-element -- small local image served by the poster
    <img
      src={`/api/tiktok/avatar?chave=${encodeURIComponent(chave)}&v=${versao}`}
      alt={`Foto de ${nome}`}
      className={cn("size-6 shrink-0 rounded-full object-cover ring-1 ring-border", className)}
      onError={() => setQuebrada(true)}
    />
  );
}

const STATUS_LABEL: Record<string, string> = {
  "na fila": "⏳ na fila",
  enviando: "⬆ enviando…",
  agendado: "✓ agendado",
  erro: "⚠ erro",
};

const MIN_AHEAD_MIN = 15; // TikTok's minimum
const MAX_AHEAD_DAYS = 30;
const GAP_MIN = 5;
const pad = (n: number) => String(n).padStart(2, "0");
const ymd = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const fmt = (d: Date) => `${pad(d.getDate())}/${pad(d.getMonth() + 1)} ${pad(d.getHours())}:${pad(d.getMinutes())}`;

/** Earliest slot TikTok accepts, on a 5-minute mark. */
function earliestSlot(): Date {
  const d = new Date(Date.now() + (MIN_AHEAD_MIN + 1) * 60_000);
  d.setSeconds(0, 0);
  d.setMinutes(Math.ceil(d.getMinutes() / GAP_MIN) * GAP_MIN);
  return d;
}

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
  const [day, setDay] = useState(() => ymd(earliestSlot()));
  const [hour, setHour] = useState(() => pad(earliestSlot().getHours()));
  const [minute, setMinute] = useState(() => pad(earliestSlot().getMinutes()));
  const ids = useMemo(() => clips.map((clip) => clip.id).join(","), [clips]);
  const [plataformas, setPlataformas] = useState<Record<string, boolean>>({});
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(PLATAFORMAS_KEY) || "null");
      if (saved && typeof saved === "object") {
        setPlataformas(saved);
        return;
      }
      // Older panel stored {tiktok, youtube}: "tiktok" meant the main account.
      const antiga = JSON.parse(localStorage.getItem(PLATAFORMAS_KEY_ANTIGA) || "null");
      if (antiga && typeof antiga === "object") setPlataformas({ tiktok: Boolean(antiga.tiktok), [YOUTUBE]: Boolean(antiga.youtube) });
    } catch {}
  }, []);
  const togglePlataforma = (destino: string, on: boolean) =>
    setPlataformas((current) => {
      const next: Record<string, boolean> = { ...current };
      // Turn the old "tiktok" (= main account) choice into the main account's own key first.
      if ("tiktok" in next) {
        const principal = status?.contas_tiktok?.find((c) => c.principal)?.conta ?? status?.esperada;
        if (next.tiktok && principal) next[tiktokId(principal)] = true;
        delete next.tiktok;
      }
      next[destino] = on;
      try {
        localStorage.setItem(PLATAFORMAS_KEY, JSON.stringify(next));
      } catch {}
      return next;
    });

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
  const contas: ContaTikTok[] = status?.contas_tiktok?.length
    ? status.contas_tiktok
    : status?.esperada
      ? [{ conta: status.esperada, principal: true, logado: status.logado ?? null, lida: status.conta ?? null }]
      : [];
  // The old saved choice "tiktok" means the main account.
  const marcado = (destino: string) =>
    Boolean(plataformas[destino]) ||
    (destino !== YOUTUBE && Boolean(plataformas.tiktok) && contas.find((c) => tiktokId(c.conta) === destino)?.principal === true);
  const destinos = [...contas.map((c) => tiktokId(c.conta)), YOUTUBE];
  const escolhidas = destinos.filter(marcado);
  const errada = (c: ContaTikTok) =>
    Boolean(online && c.logado && c.lida && c.lida.toLowerCase() !== c.conta.toLowerCase());
  const canalErrado = Boolean(
    online && status?.yt_logado && status.yt_esperado && status.yt_canal &&
      status.yt_canal.toLowerCase() !== status.yt_esperado.toLowerCase(),
  );
  const bloqueado = contas.some((c) => marcado(tiktokId(c.conta)) && errada(c)) || (marcado(YOUTUBE) && canalErrado);
  const registro = status?.registro ?? {};
  const registroYt = status?.registro_yt ?? {};
  const registroDe = (conta: ContaTikTok): Record<string, Registro> =>
    status?.registro_tt?.[conta.conta] ?? (conta.principal ? registro : {});
  // Follow the page's order ("Melhores primeiro" or "Linha do tempo"): that is the upload order.
  const ordered = clips;
  const selected = ordered.filter((clip) => picked.has(clip.id));
  const inQueue = (status?.pendentes?.length ?? 0) + (status?.atual ? 1 : 0);

  // Only the slots TikTok accepts: from 15 min ahead (5-min marks) up to MAX_AHEAD_DAYS.
  const earliest = earliestSlot();
  const minDay = ymd(earliest);
  const maxDay = ymd(new Date(Date.now() + MAX_AHEAD_DAYS * 86_400_000));
  const hourOptions = Array.from({ length: 24 }, (_, h) => h)
    .filter((h) => day !== minDay || h >= earliest.getHours())
    .map(pad);
  const minuteOptions = Array.from({ length: 60 / GAP_MIN }, (_, m) => m * GAP_MIN)
    .filter((m) => day !== minDay || Number(hour) !== earliest.getHours() || m >= earliest.getMinutes())
    .map(pad);
  const start = new Date(`${day}T${hour}:${minute}:00`);
  let startError: string | null = null;
  {
    if (Number.isNaN(start.getTime())) startError = "Escolha a data e a hora.";
    else if (start.getTime() < Date.now() + MIN_AHEAD_MIN * 60_000) startError = `O TikTok só agenda com pelo menos ${MIN_AHEAD_MIN} min de antecedência.`;
    else if (day > maxDay) startError = `O TikTok só agenda até ${MAX_AHEAD_DAYS} dias pra frente.`;
  }
  const slotOf = (index: number) => new Date(start.getTime() + index * GAP_MIN * 60_000);
  const lastSlot = slotOf(Math.max(selected.length - 1, 0));

  // Time moves on: if the chosen slot fell behind the earliest one, move it up.
  const behind = Number.isNaN(start.getTime()) || start.getTime() < earliest.getTime();
  useEffect(() => {
    if (!behind) return;
    const e = earliestSlot();
    setDay(ymd(e));
    setHour(pad(e.getHours()));
    setMinute(pad(e.getMinutes()));
  }, [behind]);

  const situacaoConta = (c: ContaTikTok): [string, string] => {
    if (errada(c)) return ["bg-destructive", `conta errada logada (${c.lida}). Nada será enviado.`];
    if (c.logado === false) return ["bg-amber-500", "não está logado: clique em Entrar"];
    if (c.logado) return ["bg-emerald-500", "conectado"];
    return ["bg-amber-500", "clique em Verificar"];
  };

  let ytDot = "bg-muted-foreground";
  let ytLine = `canal ${status?.yt_esperado ?? "?"}: clique em Verificar canal`;
  if (status?.offline) {
    ytDot = "bg-destructive";
    ytLine = "postador desligado";
  } else if (canalErrado) {
    ytDot = "bg-destructive";
    ytLine = `canal errado aberto (${status?.yt_canal}); o esperado é ${status?.yt_esperado}. Nada será enviado.`;
  } else if (status?.yt_logado === false) {
    ytDot = "bg-amber-500";
    ytLine = "não está logado: clique em Entrar no YouTube";
  } else if (status?.yt_logado) {
    ytDot = "bg-emerald-500";
    ytLine = `canal ${status.yt_canal ?? "?"}`;
  }

  const toggle = (id: string, on: boolean) =>
    setPicked((current) => {
      const next = new Set(current);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });

  async function act(acao: "login" | "checar" | "cancelar" | "login_youtube" | "checar_youtube", conta?: string) {
    setBusy(true);
    try {
      const r = await call(acao, { method: "POST", body: JSON.stringify(conta ? { conta } : {}) });
      if (r.offline) throw new Error(r.erro);
      if (acao === "login") toast(`Entre na conta ${conta ?? ""} no Chrome que abriu e depois FECHE essa janela. Em seguida clique em Verificar.`);
      if (acao === "checar") toast(`Abrindo o TikTok Studio para verificar ${conta ?? "a conta"}…`);
      if (acao === "login_youtube") toast("Entre na conta do canal no Chrome que abriu e depois FECHE essa janela. Em seguida clique em Verificar canal.");
      if (acao === "checar_youtube") toast("Abrindo o YouTube Studio para verificar o canal…");
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
    if (!escolhidas.length) {
      toast.error("Escolha TikTok e/ou YouTube Shorts.");
      return;
    }
    const nomes = escolhidas
      .map((d) => d === YOUTUBE
        ? `YouTube Shorts (${status?.yt_canal || status?.yt_esperado || "canal não verificado"})`
        : `TikTok ${d.slice("tiktok:".length)}`)
      .join(" e ");
    if (startError) {
      toast.error(startError);
      return;
    }
    const when = `O primeiro sai em ${fmt(start)}${selected.length > 1 ? ` e o último em ${fmt(lastSlot as Date)}` : ""} (um a cada ${GAP_MIN} min).`;
    if (!window.confirm(`Agendar ${selected.length} corte(s) em: ${nomes}?\n${when}`)) return;
    const again = selected.filter((clip) =>
      contas.some((c) => marcado(tiktokId(c.conta)) && registroDe(c)[clip.id]?.status === "agendado") ||
      (marcado(YOUTUBE) && registroYt[clip.id]?.status === "agendado"));
    if (again.length && !window.confirm(`${again.length} desses já foram agendados. Agendar de novo mesmo assim?`)) return;
    setBusy(true);
    try {
      const r = await call("enviar", {
        method: "POST",
        body: JSON.stringify({
          itens: selected.map((clip) => ({ clipId: clip.id, legenda: descs[clip.id] ?? "" })),
          inicio: `${day}T${hour}:${minute}`,
          plataformas: escolhidas,
        }),
      });
      if (r.offline) throw new Error(r.erro);
      toast(`${r.adicionados ?? 0} envio(s) na fila para agendar`);
      setPicked(new Set());
      void refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="mt-10 rounded-xl border p-4 sm:p-5" aria-label="Agendar no TikTok e YouTube">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-base font-semibold">Agendar no TikTok e YouTube</h2>
          {status?.offline ? (
            <p className="mt-1 flex items-center gap-2 text-sm text-destructive">
              <span className="size-2 shrink-0 rounded-full bg-destructive" aria-hidden />
              {status.erro || "Postador desligado: abra o postador.bat na pasta do SupoClip."}
            </p>
          ) : !status ? (
            <p className="mt-1 text-sm text-muted-foreground">verificando o postador…</p>
          ) : (
            <div className="mt-1 space-y-1">
              {contas.map((c) => {
                const [cor, texto] = situacaoConta(c);
                return (
                  <div key={c.conta} className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
                    <Avatar chave={avatarKey(tiktokId(c.conta))} versao={status?.avatares?.[avatarKey(tiktokId(c.conta))]} nome={c.conta} />
                    <span className={cn("size-2 shrink-0 rounded-full", cor)} aria-hidden />
                    <span className="min-w-0 break-words">TikTok {c.conta}: {texto}</span>
                    <Button size="sm" variant="ghost" className="h-7 px-2" disabled={busy} onClick={() => void act("login", c.conta)}>Entrar</Button>
                    <Button size="sm" variant="ghost" className="h-7 px-2" disabled={busy} onClick={() => void act("checar", c.conta)}>Verificar</Button>
                  </div>
                );
              })}
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
                <Avatar chave="yt" versao={status?.avatares?.yt} nome={status?.yt_canal || status?.yt_esperado || "YouTube"} />
                <span className={cn("size-2 shrink-0 rounded-full", ytDot)} aria-hidden />
                <span className="min-w-0 break-words">YouTube: {ytLine}</span>
                <Button size="sm" variant="ghost" className="h-7 px-2" disabled={busy} onClick={() => void act("login_youtube")}>Entrar</Button>
                <Button size="sm" variant="ghost" className="h-7 px-2" disabled={busy} onClick={() => void act("checar_youtube")}>Verificar</Button>
              </div>
            </div>
          )}
          {status?.msg ? <p className="mt-1 text-xs text-muted-foreground">{status.msg}</p> : null}
        </div>
        <div className="flex flex-wrap gap-2">
          {inQueue > 0 ? (
            <Button size="sm" variant="outline" disabled={busy} onClick={() => {
              if (window.confirm("Cancelar todos os cortes da fila? O que já foi agendado continua agendado.")) void act("cancelar");
            }}>Cancelar fila ({inQueue})</Button>
          ) : null}
          <Button size="sm" disabled={!online || busy || bloqueado || !escolhidas.length || !selected.length || Boolean(startError)} onClick={() => void schedule()}>
            {`Agendar${selected.length ? ` ${selected.length}` : ""}${startError ? "" : ` (${fmt(start)})`}`}
          </Button>
        </div>
      </div>

      <div className="mt-4 rounded-lg bg-muted/40 p-3">
        <div className="flex flex-wrap items-end gap-3 text-sm">
          <div className="flex basis-full flex-wrap items-center gap-x-4 gap-y-2">
            <span className="font-medium">Onde postar</span>
            {destinos.map((destino) => (
              <label key={destino} className="flex cursor-pointer items-center gap-1.5">
                <Checkbox checked={marcado(destino)} onCheckedChange={(value) => togglePlataforma(destino, value === true)} />
                <Avatar
                  chave={avatarKey(destino)}
                  versao={status?.avatares?.[avatarKey(destino)]}
                  nome={destino === YOUTUBE ? status?.yt_canal || "YouTube" : destino.slice("tiktok:".length)}
                  className="size-5"
                />
                {destino === YOUTUBE ? `YouTube Shorts${status?.yt_canal ? ` (${status.yt_canal})` : ""}` : `TikTok ${destino.slice("tiktok:".length)}`}
              </label>
            ))}
          </div>
          <span className="basis-full font-medium">Quando publicar o 1º corte</span>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Dia</span>
            <input type="date" className="h-9 rounded-md border bg-background px-2" value={day} min={minDay} max={maxDay}
              onChange={(event) => event.target.value && setDay(event.target.value)} />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Hora</span>
            <select className="h-9 rounded-md border bg-background px-2" value={hour} onChange={(event) => setHour(event.target.value)}>
              {hourOptions.map((h) => <option key={h} value={h}>{h}</option>)}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs text-muted-foreground">Minuto</span>
            <select className="h-9 rounded-md border bg-background px-2" value={minute} onChange={(event) => setMinute(event.target.value)}>
              {minuteOptions.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
          </label>
          <Button size="sm" variant="ghost" type="button" onClick={() => {
            setDay(ymd(earliest));
            setHour(pad(earliest.getHours()));
            setMinute(pad(earliest.getMinutes()));
          }}>O mais cedo possível ({fmt(earliest)})</Button>
          <p className={cn("basis-full text-xs", startError ? "text-destructive" : "text-muted-foreground")}>
            {startError ??
              (selected.length > 1
                ? `1º corte em ${fmt(start)}, depois um a cada ${GAP_MIN} min (último em ${fmt(lastSlot as Date)}). Horário do seu PC.`
                : `O 1º sai em ${fmt(start)} e cada próximo corte marcado sai ${GAP_MIN} min depois do anterior. Horário do seu PC.`)}
          </p>
        </div>
      </div>

      <div className="mt-4 flex items-center gap-3 text-xs text-muted-foreground">
        <button type="button" className="underline-offset-2 hover:underline" onClick={() => setPicked(new Set(ordered.map((clip) => clip.id)))}>Selecionar todos</button>
        <button type="button" className="underline-offset-2 hover:underline" onClick={() => setPicked(new Set())}>Limpar seleção</button>
      </div>

      <ul className="mt-3 divide-y">
        {ordered.map((clip) => {
          const regs = ([
            ...contas.map((c) => [contas.length > 1 ? `TikTok ${c.conta}` : "TikTok", registroDe(c)[clip.id]] as const),
            ["YouTube", registroYt[clip.id]] as const,
          ].filter(([, reg]) => Boolean(reg))) as (readonly [string, Registro])[];
          const label = regs.length > 0;
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
                  {start && !startError && picked.has(clip.id) && !label ? (
                    <span className="text-xs text-muted-foreground">sai em {fmt(slotOf(selected.indexOf(clip)) as Date)}</span>
                  ) : null}
                  {regs.map(([nome, reg]) => (
                    <span key={nome} className={cn("text-xs", reg.status === "erro" ? "text-destructive" : reg.status === "agendado" ? "text-emerald-600 dark:text-emerald-400" : "text-muted-foreground")}>
                      {nome}: {STATUS_LABEL[reg.status] ?? reg.status}{reg.status === "agendado" && reg.para ? ` para ${reg.para}` : ""}
                    </span>
                  ))}
                </div>
                {regs.map(([nome, reg]) => reg.status === "erro" && reg.erro
                  ? <p key={nome} className="mt-0.5 text-xs text-destructive">{nome}: {reg.erro}</p>
                  : null)}
                <Textarea
                  className="mt-2 min-h-16 text-sm"
                  rows={2}
                  value={descs[clip.id] ?? ""}
                  onChange={(event) => setDescs((current) => ({ ...current, [clip.id]: event.target.value }))}
                  aria-label={`Descrição do corte ${clip.clip_order}`}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
