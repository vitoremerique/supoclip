"""Postador do TikTok do SupoClip — agenda cada corte para daqui a 15 min.

Mesmo robô do auto-cortes, adaptado ao SupoClip. Roda no Windows (fora do
Docker) porque precisa abrir o Chrome de verdade. O site (http://localhost:3107)
manda os cortes escolhidos para cá; o robô copia o vídeo de dentro do container
(docker cp), abre o TikTok Studio, envia, escreve a descrição e agenda.

    postador.bat            -> sobe o serviço em http://127.0.0.1:5680
    postador.bat --login    -> só abre o Chrome para você entrar na conta
"""
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parent.parent          # D:\supoclip
AQUI = Path(__file__).resolve().parent
PERFIL = AQUI / "perfil_tiktok"   # sessão do Chrome da conta do TikTok
ERROS = AQUI / "erros"
VIDEOS = AQUI / "videos"          # cópia local dos cortes enquanto são enviados
REGISTRO = AQUI / "agendados.json"
UPLOAD_URL = "https://www.tiktok.com/tiktokstudio/upload?from=webapp"
PORTA = 5680                       # o postador do auto-cortes usa a 5679
CONTAINER = "supoclip-worker"      # onde ficam os arquivos dos cortes
CAMINHO_OK = re.compile(r"^/app/uploads/clips/[A-Za-z0-9._-]+\.mp4$")
PAUSA = (55, 65)                     # ~1 min entre um envio e o próximo
ADIANTAR = 15                        # agenda a publicação para daqui a 15 min (mínimo do TikTok)

app = Flask(__name__)
fila = queue.Queue()
lock = threading.Lock()
estado = {"logado": None, "conta": None, "atual": None, "pendentes": [], "msg": "", "cancelar": False, "geracao": 0}


# ---------- registro do que já foi agendado (chave = id do corte no SupoClip) ----------

def ler_registro():
    try:
        return json.loads(REGISTRO.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def marcar(clip_id, **dados):
    with lock:
        reg = ler_registro()
        reg[clip_id] = {**reg.get(clip_id, {}), **dados, "quando": time.strftime("%Y-%m-%d %H:%M:%S")}
        REGISTRO.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def copiar_do_container(caminho, nome):
    """Traz o arquivo do corte de dentro do Docker para uma pasta local."""
    if not CAMINHO_OK.match(caminho):
        raise RuntimeError(f"caminho de corte inválido: {caminho}")
    VIDEOS.mkdir(exist_ok=True)
    destino = VIDEOS / re.sub(r"[^A-Za-z0-9._-]", "_", nome or Path(caminho).name)
    r = subprocess.run(["docker", "cp", f"{CONTAINER}:{caminho}", str(destino)],
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not destino.is_file() or destino.stat().st_size == 0:
        raise RuntimeError(f"não consegui copiar o corte do Docker: {(r.stderr or r.stdout).strip()[:200]}")
    return destino


# ---------- robô do navegador (uma thread só: o Playwright não aceita várias) ----------

class Robo:
    def __init__(self):
        self.pw = self.ctx = self.login_proc = self.ultimo_alvo = None

    def abrir(self):
        if self.ctx:
            try:
                self.ctx.pages  # ainda vivo?
                return
            except Exception:
                self.ctx = None
        from playwright.sync_api import sync_playwright
        if self.login_aberto():
            raise RuntimeError("a janela de login do TikTok está aberta: entre na conta e FECHE essa janela")
        self.matar_orfaos()
        if not self.pw:
            self.pw = sync_playwright().start()
        PERFIL.mkdir(parents=True, exist_ok=True)
        self.ctx = self.pw.chromium.launch_persistent_context(
            str(PERFIL), channel="chrome", headless=False, viewport=None,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"],
            ignore_default_args=["--enable-automation", "--no-sandbox"], locale="pt-BR")
        self.ctx.on("close", lambda *_: setattr(self, "ctx", None))

    @staticmethod
    def matar_orfaos():
        """Fecha Chrome do robô que ficou aberto de uma execução anterior (trava o perfil)."""
        import subprocess
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
                        f"Where-Object {{ $_.CommandLine -like '*{PERFIL.name}*' -and $_.CommandLine -like '*remote-debugging*' }} | "
                        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                       capture_output=True)
        time.sleep(1)

    def pagina(self):
        self.abrir()
        return self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()

    @staticmethod
    def ir(page, url):
        """O TikTok demora para carregar: não falha por isso, só espera a navegação começar."""
        try:
            page.goto(url, wait_until="commit", timeout=90000)
        except Exception:
            if url.split("?")[0] not in page.url:
                raise

    def precisa_login(self, page):
        return "login" in page.url or page.locator('input[type="file"]').count() == 0

    def ler_conta(self, page):
        """@ da conta logada (pergunta à própria API do site, com a sessão do navegador)."""
        # The page may still be loading right after navigation: retry a few times.
        for tentativa in range(5):
            try:
                info = page.evaluate("""async () => {
                    const r = await fetch('https://www.tiktok.com/passport/web/account/info/?aid=1988', {credentials: 'include'});
                    return await r.json(); }""")
                nome = ((info or {}).get("data") or {}).get("username")
                if nome:
                    estado["conta"] = "@" + nome
                    return estado["conta"]
            except Exception as e:
                if tentativa == 4:
                    log(f"não consegui ler a conta: {e}")
            time.sleep(3)
        return estado["conta"]

    def checar_login(self):
        page = self.pagina()
        self.ir(page, UPLOAD_URL)
        try:
            page.locator('input[type="file"]').first.wait_for(state="attached", timeout=180000)
            estado["logado"] = True
            estado["msg"] = ""
            self.ler_conta(page)
        except Exception as e:
            log(f"checagem de login falhou: {e}")
            estado["logado"] = False
            estado["conta"] = None
        return estado["logado"]

    def tela_login(self):
        """Login num Chrome NORMAL (sem automação): com o robô no controle o TikTok não carrega o QR.
        O perfil é o mesmo, então a sessão fica salva para o robô usar depois."""
        import subprocess
        if self.ctx:
            try:
                self.ctx.close()
            except Exception:
                pass
            self.ctx = None
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        self.login_proc = subprocess.Popen([chrome, f"--user-data-dir={PERFIL}", "--no-first-run",
                                            "https://www.tiktok.com/login"])

    def login_aberto(self):
        """Verdadeiro enquanto a janela de login (Chrome normal com o perfil do robô) estiver aberta."""
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
                              f"Where-Object {{ $_.CommandLine -like '*{PERFIL.name}*' -and $_.CommandLine -notlike '*remote-debugging*' -and $_.CommandLine -notlike '*--type=*' }}).Count"],
                             capture_output=True, text=True).stdout.strip()
        return out not in ("", "0")

    AVISO = re.compile(r"^\s*(Cancelar|Agora não|Agora nao|Fechar|Entendi|OK|Ignorar|Mais tarde|Pular|Not now|Cancel|Got it|Skip)\s*$", re.I)

    def fechar_avisos(self, page):
        """Fecha os pop-ups do TikTok Studio (ex.: 'Ativar verificações automáticas') que tampam a tela."""
        for _ in range(4):
            fechou = False
            for b in page.get_by_role("button", name=self.AVISO).all():
                try:
                    if b.is_visible():
                        log(f"fechando aviso: {b.inner_text().strip()!r}")
                        b.click(timeout=5000)
                        fechou = True
                        time.sleep(1)
                except Exception:
                    pass
            if not fechou:
                return
            time.sleep(1)

    def escrever_descricao(self, page, texto):
        ed = page.locator('div[contenteditable="true"]').first
        ed.wait_for(timeout=120000)
        self.fechar_avisos(page)
        ed.click(timeout=20000)
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")   # o TikTok preenche com o nome do arquivo
        time.sleep(0.5)
        for i, parte in enumerate(re.split(r"(\s+)", texto)):
            if not parte:
                continue
            if parte.startswith("#") and len(parte) > 1:
                page.keyboard.type(parte, delay=90)      # digitada: vira hashtag de verdade
                time.sleep(1.2)
                page.keyboard.press("Escape")            # fecha a lista de sugestões sem trocar a tag
            elif "\n" in parte:
                for _ in range(parte.count("\n")):
                    page.keyboard.press("Shift+Enter")
            else:
                page.keyboard.insert_text(parte)
            time.sleep(0.05)

    def esperar_envio(self, page, limite=900):
        """Espera o upload do arquivo terminar (o botão de rascunho fica clicável)."""
        fim = time.time() + limite
        pronto = page.get_by_text(re.compile(r"^\s*(Enviado|Carregado|Uploaded)\b"))
        while time.time() < fim:
            try:
                if pronto.first.is_visible():
                    return True
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError("o upload do vídeo não terminou em 15 min")

    MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto",
             "setembro", "outubro", "novembro", "dezembro"]

    def definir_horario(self, page):
        """Marca 'Programar' e escolhe agora + ADIANTAR minutos (arredondado para os próximos 5 min)."""
        import datetime as dt
        page.locator("label").filter(has_text=re.compile(r"^\s*Programar\s*$")).first.click()
        try:   # só aparece na 1ª vez: "Permitir que o vídeo seja salvo para publicação agendada?"
            page.get_by_role("button", name="Permitir").first.click(timeout=4000)
            log("permissão de agendamento concedida")
        except Exception:
            pass
        time.sleep(1)
        alvo = dt.datetime.now() + dt.timedelta(minutes=ADIANTAR)
        resto = (5 - alvo.minute % 5) % 5
        alvo = (alvo + dt.timedelta(minutes=resto)).replace(second=0, microsecond=0)
        if resto == 0 and dt.datetime.now().second > 0:
            alvo += dt.timedelta(minutes=5)   # garante pelo menos ADIANTAR minutos
        if self.ultimo_alvo and alvo < self.ultimo_alvo + dt.timedelta(minutes=5):
            alvo = self.ultimo_alvo + dt.timedelta(minutes=5)   # um por horário: mantém a ordem 01, 02, 03…
        data_cb = page.locator('[aria-label="Select date"]').first
        if alvo.date() != dt.date.today():
            data_cb.click()
            time.sleep(1)
            page.locator(f'button[aria-label*="{alvo.day} de {self.MESES[alvo.month - 1]} de {alvo.year}"]').first.click()
            time.sleep(1)
        hora_cb = page.locator('[aria-label="Select time"]').first
        mostrado = ""
        for tentativa in range(4):   # a lista de horas rola sozinha: às vezes o 1º clique pega a hora errada
            hora_cb.click()
            time.sleep(1.2)
            for rotulo in (f"Hours {alvo:%H}", f"Minutes {alvo:%M}"):
                op = page.locator(f'[aria-label="{rotulo}"]').first
                op.scroll_into_view_if_needed()
                time.sleep(0.6)
                op.click()
                time.sleep(0.8)
            page.keyboard.press("Escape")
            time.sleep(1)
            mostrado = hora_cb.inner_text().strip()
            if f"{alvo:%H:%M}" in mostrado:
                break
            log(f"horário ficou {mostrado}, tentando de novo ({alvo:%H:%M})")
        else:
            raise RuntimeError(f"não consegui ajustar o horário (queria {alvo:%H:%M}, ficou {mostrado})")
        log(f"agendado para {alvo:%d/%m %H:%M}")
        return alvo

    def agendar(self, video, texto):
        if self.login_aberto():
            raise RuntimeError("feche a janela do Chrome de login do TikTok antes de enviar")
        page = self.pagina()
        self.ir(page, UPLOAD_URL)
        try:
            page.locator('input[type="file"]').first.wait_for(state="attached", timeout=180000)
        except Exception:
            estado["logado"] = False
            raise RuntimeError("não está logado na conta do TikTok: clique em 'Entrar no TikTok' na interface")
        estado["logado"] = True
        conta = self.ler_conta(page)
        esperada = conta_esperada()
        if esperada and (conta or "").lower() != esperada.lower():
            raise RuntimeError(f"conta errada logada ({conta or 'desconhecida'}); o esperado é {esperada}. Nada foi enviado.")
        log(f"conta {conta}: enviando arquivo {video.name}")
        page.locator('input[type="file"]').first.set_input_files(str(video))
        self.esperar_envio(page)
        log("upload concluído; escrevendo a descrição")
        self.escrever_descricao(page, texto)
        time.sleep(2)
        self.fechar_avisos(page)
        alvo = self.definir_horario(page)
        if estado["cancelar"]:
            raise RuntimeError("cancelado")
        btn = page.get_by_role("button", name=re.compile(r"^\s*Agendar\s*$")).first
        btn.wait_for(timeout=30000)
        btn.scroll_into_view_if_needed()
        time.sleep(1)
        log("clicando em 'Agendar'")
        btn.click(timeout=20000)
        # confirmação: sai da tela de upload ou aparece aviso de agendamento
        fim = time.time() + 90
        while time.time() < fim:
            try:
                if "upload" not in page.url or page.get_by_text(
                        re.compile(r"agendad|programad|scheduled", re.I)).first.is_visible():
                    self.ultimo_alvo = alvo
                    return alvo
            except Exception:
                pass
            time.sleep(1.5)
        raise RuntimeError("cliquei em 'Agendar' mas o TikTok não confirmou")

    def botoes_visiveis(self):
        if not self.ctx:
            return []
        try:
            return [b.inner_text().strip()[:40] for b in self.pagina().get_by_role("button").all() if b.is_visible()][:40]
        except Exception:
            return []

    def print_erro(self, nome):
        if not self.ctx:
            return None
        try:
            ERROS.mkdir(exist_ok=True)
            p = ERROS / f"{time.strftime('%Y%m%d_%H%M%S')}_{nome}.png"
            self.pagina().screenshot(path=str(p), full_page=True)
            return str(p)
        except Exception:
            return None


def conta_esperada():
    """TIKTOK_CONTA no .env do SupoClip: o robô só envia se essa conta estiver logada."""
    c = os.environ.get("TIKTOK_CONTA", "")
    try:
        for linha in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if linha.strip().startswith("TIKTOK_CONTA="):
                c = linha.split("=", 1)[1].split("#", 1)[0].strip().strip('"').strip("'")
    except Exception:
        pass
    return ("@" + c.lstrip("@")) if c else None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


robo = Robo()


def trabalhador():
    while True:
        tarefa = fila.get()
        if tarefa.get("acao") == "login":
            try:
                robo.tela_login()
                estado["msg"] = "Entre na conta do TikTok no Chrome que abriu e depois FECHE essa janela."
            except Exception as e:
                estado["msg"] = f"erro ao abrir o Chrome: {e}"
            continue
        if tarefa.get("acao") == "checar":
            try:
                robo.checar_login()
            except Exception as e:
                estado["msg"] = f"erro ao abrir o Chrome: {e}"
                log(estado["msg"])
            continue
        clip_id, caminho, nome, texto = tarefa["id"], tarefa["arquivo"], tarefa.get("nome"), tarefa["legenda"]
        if tarefa.get("geracao") != estado.get("geracao", 0):
            continue   # foi cancelada pelo site
        with lock:
            estado["pendentes"] = [k for k in estado["pendentes"] if k != clip_id]
            estado["atual"] = clip_id
        marcar(clip_id, status="enviando", legenda=texto, erro=None)
        video = None
        try:
            video = copiar_do_container(caminho, nome)
            alvo = robo.agendar(video, texto)
            marcar(clip_id, status="agendado", legenda=texto, erro=None, para=alvo.strftime("%d/%m %H:%M"))
            print(f"[ok] agendado para {alvo:%d/%m %H:%M}: {nome or clip_id}", flush=True)
        except Exception as e:
            if str(e) == "cancelado":
                log(f"cancelado antes de agendar: {nome or clip_id}")
                estado["cancelar"] = False
                with lock:
                    reg = ler_registro()
                    reg.pop(clip_id, None)
                    REGISTRO.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            traceback.print_exc()
            log(f"botões na tela: {robo.botoes_visiveis()}")
            marcar(clip_id, status="erro", erro=str(e)[:300], print=robo.print_erro(Path(nome or clip_id).stem[:40]))
        finally:
            estado["atual"] = None
            if video is not None:
                try:
                    video.unlink(missing_ok=True)
                except Exception:
                    pass
        if not fila.empty():
            time.sleep(random.uniform(*PAUSA))


# ---------- API (o site do SupoClip chama via /api/tiktok) ----------

@app.get("/status")
def status():
    ids = [i for i in (request.args.get("ids") or "").split(",") if i]
    with lock:
        out = {k: v for k, v in estado.items()}
    out["esperada"] = conta_esperada()
    reg = ler_registro()
    out["registro"] = {i: reg[i] for i in ids if i in reg}
    return jsonify(out)


@app.post("/enviar")
def enviar():
    itens = (request.get_json(force=True) or {}).get("itens", [])
    n = 0
    for it in itens:
        clip_id = str(it.get("id") or "")
        caminho, nome, leg = str(it.get("arquivo") or ""), str(it.get("nome") or ""), (it.get("legenda") or "").strip()
        if not clip_id or not CAMINHO_OK.match(caminho):
            continue
        with lock:
            if clip_id in estado["pendentes"] or clip_id == estado["atual"]:
                continue
            estado["pendentes"].append(clip_id)
        marcar(clip_id, status="na fila", legenda=leg, erro=None, nome=nome)
        estado["cancelar"] = False
        fila.put({"id": clip_id, "arquivo": caminho, "nome": nome, "legenda": leg, "geracao": estado.get("geracao", 0)})
        n += 1
    return jsonify(ok=True, adicionados=n)


@app.post("/cancelar")
def cancelar():
    """Esvazia a fila; o corte que está no meio do envio é descartado antes de clicar em 'Agendar'."""
    with lock:
        canc = list(estado["pendentes"]) + ([estado["atual"]] if estado["atual"] else [])
        estado["pendentes"] = []
        estado["geracao"] = estado.get("geracao", 0) + 1
        estado["cancelar"] = bool(estado["atual"])
        reg = ler_registro()
        for clip_id in canc:
            if reg.get(clip_id, {}).get("status") in ("na fila", "enviando"):
                del reg[clip_id]
        REGISTRO.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"fila cancelada ({len(canc)} cortes)")
    return jsonify(ok=True, cancelados=len(canc))


@app.post("/login")
def login():
    fila.put({"acao": "login"})
    return jsonify(ok=True)


@app.post("/checar")
def checar():
    fila.put({"acao": "checar"})
    return jsonify(ok=True)


if __name__ == "__main__":
    threading.Thread(target=trabalhador, daemon=True).start()
    if "--login" in sys.argv:
        fila.put({"acao": "login"})
    print(f"Postador do TikTok (SupoClip) em http://127.0.0.1:{PORTA} — conta {conta_esperada() or '(qualquer)'}"
          " — deixe esta janela aberta", flush=True)
    app.run(host="127.0.0.1", port=PORTA, threaded=True)
