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
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent.parent          # D:\supoclip
AQUI = Path(__file__).resolve().parent
PERFIL = AQUI / "perfil_tiktok"   # sessão do Chrome da conta principal do TikTok (e do YouTube)
PERFIS = AQUI / "perfis"          # um perfil do Chrome para cada conta extra do TikTok
ERROS = AQUI / "erros"
VIDEOS = AQUI / "videos"          # cópia local dos cortes enquanto são enviados
REGISTRO = AQUI / "agendados.json"
AVATARES = AQUI / "avatares"      # foto de perfil de cada conta (mostrada no site)
UPLOAD_URL = "https://www.tiktok.com/tiktokstudio/upload?from=webapp"
YT_STUDIO_URL = "https://studio.youtube.com/"
PORTA = 5680                       # o postador do auto-cortes usa a 5679
CONTAINER = "supoclip-worker"      # onde ficam os arquivos dos cortes
CAMINHO_OK = re.compile(r"^/app/uploads/clips/[A-Za-z0-9._-]+\.mp4$")
PAUSA = (55, 65)                     # ~1 min entre um envio e o próximo
PAUSA_YT = (10, 20)                  # no YouTube a pausa entre envios pode ser curta
ADIANTAR = 15                        # agenda a publicação para daqui a 15 min (mínimo do TikTok)
CAPA_SEGUNDO = float(os.environ.get("TIKTOK_CAPA_SEGUNDO", "1.0"))   # frame da capa: balão do gancho já visível
INTERVALO = 5                        # minutos entre um corte e o próximo quando a data é escolhida no site

app = Flask(__name__)
fila = queue.Queue()
lock = threading.Lock()
estado = {"contas": {}, "yt_logado": None, "yt_canal": None, "logado": None, "conta": None, "atual": None, "pendentes": [], "msg": "", "cancelar": False, "geracao": 0}


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


def gerar_capa(video):
    """Frame do corte com o balão do gancho já na tela (o frame 0 ainda não tem),
    para usar como capa no TikTok. Devolve None se não der (aí fica a capa padrão)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log("ffmpeg não encontrado: capa padrão do TikTok")
        return None
    capa = video.with_name(video.stem + "_capa.jpg")
    r = subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-ss", str(CAPA_SEGUNDO), "-i", str(video),
                        "-frames:v", "1", "-q:v", "2", str(capa)], capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not capa.is_file() or capa.stat().st_size == 0:
        log(f"não consegui gerar a capa: {(r.stderr or '').strip()[:200]}")
        return None
    return capa


# ---------- robô do navegador (uma thread só: o Playwright não aceita várias) ----------

def chave_avatar(conta):
    """'@Fulano' -> 'tt-fulano'; o canal do YouTube é 'yt'."""
    return "yt" if conta == "yt" else "tt-" + conta.lstrip("@").lower()


_avatar_urls = {}


def salvar_avatar(page, conta, url):
    """Baixa a foto de perfil (com a sessão do navegador) para AVATARES/<chave>.jpg."""
    if not url or _avatar_urls.get(conta) == url:
        return
    try:
        resp = page.context.request.get(url, timeout=20000)
        if resp.ok and resp.body():
            AVATARES.mkdir(exist_ok=True)
            (AVATARES / f"{chave_avatar(conta)}.jpg").write_bytes(resp.body())
            _avatar_urls[conta] = url
    except Exception as e:
        log(f"não consegui baixar a foto de {conta}: {str(e).splitlines()[0]}")


class Robo:
    def __init__(self):
        self.pw = self.ctx = self.login_proc = self.ultimo_alvo = None
        self.perfil = PERFIL

    def abrir(self, perfil=None):
        perfil = perfil or PERFIL
        if self.ctx and self.perfil != perfil:   # outra conta: fecha o Chrome da conta anterior
            try:
                self.ctx.close()
            except Exception:
                pass
            self.ctx = None
        self.perfil = perfil
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
        perfil.mkdir(parents=True, exist_ok=True)
        self.ctx = self.pw.chromium.launch_persistent_context(
            str(perfil), channel="chrome", headless=False, viewport=None,
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"],
            ignore_default_args=["--enable-automation", "--no-sandbox"], locale="pt-BR")
        self.ctx.on("close", lambda *_: setattr(self, "ctx", None))

    @staticmethod
    def matar_orfaos():
        """Fecha Chrome do robô que ficou aberto de uma execução anterior (trava o perfil)."""
        import subprocess
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        "Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
                        f"Where-Object {{ $_.CommandLine -like '*postador\\perf*' -and $_.CommandLine -like '*remote-debugging*' }} | "
                        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"],
                       capture_output=True)
        time.sleep(1)

    def pagina(self, perfil=None):
        """Página viva do robô (no perfil pedido; padrão: o atual). Se a janela do Chrome foi
        fechada, o contexto antigo está morto (mas ainda parece vivo): descarta e abre de novo."""
        perfil = perfil or self.perfil
        for _ in range(2):
            self.abrir(perfil)
            try:
                page = self.ctx.pages[0] if self.ctx.pages else self.ctx.new_page()
                page.evaluate("1")
                return page
            except Exception as e:
                log(f"Chrome do robô estava fechado, abrindo de novo ({str(e).splitlines()[0]})")
                try:
                    self.ctx.close()
                except Exception:
                    pass
                self.ctx = None
        raise RuntimeError("não consegui abrir o Chrome do robô")

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

    def ler_conta(self, page, esperada=None):
        """@ da conta logada (pergunta à própria API do site, com a sessão do navegador)."""
        esperada = esperada or conta_principal()
        # The page may still be loading right after navigation: retry a few times.
        for tentativa in range(5):
            try:
                info = page.evaluate("""async () => {
                    const r = await fetch('https://www.tiktok.com/passport/web/account/info/?aid=1988', {credentials: 'include'});
                    return await r.json(); }""")
                dados = (info or {}).get("data") or {}
                nome = dados.get("username")
                if nome:
                    situacao(esperada, logado=True, conta="@" + nome)
                    # A foto do PERFIL fica no canto do TikTok Studio; o avatar_url da conta
                    # de login costuma ser a letra padrão. Ele só vale como reserva.
                    salvar_avatar(page, esperada, self.foto_perfil_tiktok(page) or dados.get("avatar_url"))
                    return "@" + nome
            except Exception as e:
                if tentativa == 4:
                    log(f"não consegui ler a conta: {e}")
            time.sleep(3)
        return estado["contas"].get(esperada, {}).get("conta")

    @staticmethod
    def foto_perfil_tiktok(page):
        """src da foto de perfil no cabeçalho do TikTok Studio (canto superior direito)."""
        for _ in range(5):
            try:
                src = page.evaluate("""() => {
                    const imgs = [...document.querySelectorAll('img')].filter(i => {
                        const r = i.getBoundingClientRect();
                        return /tiktokcdn|ibyteimg/.test(i.src || '') && r.top < 90 && r.width >= 20 && r.width <= 80
                            && r.left > window.innerWidth / 2;
                    });
                    return imgs.length ? imgs[imgs.length - 1].src : null; }""")
                if src:
                    return src
            except Exception:
                pass
            time.sleep(2)
        return None

    def checar_login(self, conta=None):
        conta = conta or conta_principal()
        page = self.pagina(perfil_da_conta(conta))
        self.ir(page, UPLOAD_URL)
        try:
            page.locator('input[type="file"]').first.wait_for(state="attached", timeout=180000)
            situacao(conta, logado=True)
            estado["msg"] = ""
            self.ler_conta(page, conta)
        except Exception as e:
            log(f"checagem de login ({conta}) falhou: {str(e).splitlines()[0]}")
            situacao(conta, logado=False, conta=None)
        return estado["contas"].get(conta, {}).get("logado")

    def tela_login(self, url="https://www.tiktok.com/login", perfil=None):
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
        perfil = perfil or PERFIL
        perfil.mkdir(parents=True, exist_ok=True)
        self.login_proc = subprocess.Popen([chrome, f"--user-data-dir={perfil}", "--no-first-run", url])

    def login_aberto(self):
        """Verdadeiro enquanto a janela de login (Chrome normal com o perfil do robô) estiver aberta."""
        import subprocess
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" | "
                              f"Where-Object {{ $_.CommandLine -like '*postador\\perf*' -and $_.CommandLine -notlike '*remote-debugging*' -and $_.CommandLine -notlike '*--type=*' }}).Count"],
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

    def definir_capa(self, page, capa):
        """Editar capa -> Carregar capa (imagem) -> Salvar. Se falhar, segue com a capa padrão."""
        try:
            page.get_by_text("Editar capa").first.click(timeout=20000)
            editor = page.locator(".cover-editor-container").first
            editor.wait_for(timeout=30000)
            time.sleep(2)
            editor.locator('input[type="file"][accept*="image"]').first.set_input_files(str(capa))
            time.sleep(4)   # o TikTok processa a imagem antes de liberar o Salvar
            salvar = editor.get_by_role("button", name=re.compile(r"^\s*Salvar\s*$")).first
            salvar.wait_for(timeout=30000)
            salvar.click(timeout=20000)
            editor.wait_for(state="hidden", timeout=60000)
            log("capa com o gancho aplicada")
            return True
        except Exception as e:
            log(f"não consegui trocar a capa ({str(e).splitlines()[0]}); seguindo com a capa padrão")
            try:
                page.get_by_role("button", name=re.compile(r"^\s*Cancelar\s*$")).first.click(timeout=5000)
            except Exception:
                pass
            return False

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

    def escolher_data(self, page, alvo):
        """Abre o calendário do TikTok e clica no dia (passa para o mês seguinte se precisar)."""
        rotulo = f", {alvo.day} de {self.MESES[alvo.month - 1]} de {alvo.year}"
        data_cb = page.locator('[aria-label="Select date"]').first
        data_cb.scroll_into_view_if_needed()
        for _ in range(3):
            data_cb.click()
            time.sleep(1.2)
            if data_cb.get_attribute("aria-expanded") == "true":
                break
        for _ in range(3):
            dia = page.locator(f'button.DatePicker__day[aria-label$="{rotulo}"]:not([data-outside="true"])').first
            if dia.count():
                if dia.is_disabled():
                    raise RuntimeError(f"o TikTok não deixa agendar para {alvo:%d/%m/%Y} (fora do prazo permitido)")
                dia.click()
                time.sleep(1)
                return
            page.locator('[aria-label="Next month"]').first.click()
            time.sleep(1)
        raise RuntimeError(f"não achei o dia {alvo:%d/%m/%Y} no calendário do TikTok")

    def definir_horario(self, page, para=None):
        """Marca 'Programar' e escolhe o horário: o pedido pelo site (`para`) ou, sem ele,
        agora + ADIANTAR minutos (arredondado para os próximos 5 min)."""
        import datetime as dt
        page.locator("label").filter(has_text=re.compile(r"^\s*Programar\s*$")).first.click()
        try:   # só aparece na 1ª vez: "Permitir que o vídeo seja salvo para publicação agendada?"
            page.get_by_role("button", name="Permitir").first.click(timeout=4000)
            log("permissão de agendamento concedida")
        except Exception:
            pass
        time.sleep(1)
        minimo = dt.datetime.now() + dt.timedelta(minutes=ADIANTAR)
        resto = (5 - minimo.minute % 5) % 5
        minimo = (minimo + dt.timedelta(minutes=resto)).replace(second=0, microsecond=0)
        if resto == 0 and dt.datetime.now().second > 0:
            minimo += dt.timedelta(minutes=5)   # garante pelo menos ADIANTAR minutos
        if para:
            alvo = para
            if alvo < minimo:   # a fila atrasou: o TikTok exige ADIANTAR min, então vai para o 1º horário válido
                log(f"horário pedido {alvo:%d/%m %H:%M} já está perto demais; usando {minimo:%d/%m %H:%M}")
                alvo = minimo
        else:
            alvo = minimo
            if self.ultimo_alvo and alvo < self.ultimo_alvo + dt.timedelta(minutes=5):
                alvo = self.ultimo_alvo + dt.timedelta(minutes=5)   # um por horário: mantém a ordem 01, 02, 03…
        if alvo.date() != dt.date.today():
            self.escolher_data(page, alvo)
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

    VERIF_CONTEUDO = re.compile(r"verifica\w* de conte|content check", re.I)

    def desligar_verificacao(self, page):
        """Desliga a 'verificação de conteúdo simples' do TikTok Studio. Ela é opcional,
        leva até ~10 min por vídeo e trava o Agendar em 'Continuar publicando?'."""
        # Acha o texto, sobe até o bloco que tem uma chave e lê o estado pelo que existir
        # (input.checked, aria-checked, classe "checked"/"on"). Devolve um diagnóstico.
        js = r"""() => {
          const re = /verifica\S* de conte|content check/i;
          const marca = [...document.querySelectorAll('body *')].find(e =>
              e.children.length === 0 && re.test(e.textContent || ''));
          if (!marca) return {achou: false};
          const SEL = '[data-layout=switch-root] [aria-checked], [aria-checked], [role=switch], input[type=checkbox]';
          let bloco = marca, chave = null;
          for (let i = 0; i < 8 && bloco && !chave; i++) { bloco = bloco.parentElement; chave = bloco && bloco.querySelector(SEL); }
          if (!chave) return {achou: true, chave: false, html: (marca.parentElement?.parentElement?.outerHTML || '').slice(0, 1500)};
          document.querySelectorAll('[data-robo-chave]').forEach(e => e.removeAttribute('data-robo-chave'));
          chave.setAttribute('data-robo-chave', '1');   // o Playwright clica nela com o mouse de verdade
          const ligada = () => {
            const a = chave.getAttribute('aria-checked');
            if (a !== null) return a === 'true';
            if (chave.matches('input')) return chave.checked;
            return chave.getAttribute('data-state') === 'checked';
          };
          return {achou: true, chave: true, ligada: ligada(), html: chave.outerHTML.slice(0, 600)};
        }"""
        try:
            info = {}
            for _ in range(15):   # a seção "Verificações" aparece alguns segundos depois do upload
                info = page.evaluate(js)
                if info.get("chave"):
                    break
                time.sleep(1)
            if not info.get("chave"):
                log(f"TikTok: não achei a chave da verificação de conteúdo ({json.dumps(info, ensure_ascii=False)[:1600]})")
                return
            if not info.get("ligada"):
                log("TikTok: verificação de conteúdo simples já está desligada")
                return
            # clique de mouse de verdade na chave marcada pelo JS (click() via JS não liga no React do TikTok)
            chave = page.locator("[data-robo-chave]").first
            chave.evaluate("e => e.scrollIntoView({block: 'center'})")
            time.sleep(0.5)
            chave.click(timeout=5000, force=True)
            time.sleep(1.5)
            depois = page.evaluate(js)
            if depois.get("ligada"):
                log(f"TikTok: não consegui desligar a verificação de conteúdo ({depois.get('html')})")
            else:
                log("TikTok: verificação de conteúdo simples desligada")
        except Exception as e:
            log(f"TikTok: não consegui desligar a verificação ({str(e).splitlines()[0]})")

    def agendar(self, video, texto, para=None, capa=None, esperada=None):
        esperada = esperada or conta_principal()
        if self.login_aberto():
            raise RuntimeError("feche a janela do Chrome de login do TikTok antes de enviar")
        page = self.pagina(perfil_da_conta(esperada))
        self.ir(page, UPLOAD_URL)
        try:
            page.locator('input[type="file"]').first.wait_for(state="attached", timeout=180000)
        except Exception:
            situacao(esperada, logado=False)
            raise RuntimeError(f"não está logado na conta {esperada} do TikTok: clique em 'Entrar' dessa conta no site")
        situacao(esperada, logado=True)
        conta = self.ler_conta(page, esperada)
        if esperada and (conta or "").lower() != esperada.lower():
            raise RuntimeError(f"conta errada logada ({conta or 'desconhecida'}); o esperado é {esperada}. Nada foi enviado.")
        log(f"conta {conta}: enviando arquivo {video.name}")
        page.locator('input[type="file"]').first.set_input_files(str(video))
        self.esperar_envio(page)
        self.desligar_verificacao(page)   # já no início: a checagem nem chega a rodar
        log("upload concluído; escrevendo a descrição")
        if capa:
            self.definir_capa(page, capa)
            self.fechar_avisos(page)
        self.escrever_descricao(page, texto)
        time.sleep(2)
        self.fechar_avisos(page)
        alvo = self.definir_horario(page, para)
        if estado["cancelar"]:
            raise RuntimeError("cancelado")
        btn = page.get_by_role("button", name=re.compile(r"^\s*Agendar\s*$")).first
        btn.wait_for(timeout=30000)
        btn.scroll_into_view_if_needed()
        # Garante a verificação desligada antes de Agendar (o TikTok pode religar a
        # chave depois da descrição/horário); com ela rodando abre "Continuar publicando?".
        self.desligar_verificacao(page)
        time.sleep(1)
        continuar = page.get_by_text(re.compile(r"Continuar publicando|Continue to post", re.I)).first
        cancelar = page.get_by_role("button", name=re.compile(r"^\s*(Cancelar|Cancel)\s*$")).first
        tentativas = 0
        while True:
            tentativas += 1
            log("clicando em 'Agendar'" + (f" ({tentativas}ª vez)" if tentativas > 1 else ""))
            btn.click(timeout=20000)
            # confirmação: sai da tela de upload ou aparece aviso de agendamento
            fim = time.time() + 90
            aviso = False
            while time.time() < fim:
                try:
                    if continuar.is_visible():
                        # Verificação ainda rodando: NÃO clica em "Publicar agora" (poderia
                        # publicar na hora). Cancela, espera e tenta Agendar de novo.
                        aviso = True
                        log("TikTok: 'Continuar publicando?' (verificação em andamento); cancelando e tentando de novo")
                        cancelar.click(timeout=5000)
                        time.sleep(20)
                        break
                    if "upload" not in page.url or page.get_by_text(
                            re.compile(r"agendad|programad|scheduled", re.I)).first.is_visible():
                        if para is None:   # só o modo "daqui 15 min" encadeia; data escolhida não empurra os próximos
                            self.ultimo_alvo = alvo
                        return alvo
                except Exception:
                    pass
                time.sleep(1.5)
            if not aviso or tentativas >= 4:
                break
        raise RuntimeError("cliquei em 'Agendar' mas o TikTok não confirmou")

    # ---------- YouTube Shorts (YouTube Studio) ----------

    MESES_ABREV = ["jan.", "fev.", "mar.", "abr.", "mai.", "jun.", "jul.", "ago.", "set.", "out.", "nov.", "dez."]

    def abrir_studio(self):
        page = self.pagina(PERFIL)
        self.ir(page, YT_STUDIO_URL)
        try:
            page.locator("#upload-icon").first.wait_for(timeout=90000)
        except Exception:
            estado["yt_logado"] = False
            estado["yt_canal"] = None
            raise RuntimeError("não está logado no YouTube: clique em 'Entrar no YouTube' no site")
        estado["yt_logado"] = True
        try:
            estado["yt_canal"] = page.locator("#entity-name").first.inner_text(timeout=10000).strip() or None
        except Exception:
            pass
        try:
            src = page.evaluate("""() => { const i = [...document.querySelectorAll('#left-nav img')]
                .find(i => (i.src || '').includes('ggpht.com')); return i ? i.src : null; }""")
            if src:
                salvar_avatar(page, "yt", re.sub(r"=s\d+.*$", "=s176-c-k-c0x00ffffff-no-rj", src))
        except Exception:
            pass
        return page

    def checar_youtube(self):
        try:
            self.abrir_studio()
        except Exception as e:
            log(f"checagem do YouTube: {str(e).splitlines()[0]}")
        return estado["yt_logado"]

    def esperar_upload_yt(self, page, limite=1200):
        """Espera o arquivo terminar de subir (fechar a janela antes perde o vídeo)."""
        fim = time.time() + limite
        # Só segue com sinal claro de fim do envio ("Envio concluído", a verificação ou o
        # processamento já começaram); enquanto sobe o rodapé mostra a porcentagem do envio.
        pronto = re.compile(r"Envio conclu|Upload complete|Verifica|Checks|Processa|processamento|Processing", re.I)
        while time.time() < fim:
            try:
                txt = page.locator("ytcp-video-upload-progress").first.inner_text(timeout=5000)
                if txt and pronto.search(txt):
                    log(f"YouTube: {txt.strip()[:80]}")
                    return txt.strip()
            except Exception:
                pass
            time.sleep(3)
        raise TimeoutError("o upload no YouTube não terminou em 20 min")

    def digitar_campo(self, page, seletor, texto):
        caixa = page.locator(f"{seletor} #textbox").first
        caixa.wait_for(timeout=60000)
        caixa.click()
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        time.sleep(0.3)
        page.keyboard.insert_text(texto)
        time.sleep(0.5)

    def restringir_idade_yt(self, page):
        """Restrição de idade (avançado) -> 'Sim, quero restringir ... mais de 18 anos'."""
        radio = page.locator('tp-yt-paper-radio-button[name="VIDEO_AGE_RESTRICTION_SELF"]').first
        if not radio.is_visible():
            botao = page.get_by_role("button", name=re.compile(r"^\s*Restrição de idade")).first
            botao.scroll_into_view_if_needed()
            botao.click()
            time.sleep(1.5)
        radio.scroll_into_view_if_needed()
        radio.click()
        time.sleep(1)
        if radio.get_attribute("aria-checked") != "true":
            raise RuntimeError("não consegui marcar a restrição de idade no YouTube")
        log("YouTube: restrição de idade (18+) marcada")

    def miniatura_yt(self, page, capa):
        """Miniatura do Short = frame com o balão do gancho. Se falhar, fica a automática."""
        try:
            page.locator('ytcp-uploads-dialog input#file-loader').first.set_input_files(str(capa))
            time.sleep(4)
            erro = page.locator("ytcp-video-thumbnail-editor").first.inner_text(timeout=5000)
            if "verifi" in erro.lower() and "telefone" in erro.lower():
                log("YouTube: a miniatura personalizada exige canal verificado; ficou a automática")
                return False
            log("YouTube: miniatura com o gancho enviada")
            return True
        except Exception as e:
            log(f"YouTube: não consegui trocar a miniatura ({str(e).splitlines()[0]}); ficou a automática")
            return False

    def agendar_youtube(self, video, titulo, descricao, para=None, capa=None):
        import datetime as dt
        if self.login_aberto():
            raise RuntimeError("feche a janela do Chrome de login antes de enviar")
        page = self.abrir_studio()
        esperado = canal_esperado()
        canal = estado.get("yt_canal")
        if esperado and (canal or "").lower() != esperado.lower():
            raise RuntimeError(f"canal errado no YouTube ({canal or 'desconhecido'}); o esperado é {esperado}. Nada foi enviado.")
        alvo = para or horario_minimo()
        if alvo < dt.datetime.now() + dt.timedelta(minutes=2):
            alvo = horario_minimo()
        log(f"YouTube ({canal}): enviando {video.name}")
        page.locator("#upload-icon").first.click()
        time.sleep(2)
        page.locator('input[type="file"]').first.set_input_files(str(video))
        page.locator("#title-textarea").first.wait_for(timeout=120000)
        time.sleep(3)
        self.digitar_campo(page, "#title-textarea", titulo[:100])
        self.digitar_campo(page, "#description-textarea", descricao[:5000])
        if capa:
            self.miniatura_yt(page, capa)
        page.locator('tp-yt-paper-radio-button[name="VIDEO_MADE_FOR_KIDS_NOT_MFK"]').first.click()
        time.sleep(1)
        palavrao = tem_palavrao(f"{titulo}\n{descricao}")
        if restringir_idade(palavrao):
            log(f"YouTube: restrição de idade por causa de {palavrao!r}" if palavrao else "YouTube: restrição de idade (sempre ligada)")
            self.restringir_idade_yt(page)
        for _ in range(3):
            page.locator("#next-button").first.click()
            time.sleep(2.5)
        # Abre a seção "Programar" e confere que o campo de data apareceu (às vezes o 1º clique
        # chega antes da tela de Visibilidade terminar de montar e a seção continua fechada).
        expandir = page.locator("#second-container-expand-button").first
        expandir.wait_for(state="visible", timeout=60000)
        trigger = page.locator("#datepicker-trigger").first
        for _ in range(4):
            if trigger.is_visible():
                break
            expandir.scroll_into_view_if_needed()
            expandir.click()
            try:
                trigger.wait_for(state="visible", timeout=5000)
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("não consegui abrir a opção 'Programar' no YouTube")
        time.sleep(1)
        data_txt = f"{alvo.day} de {self.MESES_ABREV[alvo.month - 1]} de {alvo.year}"
        page.locator("#datepicker-trigger").first.click()
        time.sleep(1.5)
        page.locator("ytcp-date-picker tp-yt-paper-input input").first.click()
        page.keyboard.press("Control+A")
        page.keyboard.type(data_txt)
        page.keyboard.press("Enter")
        time.sleep(1.5)
        mostrada = page.locator("#datepicker-trigger").first.inner_text().strip()
        if mostrada != data_txt:
            raise RuntimeError(f"não consegui ajustar a data no YouTube (queria {data_txt}, ficou {mostrada})")
        hora = page.locator("#time-of-day-container input").first
        hora.click()
        page.keyboard.press("Control+A")
        page.keyboard.type(f"{alvo:%H:%M}")
        page.keyboard.press("Enter")
        time.sleep(1.5)
        if hora.input_value().strip() != f"{alvo:%H:%M}":
            raise RuntimeError(f"não consegui ajustar a hora no YouTube (queria {alvo:%H:%M}, ficou {hora.input_value()})")
        log(f"YouTube: programado para {alvo:%d/%m %H:%M}; esperando o upload terminar")
        self.esperar_upload_yt(page)
        if estado["cancelar"]:
            raise RuntimeError("cancelado")
        # Tira o foco do campo de hora: com a lista de horários aberta, o 1º clique em
        # 'Programar' só fecha a lista e o vídeo fica como rascunho ("Salvo como privado").
        try:
            page.locator("ytcp-uploads-dialog #dialog-title").first.click(timeout=3000)
        except Exception:
            pass
        time.sleep(1)
        done = page.locator("#done-button").first
        ok = re.compile(r"^\s*(Ok|OK|Entendi|Got it)\s*$")
        confirmacao_re = re.compile(
            r"ficará público em|vai se tornar público em|will be public on|Vídeo programado|Video scheduled|Processando vídeo|Video processing",
            re.I)
        # Confirmação só vale FORA da janela de envio: dentro dela há textos parecidos que
        # davam falso "ok"; e o aviso "Vídeo programado" abre por cima com a janela de envio
        # ainda aberta embaixo (o botão 'Programar' continua "visível" mas coberto).
        # Título exato da janela de sucesso ("Vídeo programado"; "Processando vídeo" quando
        # ainda processa): nenhum texto da janela de envio é exatamente igual a esses.
        js_confirmado = """() => {
            const re = /^(Vídeo programado|Video scheduled|Processando vídeo|Video processing)$/i;
            for (const el of document.querySelectorAll('h1, h2, yt-formatted-string, span, div')) {
                const r = el.getBoundingClientRect();
                if (!r.width || !r.height) continue;
                const t = (el.innerText || '').trim();
                if (re.test(t)) {
                    const box = el.closest('tp-yt-paper-dialog, ytcp-dialog') || el.parentElement;
                    return (box && box.innerText || t).replace(/\\s+/g, ' ').trim();
                }
            }
            return null;
        }"""
        fim = time.time() + 180
        cliques, ultimo_clique, fechado_desde = 0, 0.0, None
        while time.time() < fim:
            try:
                achado = page.evaluate(js_confirmado)
                if achado:
                    log(f"YouTube: agendamento confirmado ({achado[:120]})")
                    break
                # "Ainda estamos verificando seu conteúdo": aviso com botão Ok quando a
                # verificação de direitos ainda está rodando; o Ok confirma o agendamento.
                for b in page.get_by_role("button", name=ok).all():
                    if b.is_visible():
                        log("YouTube: aviso de verificação em andamento; clicando em Ok")
                        b.click(timeout=5000)
                        time.sleep(2)
                # Enquanto o botão 'Programar' do envio estiver na tela, NÃO foi agendado
                # (antes um texto da própria janela de envio dava falso "ok").
                if done.is_visible():
                    fechado_desde = None
                    if cliques < 4 and time.time() - ultimo_clique > 8:
                        if cliques:
                            log(f"YouTube: 'Programar' ainda aberto; clicando de novo ({cliques + 1}ª vez)")
                        else:
                            log("YouTube: clicando em 'Programar'")
                        cliques += 1
                        ultimo_clique = time.time()
                        done.click(timeout=5000)
                else:
                    fechado_desde = fechado_desde or time.time()
                    # Janela de envio fechada e estável por 5 s.
                    if time.time() - fechado_desde > 5:
                        log("YouTube: janela de envio fechou; agendamento confirmado")
                        break
            except Exception:
                pass
            time.sleep(1.5)
        else:
            raise RuntimeError("cliquei em 'Programar' mas o YouTube não confirmou (deve ter ficado como rascunho)")
        time.sleep(2)
        try:
            page.get_by_role("button", name=re.compile(r"^\s*(Fechar|Close)\s*$")).first.click(timeout=5000)
        except Exception:
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
        return alvo

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


def horario_minimo():
    """Agora + ADIANTAR minutos, no próximo múltiplo de 5."""
    import datetime as dt
    agora = dt.datetime.now()
    minimo = agora + dt.timedelta(minutes=ADIANTAR)
    resto = (5 - minimo.minute % 5) % 5
    minimo = (minimo + dt.timedelta(minutes=resto)).replace(second=0, microsecond=0)
    if resto == 0 and agora.second > 0:
        minimo += dt.timedelta(minutes=5)
    return minimo


# Palavrões (sem acento, minúsculos). Casam como palavra inteira ou começo de palavra
# ("porra", "porrada" não; "caralho", "caralhada" sim — veja PALAVRAO_RE).
PALAVROES = [
    "porra", "caralh", "merda", "puta", "puto", "putaria", "foda", "fodase", "foder", "fodido", "fudido",
    "fuder", "fodeu", "fudeu", "buceta", "boceta", "cuzao", "cuzinho", "arrombad", "desgracad", "cacete",
    "bosta", "piranha", "vagabund", "viado", "otario", "babaca", "fdp", "pqp", "vsf", "vtnc", "tnc", "krl",
    "crl", "prr", "filho da puta", "vai se foder", "vai tomar no cu", "tomar no cu", "pau no cu", "cu",
]
_RAIZES = ("caralh", "arrombad", "desgracad", "vagabund")


def _sem_acento(texto):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn")


PALAVRAO_RE = re.compile(
    r"(?<![a-z0-9])(" + "|".join(
        re.escape(p) + (r"[a-z]*" if p in _RAIZES else "") for p in sorted(PALAVROES, key=len, reverse=True)
    ) + r")(?![a-z0-9])")


def tem_palavrao(texto):
    """O primeiro palavrão no texto (título + descrição), ou None. Hashtags contam sem o '#'."""
    achado = PALAVRAO_RE.search(_sem_acento(texto or ""))
    return achado.group(1) if achado else None


def restringir_idade(palavrao=None):
    """YOUTUBE_RESTRICAO_IDADE no .env: auto (padrão) = só quando o texto tem palavrão;
    true = sempre; false = nunca."""
    v = os.environ.get("YOUTUBE_RESTRICAO_IDADE", "auto")
    try:
        for linha in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if linha.strip().startswith("YOUTUBE_RESTRICAO_IDADE="):
                v = linha.split("=", 1)[1].split("#", 1)[0].strip().strip('"').strip("'")
    except Exception:
        pass
    v = v.lower()
    if v in ("0", "false", "nao", "não", "no"):
        return False
    if v in ("1", "true", "sim", "yes", "sempre"):
        return True
    return bool(palavrao)


def canal_esperado():
    """YOUTUBE_CANAL no .env: o robô só envia se esse canal estiver aberto no YouTube Studio."""
    c = os.environ.get("YOUTUBE_CANAL", "")
    try:
        for linha in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if linha.strip().startswith("YOUTUBE_CANAL="):
                c = linha.split("=", 1)[1].split("#", 1)[0].strip().strip('"').strip("'")
    except Exception:
        pass
    return c or None


def titulo_youtube(legenda, gancho):
    """Título do Short: o gancho (sem as hashtags) + #shorts, até 100 caracteres."""
    base = (gancho or re.split(r"\s#", " " + legenda, 1)[0]).strip() or "Corte"
    return (base[:90] + " #shorts").strip()


def ler_horario(texto):
    """'2026-11-21T18:00' (horário local, vindo do site) -> datetime, ou None."""
    import datetime as dt
    try:
        return dt.datetime.strptime(str(texto), "%Y-%m-%dT%H:%M") if texto else None
    except ValueError:
        return None


def ler_env(nome, padrao=""):
    v = os.environ.get(nome, padrao)
    try:
        for linha in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
            if linha.strip().startswith(nome + "="):
                v = linha.split("=", 1)[1].split("#", 1)[0].strip().strip('"').strip("'")
    except Exception:
        pass
    return v


def contas_tiktok():
    """Contas do TikTok: TIKTOK_CONTAS=@principal,@outra (a 1ª usa o perfil antigo).
    Sem ela, vale a TIKTOK_CONTA de sempre."""
    bruto = ler_env("TIKTOK_CONTAS") or ler_env("TIKTOK_CONTA")
    contas = []
    for c in re.split(r"[,;\s]+", bruto):
        c = c.strip().lstrip("@")
        if re.fullmatch(r"[A-Za-z0-9._]{2,40}", c or "") and ("@" + c).lower() not in [x.lower() for x in contas]:
            contas.append("@" + c)
    return contas


def conta_principal():
    contas = contas_tiktok()
    return contas[0] if contas else None


def conta_esperada():
    """Compatibilidade: a conta principal do TikTok."""
    return conta_principal()


def achar_conta(nome):
    """'@Fulano' -> o @ como está no .env (ou None se não for uma conta configurada)."""
    nome = (nome or "").strip()
    if not nome:
        return conta_principal()
    for c in contas_tiktok():
        if c.lower() == ("@" + nome.lstrip("@")).lower():
            return c
    return None


def perfil_da_conta(conta):
    """Perfil do Chrome de cada conta: a principal usa o antigo (perfil_tiktok)."""
    if not conta or conta == conta_principal():
        return PERFIL
    return PERFIS / conta.lstrip("@").lower()


def chave_tiktok(conta, clip_id):
    """Chave no registro: a conta principal mantém só o id (como sempre foi)."""
    return clip_id if conta == conta_principal() else f"tt:{conta.lower()}:{clip_id}"


def situacao(quem, **dados):
    """Login/conta lida de cada conta do TikTok (e os campos antigos para a principal)."""
    if not quem:
        return
    with lock:
        estado["contas"].setdefault(quem, {"logado": None, "conta": None}).update(dados)
        if quem == conta_principal():
            for k in ("logado", "conta"):
                if k in dados:
                    estado[k] = dados[k]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


robo = Robo()


def trabalhador():
    while True:
        tarefa = fila.get()
        if tarefa.get("acao") == "login":
            try:
                conta = tarefa.get("conta") or conta_principal()
                robo.tela_login(perfil=perfil_da_conta(conta))
                estado["msg"] = f"Entre na conta {conta} do TikTok no Chrome que abriu e depois FECHE essa janela."
            except Exception as e:
                estado["msg"] = f"erro ao abrir o Chrome: {e}"
            continue
        if tarefa.get("acao") == "login_youtube":
            try:
                robo.tela_login(YT_STUDIO_URL)
                estado["msg"] = "Entre na conta do canal no Chrome que abriu (YouTube Studio) e depois FECHE essa janela."
            except Exception as e:
                estado["msg"] = f"erro ao abrir o Chrome: {e}"
            continue
        if tarefa.get("acao") == "checar_youtube":
            try:
                robo.checar_youtube()
            except Exception as e:
                estado["msg"] = f"erro ao abrir o Chrome: {e}"
            continue
        if tarefa.get("acao") == "checar":
            try:
                robo.checar_login(tarefa.get("conta"))
            except Exception as e:
                estado["msg"] = f"erro ao abrir o Chrome: {e}"
                log(estado["msg"])
            continue
        clip_id, caminho, nome, texto = tarefa["chave"], tarefa["arquivo"], tarefa.get("nome"), tarefa["legenda"]
        youtube = tarefa.get("plataforma") == "youtube"
        if tarefa.get("geracao") != estado.get("geracao", 0):
            continue   # foi cancelada pelo site
        with lock:
            estado["pendentes"] = [k for k in estado["pendentes"] if k != clip_id]
            estado["atual"] = clip_id
        marcar(clip_id, status="enviando", legenda=texto, erro=None)
        video = None
        try:
            video = copiar_do_container(caminho, nome)
            if youtube:
                alvo = robo.agendar_youtube(video, titulo_youtube(texto, tarefa.get("gancho")), texto,
                                            ler_horario(tarefa.get("para")), gerar_capa(video))
            else:
                capa = gerar_capa(video)
                alvo = robo.agendar(video, texto, ler_horario(tarefa.get("para")), capa, tarefa.get("conta"))
            marcar(clip_id, status="agendado", legenda=texto, erro=None, para=alvo.strftime("%d/%m %H:%M"))
            print(f"[ok] {'YouTube' if youtube else 'TikTok ' + str(tarefa.get('conta'))} agendado para {alvo:%d/%m %H:%M}: {nome or clip_id}", flush=True)
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
                    video.with_name(video.stem + "_capa.jpg").unlink(missing_ok=True)
                except Exception:
                    pass
        if not fila.empty():
            time.sleep(random.uniform(*(PAUSA_YT if youtube else PAUSA)))


# ---------- API (o site do SupoClip chama via /api/tiktok) ----------

@app.get("/status")
def status():
    ids = [i for i in (request.args.get("ids") or "").split(",") if i]
    with lock:
        out = {k: v for k, v in estado.items()}
    out["esperada"] = conta_principal()
    out["yt_esperado"] = canal_esperado()
    reg = ler_registro()
    out["registro"] = {i: reg[i] for i in ids if i in reg}
    out["contas_tiktok"] = [{"conta": c, "principal": c == conta_principal(), **{"logado": None, "lida": None},
                             **{k: v for k, v in (out["contas"].get(c) or {}).items() if k == "logado"},
                             "lida": (out["contas"].get(c) or {}).get("conta")} for c in contas_tiktok()]
    out["avatares"] = {f.stem: int(f.stat().st_mtime) for f in AVATARES.glob("*.jpg")} if AVATARES.exists() else {}
    out["registro_tt"] = {c: {i: reg[chave_tiktok(c, i)] for i in ids if chave_tiktok(c, i) in reg}
                          for c in contas_tiktok()}
    out["registro_yt"] = {i: reg["yt:" + i] for i in ids if "yt:" + i in reg}
    return jsonify(out)


@app.post("/enviar")
def enviar():
    import datetime as dt
    corpo = request.get_json(force=True) or {}
    itens = corpo.get("itens", [])
    # Data/hora escolhida no site: o 1º corte sai nela e os próximos a cada INTERVALO min.
    inicio = ler_horario(corpo.get("inicio"))
    if corpo.get("inicio") and not inicio:
        return jsonify(ok=False, erro="data/hora inválida"), 400
    if inicio and inicio < dt.datetime.now() + dt.timedelta(minutes=ADIANTAR):
        return jsonify(ok=False, erro=f"escolha um horário pelo menos {ADIANTAR} min no futuro (regra do TikTok)"), 400
    plataformas = []
    for p in corpo.get("plataformas") or ["tiktok"]:
        p = str(p)
        if p == "youtube":
            plataformas.append(p)
        elif p == "tiktok" or p.startswith("tiktok:"):
            conta = achar_conta(p.split(":", 1)[1] if ":" in p else "")
            if not conta:
                return jsonify(ok=False, erro=f"conta do TikTok não configurada: {p.split(':', 1)[-1]}"), 400
            plataformas.append("tiktok:" + conta)
    if not plataformas:
        return jsonify(ok=False, erro="escolha TikTok e/ou YouTube"), 400
    total = 0
    for plataforma in plataformas:
        total += enfileirar(itens, plataforma, inicio)
    return jsonify(ok=True, adicionados=total)


def enfileirar(itens, plataforma, inicio):
    """Põe os cortes na fila de uma plataforma; o 1º sai em `inicio` e cada próximo INTERVALO min depois."""
    import datetime as dt
    n = 0
    for it in itens:
        clip_id = str(it.get("id") or "")
        caminho, nome, leg = str(it.get("arquivo") or ""), str(it.get("nome") or ""), (it.get("legenda") or "").strip()
        if not clip_id or not CAMINHO_OK.match(caminho):
            continue
        conta = plataforma.split(":", 1)[1] if plataforma.startswith("tiktok:") else None
        chave = ("yt:" + clip_id) if plataforma == "youtube" else chave_tiktok(conta, clip_id)
        with lock:
            if chave in estado["pendentes"] or chave == estado["atual"]:
                continue
            estado["pendentes"].append(chave)
        para = (inicio + dt.timedelta(minutes=INTERVALO * n)).strftime("%Y-%m-%dT%H:%M") if inicio else None
        marcar(chave, status="na fila", legenda=leg, erro=None, nome=nome, pedido=para)
        estado["cancelar"] = False
        fila.put({"chave": chave, "id": clip_id, "plataforma": "youtube" if plataforma == "youtube" else "tiktok",
                  "conta": conta, "arquivo": caminho, "nome": nome,
                  "legenda": leg, "gancho": str(it.get("gancho") or ""), "para": para,
                  "geracao": estado.get("geracao", 0)})
        n += 1
    return n


@app.get("/avatar")
def avatar():
    chave = request.args.get("chave") or ""
    if not re.fullmatch(r"yt|tt-[a-z0-9._]{2,40}", chave):
        return jsonify(ok=False), 404
    arq = AVATARES / f"{chave}.jpg"
    if not arq.is_file():
        return jsonify(ok=False), 404
    return send_file(arq, mimetype="image/jpeg", max_age=60)


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
    conta = achar_conta((request.get_json(silent=True) or {}).get("conta"))
    if not conta:
        return jsonify(ok=False, erro="conta do TikTok não configurada"), 400
    fila.put({"acao": "login", "conta": conta})
    return jsonify(ok=True)


@app.post("/login_youtube")
def login_youtube():
    fila.put({"acao": "login_youtube"})
    return jsonify(ok=True)


@app.post("/checar_youtube")
def checar_youtube():
    fila.put({"acao": "checar_youtube"})
    return jsonify(ok=True)


@app.post("/checar")
def checar():
    conta = achar_conta((request.get_json(silent=True) or {}).get("conta"))
    if not conta:
        return jsonify(ok=False, erro="conta do TikTok não configurada"), 400
    fila.put({"acao": "checar", "conta": conta})
    return jsonify(ok=True)


def limpar_interrompidos():
    """A fila vive só na memória: o que ficou 'na fila'/'enviando' num reinício não foi
    agendado. Marca como erro para o painel deixar mandar de novo."""
    reg = ler_registro()
    n = 0
    for v in reg.values():
        if v.get("status") in ("na fila", "enviando"):
            v.update(status="erro", erro="interrompido (o postador reiniciou); não foi agendado, mande de novo")
            n += 1
    if n:
        REGISTRO.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"{n} envio(s) interrompidos pelo reinício marcados para reenviar")


if __name__ == "__main__":
    limpar_interrompidos()
    threading.Thread(target=trabalhador, daemon=True).start()
    if "--login" in sys.argv:
        fila.put({"acao": "login"})
    print(f"Postador do TikTok (SupoClip) em http://127.0.0.1:{PORTA} — contas {', '.join(contas_tiktok()) or '(qualquer)'}"
          " — deixe esta janela aberta", flush=True)
    app.run(host="127.0.0.1", port=PORTA, threaded=True)
