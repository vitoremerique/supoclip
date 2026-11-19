' Starts the SupoClip TikTok poster with no window (used at Windows logon).
' Output goes to postador\postador.log. If it is already running, the new copy exits.
Set fso = CreateObject("Scripting.FileSystemObject")
pasta = fso.GetParentFolderName(WScript.ScriptFullName)
raiz = fso.GetParentFolderName(pasta)
cmd = "cmd /c set PYTHONIOENCODING=utf-8&& cd /d """ & raiz & """ && ""postador\.venv\Scripts\python.exe"" -u postador\servidor.py >> ""postador\postador.log"" 2>&1"
CreateObject("WScript.Shell").Run cmd, 0, False
