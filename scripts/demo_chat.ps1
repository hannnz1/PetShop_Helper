$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = '1'
& .\.venv\Scripts\python.exe scripts\demo_chat.py @args
exit $LASTEXITCODE
