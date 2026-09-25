$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
& $python -X utf8 (Join-Path $PSScriptRoot 'demo_agent.py') @args
exit $LASTEXITCODE
