$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Join-Path $PSScriptRoot "..")
$env:PYTHONUTF8 = "1"

if (-not (Test-Path -LiteralPath ".env" -PathType Leaf)) {
    Write-Error "Missing .env. Run scripts/configure.ps1, then set CHAT_MODEL and CHAT_API_KEY."
    exit 1
}

uv sync --locked
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& (Join-Path $PSScriptRoot 'mcp.ps1') -Action Start

uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000
exit $LASTEXITCODE
