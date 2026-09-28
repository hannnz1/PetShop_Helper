param(
    [string]$EnvFile = '',
    [ValidateRange(1024,65535)][int]$Port = 8000,
    [switch]$SkipMcp
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $taskRoot
if (-not $EnvFile) { $EnvFile = Join-Path $taskRoot '.env' }
if (-not (Test-Path -LiteralPath $EnvFile -PathType Leaf)) {
    throw 'Configuration file missing. Pass -EnvFile with your existing .env path; keys are never printed.'
}
$taskConfig = (Resolve-Path -LiteralPath $EnvFile).Path
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'Python environment missing. Run uv sync --locked in the current worktree first.'
}
Write-Host "Starting latest code from $taskRoot on http://127.0.0.1:$Port/admin"
if (-not $SkipMcp) { & (Join-Path $PSScriptRoot 'mcp.ps1') -Action Start }
& $taskPython -X utf8 -m uvicorn app.main:app --env-file $taskConfig --host 127.0.0.1 --port $Port
exit $LASTEXITCODE
