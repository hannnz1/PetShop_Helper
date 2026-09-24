$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Join-Path $PSScriptRoot "..")

if (Test-Path -LiteralPath ".env") {
    Write-Host ".env already exists; it was not changed. Edit CHAT_MODEL and CHAT_API_KEY before starting."
    exit 0
}

if (-not (Test-Path -LiteralPath ".env.example" -PathType Leaf)) {
    Write-Error "Missing .env.example."
    exit 1
}

Copy-Item -LiteralPath ".env.example" -Destination ".env"
Write-Host "Created .env from .env.example. Edit CHAT_MODEL and CHAT_API_KEY before starting."
