param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Start', 'Stop', 'Status')]
    [string]$Action
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$composeFile = Join-Path $repoRoot 'infra/langfuse/compose.yaml'
$envFile = Join-Path $repoRoot 'infra/langfuse/.env.langfuse'

if (-not (Test-Path -LiteralPath $envFile)) {
    throw "Create $envFile from infra/langfuse/.env.example and replace every CHANGE_ME value."
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker CLI is unavailable.'
}

$dockerArgs = @('compose', '--env-file', $envFile, '-f', $composeFile)

if ($Action -eq 'Start') {
    $contents = Get-Content -LiteralPath $envFile -Raw
    if ($contents -match 'CHANGE_ME|mysalt|mysecret|myredissecret|miniosecret') {
        throw 'Replace all example/default secrets before starting Langfuse.'
    }
    foreach ($required in @('POSTGRES_PASSWORD', 'SALT', 'ENCRYPTION_KEY', 'NEXTAUTH_SECRET', 'CLICKHOUSE_PASSWORD', 'MINIO_ROOT_PASSWORD', 'REDIS_AUTH')) {
        if ($contents -notmatch "(?m)^$required=\S+") {
            throw "Missing $required in .env.langfuse."
        }
    }
    if ($contents -notmatch '(?m)^ENCRYPTION_KEY=[0-9a-fA-F]{64}\s*$') {
        throw 'ENCRYPTION_KEY must be 64 hexadecimal characters.'
    }

    $memoryBytes = & docker info --format '{{.MemTotal}}' 2>$null
    if ($LASTEXITCODE -ne 0 -or -not ($memoryBytes -match '^\d+$')) {
        throw 'Docker daemon is unavailable.'
    }
    if ([long]$memoryBytes -lt 18GB) {
        throw 'Docker has less than 18 GiB assigned; increase the memory allocation before starting Langfuse.'
    }
    $cpuCount = & docker info --format '{{.NCPU}}' 2>$null
    if ($LASTEXITCODE -ne 0 -or -not ($cpuCount -match '^\d+$') -or [int]$cpuCount -lt 4) {
        throw 'Docker needs at least four CPUs assigned before starting Langfuse.'
    }

    foreach ($port in @(3000, 3030, 5432, 6379, 8123, 9000, 9090, 9091)) {
        if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
            throw "Port $port is in use. Resolve the conflict without stopping unrelated services."
        }
    }
    & docker @dockerArgs config --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose configuration is invalid.' }
    & docker @dockerArgs up -d
} elseif ($Action -eq 'Stop') {
    & docker @dockerArgs down
} else {
    & docker @dockerArgs ps
}
if ($LASTEXITCODE -ne 0) { throw "Langfuse $Action failed." }
