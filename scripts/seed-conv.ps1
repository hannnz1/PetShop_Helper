$ErrorActionPreference = 'Stop'

Push-Location (Join-Path $PSScriptRoot '..')
try {
    docker compose exec -T mysql sh -c 'mysql -uroot -proot --default-character-set=utf8mb4 mewhelp < /docker-entrypoint-initdb.d/ch03-seed.sql'
    if ($LASTEXITCODE -ne 0) {
        throw "Ch03 conversation seed failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
