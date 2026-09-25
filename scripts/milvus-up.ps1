$ErrorActionPreference = 'Stop'

$image = 'mewhelp/minio:RELEASE.2024-05-28T17-19-04Z'
$sourceUrl = 'https://codeload.github.com/minio/minio/tar.gz/refs/tags/RELEASE.2024-05-28T17-19-04Z'
$sourceHash = '987C59F79100A8F7E30362C3ACD4D8601FDDBFC93558E4B05C3B6AD78A8E3EF0'
$buildDir = Join-Path (Get-Location).Path 'work\minio-build'
$archive = Join-Path $buildDir 'minio-source.tar.gz'

$localImages = docker image ls --format '{{.Repository}}:{{.Tag}}'
if ($LASTEXITCODE -ne 0) { throw 'Unable to list local Docker images' }
if ($localImages -notcontains $image) {
    New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
    if (-not (Test-Path -LiteralPath $archive)) {
        Invoke-WebRequest -Uri $sourceUrl -OutFile $archive -UseBasicParsing -TimeoutSec 120
    }
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $sourceHash) {
        throw 'MinIO source archive SHA-256 mismatch'
    }
    docker build -t $image -f docker/minio/Dockerfile $buildDir
    if ($LASTEXITCODE -ne 0) { throw 'Unable to build pinned MinIO source image' }
}

docker compose up -d etcd minio milvus-standalone
if ($LASTEXITCODE -ne 0) { throw 'Unable to start Milvus Standalone services' }

for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $status = Invoke-WebRequest -Uri 'http://127.0.0.1:9091/healthz' -TimeoutSec 3 -UseBasicParsing
        if ($status.StatusCode -eq 200) {
            Write-Output 'Milvus Standalone healthy'
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 3
    }
}
throw 'Milvus Standalone did not become healthy within 180 seconds'
