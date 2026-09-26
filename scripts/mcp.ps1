param(
    [Parameter(Mandatory=$true)][ValidateSet('Start', 'Stop')][string]$Action,
    [int]$LogisticsPort = 8101,
    [int]$AftersalesPort = 8102
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$stateDir = Join-Path $repo 'data\ch08'
$logDir = Join-Path $repo 'log'
$pythonCandidates = @(
    (Join-Path $repo '.venv\Scripts\python.exe'),
    (Join-Path $repo '..\..\.venv\Scripts\python.exe')
)
$pythonPath = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $pythonPath) {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
}
$servers = @(
    @{ Name='logistics'; Port=$LogisticsPort; Script='mcp_servers/logistics_server.py' },
    @{ Name='aftersales'; Port=$AftersalesPort; Script='mcp_servers/aftersales_server.py' }
)

foreach ($server in $servers) {
    $pidFile = Join-Path $stateDir ("mcp-{0}-{1}.pid" -f $server.Name, $server.Port)
    if ($Action -eq 'Stop') {
        if (-not (Test-Path -LiteralPath $pidFile -PathType Leaf)) { continue }
        $savedPid = [int](Get-Content -Raw -LiteralPath $pidFile)
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$savedPid" -ErrorAction SilentlyContinue
        if ($processInfo -and $processInfo.CommandLine -like "*$($server.Script)*") {
            Stop-Process -Id $savedPid -ErrorAction Stop
        }
        Remove-Item -LiteralPath $pidFile -ErrorAction Stop
        continue
    }

    New-Item -ItemType Directory -Force -Path $stateDir, $logDir | Out-Null
    if (Test-Path -LiteralPath $pidFile -PathType Leaf) {
        $savedPid = [int](Get-Content -Raw -LiteralPath $pidFile)
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId=$savedPid" -ErrorAction SilentlyContinue
        if ($processInfo -and $processInfo.CommandLine -like "*$($server.Script)*") { continue }
        Remove-Item -LiteralPath $pidFile -ErrorAction Stop
    }
    $priorPort = $env:PORT
    try {
        $env:PORT = [string]$server.Port
        $stdout = Join-Path $logDir ("mcp-{0}-{1}.log" -f $server.Name, $server.Port)
        $stderr = Join-Path $logDir ("mcp-{0}-{1}.err.log" -f $server.Name, $server.Port)
        $processInfo = Start-Process -FilePath $pythonPath -ArgumentList @('-X', 'utf8', $server.Script) `
            -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        Set-Content -Encoding ASCII -LiteralPath $pidFile -Value $processInfo.Id
    } finally {
        $env:PORT = $priorPort
    }
}

if ($Action -eq 'Start') {
    foreach ($server in $servers) {
        $ready = $false
        for ($attempt = 0; $attempt -lt 100; $attempt++) {
            $client = New-Object System.Net.Sockets.TcpClient
            try {
                $client.Connect('127.0.0.1', $server.Port)
                $ready = $true
                break
            } catch {
                Start-Sleep -Milliseconds 100
            } finally {
                $client.Dispose()
            }
        }
        if (-not $ready) { throw "MCP $($server.Name) failed to listen on port $($server.Port)" }
    }
}
