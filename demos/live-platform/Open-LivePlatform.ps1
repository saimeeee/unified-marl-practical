param(
    [ValidateRange(1, 65535)][int]$Port = 18765,
    [switch]$Rebuild,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$previousPort = $env:MARL_LOCAL_PORT
$compose = @('compose', '--project-name', 'unified-marl-live', '--project-directory', $PSScriptRoot,
             '--file', (Join-Path $PSScriptRoot 'compose.yaml'))
$url = "http://127.0.0.1:$Port/"
Push-Location $PSScriptRoot
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Install Docker Desktop and start its Linux engine, then run this launcher again. https://www.docker.com/products/docker-desktop/'
    }
    & docker compose version
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose v2 is required. Update Docker Desktop.' }
    $engine = & docker info --format '{{.OSType}}' 2>$null
    if ($LASTEXITCODE -ne 0 -or $engine -ne 'linux') {
        throw 'Start Docker Desktop in Linux-container mode, then retry.'
    }
    foreach ($scene in @('warehouse', 'mpe', 'traffic', 'powergrid')) {
        $modelPath = Join-Path $PSScriptRoot "bundle\models\$scene\policy.pt"
        if (-not (Test-Path -LiteralPath $modelPath -PathType Leaf)) {
            throw "The local demo bundle is missing: $modelPath. Use GitHub Download ZIP or the complete local-platform release, and retain the bundle directory."
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'bundle\vendor\CityFlow\setup.py') -PathType Leaf)) {
        throw 'The bundled CityFlow source is missing. Use the complete local-platform download.'
    }
    $env:MARL_LOCAL_PORT = "$Port"
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'local-state') | Out-Null
    # A missing image is expected on first launch. Avoid inspect's stderr,
    # which PowerShell 5.1 treats as terminating with ErrorActionPreference=Stop.
    $imageId = & docker image ls --quiet 'unified-marl-live:local-v1'
    if ($LASTEXITCODE -ne 0) { throw 'Could not list local Docker images.' }
    $hasImage = -not [string]::IsNullOrWhiteSpace(($imageId -join ''))
    if ($Rebuild -or -not $hasImage) {
        Write-Host 'Building the local CPU environment. The first build downloads dependencies and may take several minutes.'
        & docker @compose build platform
        if ($LASTEXITCODE -ne 0) { throw 'Local image build failed. Read the build error above; no remote server is used.' }
    }
    Write-Host 'Starting local simulation service...'
    & docker @compose up --detach --no-build platform
    if ($LASTEXITCODE -ne 0) {
        throw "Local startup failed. If port $Port is occupied, close the other program or run this script with -Port 18766."
    }
    $containerId = & docker @compose ps --quiet platform
    if ($LASTEXITCODE -ne 0 -or -not $containerId) { throw 'The local platform container did not start.' }
    $deadline = (Get-Date).AddMinutes(3)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        $running = & docker inspect --format '{{.State.Running}}' $containerId 2>$null
        if ($LASTEXITCODE -ne 0 -or $running -ne 'true') { break }
        try {
            $health = Invoke-RestMethod -Uri ($url + 'api/catalog') -TimeoutSec 3
            if ($health.mode -eq 'live_inference' -and $health.models.Count -ge 4) {
                $ready = $true
                break
            }
        } catch { }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) {
        & docker @compose logs --tail 60 platform
        throw 'The local service did not become ready. Check the container log above.'
    }
    Write-Host "Local platform is ready: $url"
    Write-Host 'Models, simulators, and run data are on this computer. No login to a remote machine is needed.'
    Write-Host 'Use the stop launcher to stop the local container. Closing the browser does not stop it.'
    if (-not $NoBrowser) { Start-Process $url }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
} finally {
    $env:MARL_LOCAL_PORT = $previousPort
    Pop-Location
}
