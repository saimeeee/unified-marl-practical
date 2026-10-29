$ErrorActionPreference = 'Stop'
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Docker is not installed.'
    }
    & docker compose --project-name unified-marl-live --project-directory $PSScriptRoot --file (Join-Path $PSScriptRoot 'compose.yaml') down
    if ($LASTEXITCODE -ne 0) { throw 'Could not stop the local platform. Check Docker Desktop.' }
    Write-Host 'Local platform stopped. Model assets and local-state run data were retained.'
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
