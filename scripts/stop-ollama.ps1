$ErrorActionPreference = 'Stop'
$pilotRoot = Split-Path -Parent $PSScriptRoot
$pilotRecordPath = Join-Path $pilotRoot '.pilot-cache\ollama-process.json'
if (-not (Test-Path -LiteralPath $pilotRecordPath)) { throw 'No pilot-owned process record exists.' }
$pilotRecord = Get-Content -LiteralPath $pilotRecordPath -Raw -Encoding utf8 | ConvertFrom-Json
$pilotProcess = Get-Process -Id $pilotRecord.pid -ErrorAction SilentlyContinue
if (-not $pilotProcess) { Write-Output 'The recorded pilot process is already stopped.'; exit 0 }
if ($pilotProcess.Path -ne $pilotRecord.executable -or
    $pilotProcess.StartTime.ToUniversalTime().ToString('o') -ne $pilotRecord.start_time_utc) {
    throw 'PID identity changed; refusing to stop an unverified process.'
}
# Unload through the owned server before terminating it. Killing only the
# Windows parent can leave inference workers holding GPU memory.
$pilotLoaded = Invoke-RestMethod -Uri "$($pilotRecord.endpoint)/api/ps" -TimeoutSec 10
foreach ($pilotModel in $pilotLoaded.models) {
    $pilotUnload = @{ model = $pilotModel.name; keep_alive = 0 } | ConvertTo-Json -Compress
    Invoke-RestMethod -Uri "$($pilotRecord.endpoint)/api/generate" -Method Post `
        -ContentType 'application/json' -Body $pilotUnload -TimeoutSec 60 | Out-Null
}
$pilotRemaining = Invoke-RestMethod -Uri "$($pilotRecord.endpoint)/api/ps" -TimeoutSec 10
if ($pilotRemaining.models.Count -gt 0) {
    throw 'Models remain loaded; refusing to leave GPU workers behind. Retry after active requests finish.'
}
$pilotProcess.Kill()
$pilotProcess.WaitForExit(5000) | Out-Null
Write-Output "Stopped pilot Ollama PID $($pilotProcess.Id)."
