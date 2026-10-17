param(
    [string]$OllamaPath,
    [ValidateRange(1024, 65535)][int]$Port = 11435,
    [ValidateRange(4096, 65536)][int]$ContextSize = 16384
)

$ErrorActionPreference = 'Stop'
$pilotRoot = Split-Path -Parent $PSScriptRoot
$pilotCache = Join-Path $pilotRoot '.pilot-cache'
New-Item -ItemType Directory -Path $pilotCache -Force | Out-Null

if (-not $OllamaPath) {
    $pilotPortable = Join-Path $pilotRoot '.runtime\ollama\ollama.exe'
    if (Test-Path -LiteralPath $pilotPortable) {
        $OllamaPath = $pilotPortable
    } else {
        $pilotCommand = Get-Command ollama -ErrorAction SilentlyContinue
        if (-not $pilotCommand) { throw 'Install Ollama or pass -OllamaPath to a portable executable.' }
        $OllamaPath = $pilotCommand.Source
    }
}
$pilotExecutable = (Resolve-Path -LiteralPath $OllamaPath).Path
$pilotEndpoint = "http://127.0.0.1:$Port"
try {
    $pilotExisting = Invoke-RestMethod -Uri "$pilotEndpoint/api/version" -TimeoutSec 2
} catch { $pilotExisting = $null }
if ($pilotExisting) {
    Write-Output "An Ollama server already responds at $pilotEndpoint (version $($pilotExisting.version)); no process was started."
    exit 0
}

$pilotSettings = @{
    OLLAMA_HOST = "127.0.0.1:$Port"
    OLLAMA_MODELS = Join-Path $pilotCache 'ollama-models'
    OLLAMA_NO_CLOUD = 'true'
    OLLAMA_CONTEXT_LENGTH = [string]$ContextSize
    OLLAMA_NUM_PARALLEL = '1'
    OLLAMA_KEEP_ALIVE = '5m'
}
$pilotPreviousSettings = @{}
try {
    foreach ($pilotSetting in $pilotSettings.GetEnumerator()) {
        $pilotPreviousSettings[$pilotSetting.Key] = [Environment]::GetEnvironmentVariable($pilotSetting.Key, 'Process')
        [Environment]::SetEnvironmentVariable($pilotSetting.Key, $pilotSetting.Value, 'Process')
    }
    $pilotProcess = Start-Process -FilePath $pilotExecutable -ArgumentList 'serve' `
        -WindowStyle Hidden -PassThru -WorkingDirectory $pilotRoot `
        -RedirectStandardOutput (Join-Path $pilotCache 'ollama-stdout.log') `
        -RedirectStandardError (Join-Path $pilotCache 'ollama-stderr.log')
    $pilotRecord = @{
        pid = $pilotProcess.Id
        executable = $pilotExecutable
        start_time_utc = $pilotProcess.StartTime.ToUniversalTime().ToString('o')
        endpoint = $pilotEndpoint
        cloud_disabled = $true
    }
    $pilotRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $pilotCache 'ollama-process.json') -Encoding utf8
} finally {
    foreach ($pilotSetting in $pilotPreviousSettings.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($pilotSetting.Key, $pilotSetting.Value, 'Process')
    }
}

$pilotDeadline = (Get-Date).AddSeconds(30)
while ((Get-Date) -lt $pilotDeadline) {
    try {
        $pilotVersion = Invoke-RestMethod -Uri "$pilotEndpoint/api/version" -TimeoutSec 2
        Write-Output "Ollama $($pilotVersion.version) ready at $pilotEndpoint; PID $($pilotProcess.Id); cloud disabled."
        exit 0
    } catch {
        if ($pilotProcess.HasExited) { throw "Ollama exited; inspect $pilotCache\ollama-stderr.log." }
        Start-Sleep -Milliseconds 250
    }
}
throw "Ollama did not become ready within 30 seconds; inspect $pilotCache\ollama-stderr.log."
