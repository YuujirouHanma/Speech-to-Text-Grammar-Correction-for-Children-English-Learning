<#
.SYNOPSIS
  Public SpeakFix demo from this laptop: local server + Cloudflare Quick Tunnel.

.DESCRIPTION
  Starts SpeakFix on 127.0.0.1 (not reachable from the Wi-Fi/LAN) in demo mode, then
  opens a temporary https://<random>.trycloudflare.com link to it. Share that link: it
  works from any device, and HTTPS lets browsers use the microphone.
  Stop with Ctrl+C (server and tunnel both stop). The link changes on every run.

  On first run cloudflared.exe is downloaded from Cloudflare's GitHub releases into
  .tools/ and its Authenticode signature is checked.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run_public_demo.ps1
#>
param(
  [int]$Port = 8000,
  [int]$Workers = 2,
  # A whole class on one Wi-Fi shares one public IP, and the limit counts per IP.
  [int]$RateLimitPerMinute = 600,
  # Everyone may press the button at once: 30 recordings wait about 6 s at most instead of being refused.
  [int]$MaxPendingRequests = 32
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$tools = Join-Path $root ".tools"
$cloudflared = Join-Path $tools "cloudflared.exe"

if (-not (Test-Path $python)) {
  throw "No virtual environment at $python. Follow 'Menjalankan secara lokal' in README.md first."
}
# The tunnel must expose this demo only, never another server left running on the port.
$probe = New-Object System.Net.Sockets.TcpClient
try {
  $probe.Connect("127.0.0.1", $Port)
  $portInUse = $true
} catch {
  $portInUse = $false
} finally {
  $probe.Close()
}
if ($portInUse) {
  throw "Port $Port is already in use. Stop that program first, or pass -Port <other port>."
}

if (-not (Test-Path $cloudflared)) {
  New-Item -ItemType Directory -Force $tools | Out-Null
  $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
  Write-Host "Downloading cloudflared from $url"
  Invoke-WebRequest -Uri $url -OutFile $cloudflared -UseBasicParsing
}
$signature = Get-AuthenticodeSignature $cloudflared
if ($signature.Status -ne "Valid" -or $signature.SignerCertificate.Subject -notmatch "Cloudflare") {
  Remove-Item $cloudflared
  throw "cloudflared.exe is not validly signed by Cloudflare ($($signature.Status)); deleted it."
}

# The models commit about 3 GB on Windows, and a full C: drive stops the page file from growing.
$freeCommitGb = (Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory / 1MB
if ($freeCommitGb -lt 6) {
  Write-Warning ("Only {0:N1} GB of commit memory is free. Close other programs and free up space on C:, or requests may fail with 'server penuh'." -f $freeCommitGb)
}

$env:HF_HUB_OFFLINE = "1"
$env:SPEAKFIX_HOST = "127.0.0.1"        # only the tunnel can reach the server
$env:SPEAKFIX_PORT = "$Port"
$env:SPEAKFIX_DEMO_MODE = "true"        # UI asks for adult voices only
$env:SPEAKFIX_INFERENCE_WORKERS = "$Workers"
$env:SPEAKFIX_RATE_LIMIT_PER_MINUTE = "$RateLimitPerMinute"
$env:SPEAKFIX_MAX_PENDING_REQUESTS = "$MaxPendingRequests"
$env:FORWARDED_ALLOW_IPS = "127.0.0.1"  # trust the tunnel's X-Forwarded-For (per-visitor rate limit)

$server = Start-Process -FilePath $python -ArgumentList "-m", "speakfix" -WorkingDirectory $root `
  -PassThru -NoNewWindow
try {
  Write-Host "Loading models..."
  $ready = $false
  for ($i = 0; $i -lt 180 -and -not $ready; $i++) {
    if ($server.HasExited) { throw "SpeakFix exited during startup; see the log above." }
    try {
      $ready = (Invoke-WebRequest "http://127.0.0.1:$Port/readyz" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200
    } catch {
      Start-Sleep -Seconds 1
    }
  }
  if (-not $ready) { throw "SpeakFix did not become ready; see the log above." }
  Write-Host "SpeakFix is ready. The public link is the https://...trycloudflare.com line below."
  & $cloudflared tunnel --no-autoupdate --url "http://127.0.0.1:$Port"
}
finally {
  Stop-Process -Id $server.Id -ErrorAction SilentlyContinue
  Write-Host "Demo stopped."
}
