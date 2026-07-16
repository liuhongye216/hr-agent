$ErrorActionPreference = "Stop"

$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if (-not (Test-Path -LiteralPath $edge)) {
    $edge = "C:\Program Files\Google\Chrome\Application\chrome.exe"
}
if (-not (Test-Path -LiteralPath $edge)) {
    throw "Microsoft Edge or Google Chrome was not found."
}

$profile = Join-Path $PSScriptRoot ".cdp_profile"
New-Item -ItemType Directory -Force -Path $profile | Out-Null

try {
    Invoke-RestMethod "http://127.0.0.1:9222/json/version" -TimeoutSec 2 | Out-Null
    Write-Host "Browser on port 9222 is already running."
    exit 0
} catch {
    # Port is not active. Start a dedicated visible browser session.
}

Start-Process -FilePath $edge -ArgumentList @(
    "--remote-debugging-port=9222",
    "--user-data-dir=`"$profile`"",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-sync",
    "https://www.zhipin.com/"
)

Write-Host "Browser started. Complete any visible security check before running collector.py."
