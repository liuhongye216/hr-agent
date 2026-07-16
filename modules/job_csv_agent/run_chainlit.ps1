param(
    [int]$Port = 8001,
    [string]$ApiUrl = "http://127.0.0.1:8000"
)

$ErrorActionPreference = "Stop"
$env:JOB_AGENT_API_URL = $ApiUrl

Push-Location $PSScriptRoot
try {
    chainlit run job_csv_agent/chainlit_app.py -w --port $Port
}
finally {
    Pop-Location
}
