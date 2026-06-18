$ErrorActionPreference = "Stop"

$skillRoot = "C:\Users\刘天翔\.codex\skills\codex-ppt"
$projectDir = "E:\GitHub\MLLMProject\Report\Final\ppt_handdraw_whiteboard"
$promptFile = Join-Path $projectDir "prompts\sample_slide_07_prompt.txt"
$outDir = Join-Path $projectDir "origin_image"
$outFile = Join-Path $outDir "slide_07.png"
$runtimePython = Join-Path $env:USERPROFILE ".codex-ppt-skill\.venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

if (-not (Test-Path -LiteralPath $runtimePython)) {
  python (Join-Path $skillRoot "scripts\codex_ppt_runtime.py") bootstrap
}

& $runtimePython (Join-Path $skillRoot "scripts\image_gen.py") generate `
  --prompt-file $promptFile `
  --size 2560x1440 `
  --quality medium `
  --out $outFile

Write-Host "Generated sample slide: $outFile"
