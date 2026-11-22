$ErrorActionPreference = 'Stop'
$tree = Get-Content $env:TEMP\lieder-tree.json -Raw -Encoding utf8 | ConvertFrom-Json
$scores = @($tree.tree | Where-Object { $_.path -match '\.mscx$' } | Select-Object -ExpandProperty path)
New-Item -ItemType Directory -Force data-gen/score-corpus/big500 | Out-Null
if (-not (Test-Path data-gen/score-corpus/big500/index.csv)) {
  'local,source' | Set-Content data-gen/score-corpus/big500/index.csv
}
$done = @{}
Import-Csv data-gen/score-corpus/big500/index.csv | ForEach-Object { $done[$_.local] = $true }
$count = 0
for ($i = 0; $i -lt 500; $i++) {
  $p = $scores[[int]($i * $scores.Count / 500)]
  $local = 'ds{0:d4}.mscx' -f ($i + 1)
  if ($done.ContainsKey($local)) { $count++; continue }
  $esc = ($p -split '/' | ForEach-Object { [uri]::EscapeDataString($_) }) -join '/'
  curl.exe -sL --max-time 120 -o "data-gen/score-corpus/big500/$local" ('https://raw.githubusercontent.com/OpenScore/Lieder/main/' + $esc)
  $f = Get-Item "data-gen/score-corpus/big500/$local" -ErrorAction SilentlyContinue
  if ($f -and $f.Length -gt 1000) {
    "$local,$p" | Add-Content data-gen/score-corpus/big500/index.csv
    $count++
  } else {
    Remove-Item "data-gen/score-corpus/big500/$local" -ErrorAction SilentlyContinue
    Write-Output "RETRY-LATER $local $p"
  }
  if ($count % 50 -eq 0) { Write-Output "progress: $count/500" }
}
Write-Output "done: $count/500"
