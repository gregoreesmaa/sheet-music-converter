$rows = Import-Csv data-gen/score-corpus/handwritten-500/PAIRS.csv
$miss = 0
foreach ($r in $rows) {
  if (-not (Test-Path "data-gen/score-corpus/handwritten-500/$($r.handwritten_path)")) { $miss++; Write-Output ("MISSING img: " + $r.handwritten_path) }
  if (-not (Test-Path "data-gen/score-corpus/handwritten-500/$($r.transcription_path)")) { $miss++; Write-Output ("MISSING tr: " + $r.transcription_path) }
}
Write-Output ("rows={0} missing={1}" -f $rows.Count, $miss)
Write-Output "---distinct transcriptions---"
$rows.transcription_path | Sort-Object -Unique
Write-Output "---png dimension sample---"
Add-Type -AssemblyName System.Drawing
foreach ($w in @("mozart-k464", "beethoven-op133-ms1", "schubert-d810", "beethoven-op131-sbb")) {
  $f = Get-ChildItem "data-gen/score-corpus/handwritten-500/handwritten/$w/*.png" | Select-Object -First 1
  $img = [System.Drawing.Image]::FromFile($f.FullName)
  Write-Output ("{0}/{1}: {2}x{3} {4}KB" -f $w, $f.Name, $img.Width, $img.Height, [math]::Round($f.Length / 1024, 0))
  $img.Dispose()
}
Write-Output "---mscx/xml + mscz/zip validity---"
foreach ($t in @("transcriptions/mozart-k387_sq7103818.mscx", "transcriptions/mozart-k464-m1_sq7093885.mscx")) {
  $h = (Get-Content "data-gen/score-corpus/handwritten-500/$t" -TotalCount 3) -join "`n"
  Write-Output ("{0}: xml_decl={1} musescore={2}" -f $t, ($h -match '<\?xml'), ($h -match '<museScore'))
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
foreach ($t in @("transcriptions/beethoven-op131_sq20090908.mscz", "transcriptions/beethoven-op135_sq12397213.mscz")) {
  $z = [System.IO.Compression.ZipFile]::OpenRead((Join-Path (Get-Location) "data-gen/score-corpus/handwritten-500/$t"))
  $names = @($z.Entries | ForEach-Object { $_.FullName })
  Write-Output ("{0}: entries={1} has_mscx={2}" -f $t, $names.Count, (($names -match '\.mscx$').Count -gt 0))
  $z.Dispose()
}
