param(
    [string]$ExcelPath = "Auswertung_20260407b.xlsx",
    [string]$OutputChart = "phase_match_comparison.png",
    [string]$OutputCsv = "phase_match_summary.csv"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

function Get-ColIndex {
    param([string]$CellRef)

    $letters = ($CellRef -replace "\d", "")
    $idx = 0
    foreach ($ch in $letters.ToCharArray()) {
        $idx = $idx * 26 + ([int][char]::ToUpper($ch) - [int][char]"A" + 1)
    }
    return $idx
}

function Read-XlsxRows {
    param([string]$Path)

    $fs = [System.IO.File]::Open(
        $Path,
        [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Read,
        [System.IO.FileShare]::ReadWrite
    )
    $zip = New-Object System.IO.Compression.ZipArchive(
        $fs,
        [System.IO.Compression.ZipArchiveMode]::Read,
        $false
    )

    try {
        $sharedStrings = @()
        $sharedEntry = $zip.Entries | Where-Object FullName -eq "xl/sharedStrings.xml"
        if ($sharedEntry) {
            $reader = New-Object System.IO.StreamReader($sharedEntry.Open())
            try {
                [xml]$sharedXml = $reader.ReadToEnd()
            }
            finally {
                $reader.Close()
            }

            foreach ($si in $sharedXml.sst.si) {
                if ($si.t) {
                    $sharedStrings += [string]$si.t
                }
                elseif ($si.r) {
                    $sharedStrings += (($si.r | ForEach-Object { $_.t }) -join "")
                }
                else {
                    $sharedStrings += ""
                }
            }
        }

        $sheetEntry = $zip.Entries | Where-Object FullName -eq "xl/worksheets/sheet1.xml"
        if (-not $sheetEntry) {
            throw "Could not find sheet1.xml in workbook."
        }

        $sheetReader = New-Object System.IO.StreamReader($sheetEntry.Open())
        try {
            [xml]$sheetXml = $sheetReader.ReadToEnd()
        }
        finally {
            $sheetReader.Close()
        }

        $rows = @{}
        foreach ($row in $sheetXml.worksheet.sheetData.row) {
            $rowIdx = [int]$row.r
            $cells = @{}

            foreach ($c in $row.c) {
                $col = Get-ColIndex -CellRef ([string]$c.r)
                $type = [string]$c.t
                $val = ""

                if ($type -eq "s") {
                    $sharedIndex = [int]$c.v
                    if ($sharedIndex -ge 0 -and $sharedIndex -lt $sharedStrings.Count) {
                        $val = $sharedStrings[$sharedIndex]
                    }
                }
                elseif ($type -eq "inlineStr") {
                    $val = [string]$c.is.t
                }
                elseif ($c.v -ne $null) {
                    $val = [string]$c.v
                }

                $cells[$col] = $val
            }

            $rows[$rowIdx] = $cells
        }

        return $rows
    }
    finally {
        $zip.Dispose()
        $fs.Dispose()
    }
}

function Normalize-Key {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ""
    }
    return (($Value.ToLowerInvariant()) -replace "[^a-z0-9]", "")
}

function Get-BaseNameFromSpectrum {
    param([string]$SpectrumName)

    if ($SpectrumName -match "^(.*?)(?:_\d|\s\d)") {
        return $matches[1]
    }
    return $SpectrumName
}

function Get-TruthKeys {
    param(
        [string]$SpectrumName,
        [hashtable]$AliasMap
    )

    $baseName = Get-BaseNameFromSpectrum -SpectrumName $SpectrumName
    $baseNorm = Normalize-Key -Value $baseName

    if ($AliasMap.ContainsKey($baseNorm)) {
        return $AliasMap[$baseNorm]
    }

    $tokens = ($baseName.ToLowerInvariant() -split "[^a-z0-9]+") |
        Where-Object { $_.Length -ge 3 } |
        ForEach-Object { Normalize-Key -Value $_ } |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        Select-Object -Unique

    if (-not ($tokens -contains $baseNorm) -and -not [string]::IsNullOrWhiteSpace($baseNorm)) {
        $tokens += $baseNorm
    }

    return @($tokens | Select-Object -Unique)
}

function Is-PerfectMatch {
    param(
        [string]$CandidateNorm,
        [string[]]$TruthKeys
    )

    foreach ($truth in $TruthKeys) {
        if ([string]::IsNullOrWhiteSpace($truth)) {
            continue
        }

        if ($CandidateNorm -eq $truth) {
            return $true
        }

        if ($truth.Length -ge 4 -and $CandidateNorm.Contains($truth)) {
            return $true
        }

        if ($CandidateNorm.Length -ge 4 -and $truth.Contains($CandidateNorm)) {
            return $true
        }
    }

    return $false
}

$excelFile = Join-Path (Get-Location) $ExcelPath
if (-not (Test-Path -LiteralPath $excelFile)) {
    throw "Workbook not found: $excelFile"
}

$rows = Read-XlsxRows -Path $excelFile

$appColumns = [ordered]@{
    "Raman Match"                = 2..11
    "RamanLab MultiWindow"       = 14..23
    "RamanLab MineralVibration"  = 26..35
    "RamanPhaseID"               = 38..47
}

$topThresholds = @(1, 3, 6, 10)

$truthAliases = @{
    "aquamarine"          = @("beryl")
    "beryltransparent"    = @("beryl")
    "calcite"             = @("calcite")
    "cordierite"          = @("cordierite")
    "corundumtransparent" = @("corundum")
    "cubiczirconia"       = @("zirconia")
    "diamond"             = @("diamond")
    "emerald"             = @("beryl")
    "moissanite"          = @("moissanite")
    "olivineperidote"     = @("forsterite", "olivine", "peridote")
    "pinkquartzglas"      = @("quartz", "quartzglass")
    "quartz"              = @("quartz")
    "saphire"             = @("corundum", "sapphire")
    "synspinel"           = @("spinel")
    "topaz"               = @("topaz")
    "zircon"              = @("zircon")
}

$stats = @{}
foreach ($app in $appColumns.Keys) {
    $stats[$app] = @{}
    foreach ($k in $topThresholds) {
        $stats[$app][$k] = 0
    }
}

$totalSpectra = 0
$rowKeys = $rows.Keys | Sort-Object
foreach ($rowIdx in $rowKeys) {
    if ($rowIdx -lt 3) {
        continue
    }

    $row = $rows[$rowIdx]
    $spectrumName = [string]$row[1]
    if ([string]::IsNullOrWhiteSpace($spectrumName)) {
        continue
    }

    $truthKeys = Get-TruthKeys -SpectrumName $spectrumName -AliasMap $truthAliases
    if ($truthKeys.Count -eq 0) {
        continue
    }

    $totalSpectra++

    foreach ($app in $appColumns.Keys) {
        $candidates = @()
        foreach ($col in $appColumns[$app]) {
            $raw = [string]$row[$col]
            if ([string]::IsNullOrWhiteSpace($raw)) {
                continue
            }
            if ($raw.Trim().ToLowerInvariant() -eq "x") {
                continue
            }

            $norm = Normalize-Key -Value $raw
            if (-not [string]::IsNullOrWhiteSpace($norm)) {
                $candidates += $norm
            }
        }

        foreach ($k in $topThresholds) {
            $limit = [Math]::Min($k, $candidates.Count)
            $hit = $false
            for ($i = 0; $i -lt $limit; $i++) {
                if (Is-PerfectMatch -CandidateNorm $candidates[$i] -TruthKeys $truthKeys) {
                    $hit = $true
                    break
                }
            }

            if ($hit) {
                $stats[$app][$k]++
            }
        }
    }
}

if ($totalSpectra -eq 0) {
    throw "No spectra rows found (expected data from row 3 onward)."
}

$summary = foreach ($app in $appColumns.Keys) {
    foreach ($k in $topThresholds) {
        $hits = [int]$stats[$app][$k]
        $pct = [Math]::Round(100.0 * $hits / $totalSpectra, 1)
        [pscustomobject]@{
            App     = $app
            TopN    = "Top$k"
            Hits    = $hits
            Total   = $totalSpectra
            Percent = $pct
        }
    }
}

$csvPath = Join-Path (Get-Location) $OutputCsv
$summary | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8

$chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$chart.Width = 1280
$chart.Height = 720

$chartArea = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea("Main")
$chartArea.AxisX.Title = "Ranking Threshold"
$chartArea.AxisY.Title = "Perfect Match Rate (%)"
$chartArea.AxisY.Minimum = 0
$chartArea.AxisY.Maximum = 100
$chartArea.AxisY.Interval = 10
$chartArea.AxisX.Minimum = 0.5
$chartArea.AxisX.Maximum = 4.5
$chartArea.AxisX.Interval = 1
$chartArea.AxisX.MajorGrid.Enabled = $false
$chartArea.AxisY.MajorGrid.LineColor = [System.Drawing.Color]::LightGray
$chartArea.AxisY.MajorGrid.LineDashStyle = [System.Windows.Forms.DataVisualization.Charting.ChartDashStyle]::Dash
$chart.ChartAreas.Add($chartArea)

$chartArea.AxisX.CustomLabels.Clear()
$chartArea.AxisX.CustomLabels.Add(0.5, 1.5, "Top1") | Out-Null
$chartArea.AxisX.CustomLabels.Add(1.5, 2.5, "Top3") | Out-Null
$chartArea.AxisX.CustomLabels.Add(2.5, 3.5, "Top6") | Out-Null
$chartArea.AxisX.CustomLabels.Add(3.5, 4.5, "Top10") | Out-Null

$legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend("Legend")
$legend.Docking = [System.Windows.Forms.DataVisualization.Charting.Docking]::Top
$chart.Legends.Add($legend)

$colorMap = @{
    "Raman Match"               = [System.Drawing.Color]::FromArgb(53, 123, 184)
    "RamanLab MultiWindow"      = [System.Drawing.Color]::FromArgb(230, 126, 34)
    "RamanLab MineralVibration" = [System.Drawing.Color]::FromArgb(214, 39, 40)
    "RamanPhaseID"              = [System.Drawing.Color]::FromArgb(46, 173, 102)
}

foreach ($app in $appColumns.Keys) {
    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series($app)
    $series.ChartType = [System.Windows.Forms.DataVisualization.Charting.SeriesChartType]::Column
    $series.ChartArea = "Main"
    $series.IsValueShownAsLabel = $true
    $series["PointWidth"] = "0.7"
    $series.Color = $colorMap[$app]
    $series.BorderColor = [System.Drawing.Color]::Black
    $series.BorderWidth = 1

    for ($j = 0; $j -lt $topThresholds.Count; $j++) {
        $k = $topThresholds[$j]
        $point = $summary | Where-Object { $_.App -eq $app -and $_.TopN -eq "Top$k" }
        $x = [double]($j + 1)
        $idx = $series.Points.AddXY($x, [double]$point.Percent)
        $series.Points[$idx].Label = ("{0}%" -f $point.Percent)
    }

    $chart.Series.Add($series)
}

$title = New-Object System.Windows.Forms.DataVisualization.Charting.Title
$title.Text = "Raman App Comparison: Perfect Phase Match Rate"
$title.Font = New-Object System.Drawing.Font("Segoe UI", 15, [System.Drawing.FontStyle]::Bold)
$chart.Titles.Add($title)

$chartPath = Join-Path (Get-Location) $OutputChart
$chart.SaveImage($chartPath, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)

Write-Output "Processed spectra: $totalSpectra"
Write-Output "Saved chart: $chartPath"
Write-Output "Saved summary CSV: $csvPath"
Write-Output ""
$summary | Format-Table -AutoSize
