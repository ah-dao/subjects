param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$workspacePath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$archivePath = Join-Path $workspacePath 'archive/v2'
$manifestPath = Join-Path $archivePath 'snapshot_manifest.json'
$frozen = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if (-not $frozen.verified) { throw 'Unverified snapshot' }
if ((Get-FileHash -LiteralPath (Join-Path $archivePath 'pre_v3_snapshot.zip') -Algorithm SHA256).Hash.ToLower() -ne $frozen.archive_sha256) {
    throw 'Snapshot hash mismatch'
}
$fileHashes = @{}
foreach ($row in $frozen.files) { $fileHashes[$row.path] = $row.sha256 }
$journalPath = Join-Path $archivePath 'organization_journal.json'
if (Test-Path -LiteralPath $journalPath) { throw 'Organization already recorded; inspect journal before retry' }
$journal = [Collections.Generic.List[object]]::new()

function Guard-WorkspacePath([string]$value) {
    $absolute = [IO.Path]::GetFullPath($value)
    if (-not $absolute.StartsWith($workspacePath + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside intended workspace: $absolute"
    }
    return $absolute
}
function Record-Action($row) {
    $journal.Add($row)
    if ($Apply) { $journal | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $journalPath -Encoding UTF8 }
}
function Move-Archived([string]$source, [string]$destination) {
    $source = Guard-WorkspacePath $source
    $destination = Guard-WorkspacePath $destination
    if (-not (Test-Path -LiteralPath $source)) { return }
    if (Test-Path -LiteralPath $destination) { throw "Destination exists: $destination" }
    $isDirectory = (Get-Item -LiteralPath $source).PSIsContainer
    $digest = if (-not $isDirectory) { (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower() } else { $null }
    if ($Apply) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Move-Item -LiteralPath $source -Destination $destination
        if ($digest -and (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLower() -ne $digest) { throw 'Move hash mismatch' }
    }
    Record-Action ([ordered]@{action='move'; source=$source; destination=$destination; sha256=$digest})
}
function Remove-FrozenFile([string]$source, [string]$reason) {
    $source = Guard-WorkspacePath $source
    $relative = [IO.Path]::GetRelativePath($workspacePath, $source).Replace('\', '/')
    $digest = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLower()
    if (-not $fileHashes.ContainsKey($relative) -or $digest -ne $fileHashes[$relative]) {
        throw "Not identical to verified V2 snapshot: $relative"
    }
    $bytes = (Get-Item -LiteralPath $source).Length
    if ($Apply) { Remove-Item -LiteralPath $source }
    Record-Action ([ordered]@{action='delete_recoverable'; path=$relative; bytes=$bytes; sha256=$digest; reason=$reason})
}

# Source modules, canonical feature tables, raw inputs and final results remain
# at their old paths so pending V2 server returns can still pass provenance checks.
Move-Archived (Join-Path $workspacePath 'docs') (Join-Path $archivePath 'docs')
foreach ($name in @('ADMIN_TABPFN_4K_RUNBOOK.md', 'SERVER_RUNBOOK_TABPFN.md')) {
    Move-Archived (Join-Path $workspacePath $name) (Join-Path $archivePath "root_docs/$name")
}
foreach ($file in @(Get-ChildItem -LiteralPath $workspacePath -File -Filter '*.zip')) {
    Move-Archived $file.FullName (Join-Path $archivePath "returns/$($file.Name)")
}
foreach ($file in @(Get-ChildItem -LiteralPath (Join-Path $workspacePath 'dist') -File)) {
    Move-Archived $file.FullName (Join-Path $archivePath "packages/$($file.Name)")
}
foreach ($file in @(Get-ChildItem -LiteralPath (Join-Path $workspacePath 'features/v2/ablation') -File -Filter 'r10*.csv')) {
    Remove-FrozenFile $file.FullName 'Regenerable R10 null/ablation feature tables; source, results and verified snapshot retained'
}
foreach ($name in @('initial_20261020', 'initial_20261020_v2', 'initial_20261020_final')) {
    $targetPath = Guard-WorkspacePath (Join-Path $workspacePath "results/pixel_tabpfn_diagnostics_v1/$name")
    if (Test-Path -LiteralPath $targetPath) {
        foreach ($file in @(Get-ChildItem -LiteralPath $targetPath -Recurse -File)) {
            Remove-FrozenFile $file.FullName 'Superseded local diagnostics draft; initial_release and natural_breaks_20261020 retained'
        }
        if ($Apply) { Remove-Item -LiteralPath $targetPath -Recurse }
    }
}
foreach ($name in @('pixel_bag_tests', 'pixel_geo_ablation_tests', 'pixel_geo_repr_tests', 'pixel_stats_attribution_tests', 'pixel_tabpfn_diagnostics_tests', 'pixel_tabpfn_suite_tests', 'pixel_tabpfn_tests', 'representation_tests')) {
    $targetPath = Guard-WorkspacePath (Join-Path $workspacePath "results/$name")
    if ((Test-Path -LiteralPath $targetPath) -and -not (Get-ChildItem -LiteralPath $targetPath -Force | Select-Object -First 1)) {
        if ($Apply) { Remove-Item -LiteralPath $targetPath }
        Record-Action ([ordered]@{action='delete_empty_directory'; path=$targetPath})
    }
}
$removed = ($journal | Where-Object { $_['action'] -eq 'delete_recoverable' } | ForEach-Object { $_['bytes'] } | Measure-Object -Sum).Sum
Write-Output "Applied=$Apply; actions=$($journal.Count); deleted recoverable bytes=$removed"
if (-not $Apply) { $journal | ConvertTo-Json -Depth 8 }
