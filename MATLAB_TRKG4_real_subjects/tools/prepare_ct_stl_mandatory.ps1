param(
    [string]$Blender = 'E:\hard_Programms\blender\4_2_1\blender.exe',
    [string]$Subject = 'nik'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$stl = Join-Path $root "data\$Subject\stl"
$output = Join-Path $root 'output'
$source = Join-Path $stl 'lungs_ct_full_clean.stl'
$candidate = Join-Path $stl 'lungs_solid_v3_volume_fill.stl'
$bodySource = Join-Path $stl 'body.stl'
$bodyCandidate = Join-Path $stl 'body_solid_v3_1mm_r10_volume_fill.stl'
$heart = Join-Path $stl 'heart.stl'
$maskValidation = Join-Path $output 'lungs_volume_mask_validation.stl'

if (-not (Test-Path -LiteralPath $source)) {
    throw "Missing full clean CT lung surface: $source"
}

python (Join-Path $PSScriptRoot 'solid_fill_stl.py') `
    $bodySource $bodyCandidate --spacing-mm 1 --closing-radius-mm 10 `
    --closing-method edt --components 1 --target-faces 120000
if ($LASTEXITCODE -ne 0) { throw 'Body volumetric fill failed.' }

python (Join-Path $PSScriptRoot 'solid_fill_stl.py') `
    $source $maskValidation --spacing-mm 1 --closing-radius-mm 0 `
    --components 1 --target-faces 120000
if ($LASTEXITCODE -ne 0) { throw 'Volumetric fill validation failed.' }

python (Join-Path $PSScriptRoot 'decimate_stl.py') `
    $source $candidate --target-faces 120000
if ($LASTEXITCODE -ne 0) { throw 'Topology-preserving lung decimation failed.' }

& $Blender --background --factory-startup `
    --python (Join-Path $PSScriptRoot 'blender_analyze_stl.py') -- $candidate
if ($LASTEXITCODE -ne 0) { throw 'Lung topology validation failed.' }

& $Blender --background --factory-startup `
    --python (Join-Path $PSScriptRoot 'blender_intersect_stl.py') -- `
    $candidate $heart (Join-Path $output 'intersection_lungs_v3_heart.stl')
if ($LASTEXITCODE -ne 0) { throw 'Lung/heart intersection check failed.' }

& $Blender --background --factory-startup `
    --python (Join-Path $PSScriptRoot 'blender_render_stl.py') -- `
    $candidate $heart (Join-Path $output 'lungs_solid_v3_volume_fill')
if ($LASTEXITCODE -ne 0) { throw 'Lung control rendering failed.' }
