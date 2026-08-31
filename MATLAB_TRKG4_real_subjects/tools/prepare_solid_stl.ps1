param(
    [string]$Blender = $env:BLENDER_EXE
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Blender)) {
    $blenderCommand = Get-Command blender -ErrorAction SilentlyContinue
    if ($null -eq $blenderCommand) {
        throw 'Blender not found. Set BLENDER_EXE or pass -Blender explicitly.'
    }
    $Blender = $blenderCommand.Source
}
if (-not (Test-Path -LiteralPath $Blender)) {
    throw "Blender executable not found: $Blender"
}

$root = Split-Path -Parent $PSScriptRoot
$stl = Join-Path $root 'data\nik\stl'
$output = Join-Path $root 'output'
$voxelScript = Join-Path $PSScriptRoot 'blender_voxel_remesh_stl.py'
$splitScript = Join-Path $PSScriptRoot 'blender_split_lungs_stl.py'
$analyzeScript = Join-Path $PSScriptRoot 'blender_analyze_stl.py'
$lungsV2 = Join-Path $stl 'lungs_solid_v2.stl'

& $Blender --background --factory-startup --python $voxelScript -- `
    (Join-Path $stl 'body.stl') (Join-Path $stl 'body_solid.stl') 6
if ($LASTEXITCODE -ne 0) { throw 'Body voxel remesh failed.' }

& $Blender --background --factory-startup --python $voxelScript -- `
    (Join-Path $stl 'lungs.stl') $lungsV2 4
if ($LASTEXITCODE -ne 0) { throw 'Lung voxel remesh failed.' }

& $Blender --background --factory-startup --python $splitScript -- `
    $lungsV2 (Join-Path $stl 'lungs_solid_v1.stl') -7.9 2 0
if ($LASTEXITCODE -ne 0) { throw 'Lung solid split failed.' }

& $Blender --background --factory-startup --python $analyzeScript -- `
    (Join-Path $stl 'body_solid.stl') `
    (Join-Path $stl 'lungs_solid_v1.stl') `
    (Join-Path $stl 'lungs_solid_v2.stl')
if ($LASTEXITCODE -ne 0) { throw 'STL validation failed.' }
