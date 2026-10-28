param(
    [Parameter(Mandatory=$true)][string]$InputDocx,
    [Parameter(Mandatory=$true)][string]$OutputPdf,
    [switch]$UpdateAndSave
)
$ErrorActionPreference = 'Stop'
$sourceFile = (Resolve-Path -LiteralPath $InputDocx).Path
$pdfFile = [IO.Path]::GetFullPath($OutputPdf)
$pdfParent = [IO.Path]::GetDirectoryName($pdfFile)
if (-not (Test-Path -LiteralPath $pdfParent)) {
    New-Item -ItemType Directory -Path $pdfParent -Force | Out-Null
}
$wordApplication = $null
$wordDocument = $null
try {
    $wordApplication = New-Object -ComObject Word.Application
    $wordApplication.Visible = $false
    $wordApplication.DisplayAlerts = 0
    $wordApplication.Options.UpdateLinksAtOpen = $false
    $wordDocument = $wordApplication.Documents.Open($sourceFile, $false, (-not $UpdateAndSave.IsPresent))
    if ($UpdateAndSave) {
        $null = $wordDocument.Fields.Update()
        $wordDocument.Repaginate()
        foreach ($toc in $wordDocument.TablesOfContents) {
            $toc.Update()
            $toc.Range.Font.Name = 'Times New Roman'
            $toc.Range.Font.Size = 12
            $toc.Range.ParagraphFormat.LineSpacingRule = 0
            $toc.Range.ParagraphFormat.SpaceBefore = 0
            $toc.Range.ParagraphFormat.SpaceAfter = 3
            $toc.Range.ParagraphFormat.Alignment = 0
        }
        $wordDocument.Repaginate()
        foreach ($toc in $wordDocument.TablesOfContents) { $toc.UpdatePageNumbers() }
        $wordDocument.Save()
    }
    $pages = $wordDocument.ComputeStatistics(2)
    $wordDocument.ExportAsFixedFormat($pdfFile, 17)
    [PSCustomObject]@{pages=$pages; pdf=$pdfFile; updated_docx=$UpdateAndSave.IsPresent} | ConvertTo-Json
}
finally {
    if ($null -ne $wordDocument) { $wordDocument.Close(0) }
    if ($null -ne $wordApplication) { $wordApplication.Quit(0) }
    if ($null -ne $wordDocument) { [Runtime.InteropServices.Marshal]::ReleaseComObject($wordDocument) | Out-Null }
    if ($null -ne $wordApplication) { [Runtime.InteropServices.Marshal]::ReleaseComObject($wordApplication) | Out-Null }
}
