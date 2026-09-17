param(
    [Parameter(Mandatory = $true)][string]$ParentKey,
    [string]$PdfPath,
    [string]$SourceUrl = '',
    [switch]$CreateOnly
)

$ErrorActionPreference = 'Stop'
$api = 'http://127.0.0.1:23119'
$resolvedPdf = $null
$pdfBytes = $null
if (-not $CreateOnly) {
    if (-not $PdfPath) { throw 'PdfPath is required unless CreateOnly is set' }
    $resolvedPdf = (Resolve-Path -LiteralPath $PdfPath).Path
    $pdfBytes = [IO.File]::ReadAllBytes($resolvedPdf)
    if ($pdfBytes.Length -lt 5 -or [Text.Encoding]::ASCII.GetString($pdfBytes, 0, 5) -ne '%PDF-') {
        throw "Not a valid PDF: $resolvedPdf"
    }
}

$parentJson = & curl.exe -sS "$api/api/users/0/items/$ParentKey"
$parent = $parentJson | ConvertFrom-Json
if (-not $parent.data -or $parent.data.itemType -eq 'attachment') {
    throw "Parent item not found or invalid: $ParentKey"
}

$beforeJson = & curl.exe -sS "$api/api/users/0/items/top?sort=dateAdded&direction=desc&limit=100"
$beforeRaw = $beforeJson | ConvertFrom-Json
$beforeKeys = @()
foreach ($item in $beforeRaw) { $beforeKeys += $item.key }

$sessionId = [Guid]::NewGuid().ToString('N')
$connectorItemId = [Guid]::NewGuid().ToString('N')
$d = $parent.data
$newItem = [ordered]@{
    id = $connectorItemId
    itemType = $d.itemType
    title = $d.title
    creators = @($d.creators)
    abstractNote = $d.abstractNote
    date = $d.date
    language = $d.language
    DOI = $d.DOI
    url = $d.url
    publicationTitle = $d.publicationTitle
    volume = $d.volume
    issue = $d.issue
    pages = $d.pages
    ISSN = $d.ISSN
    extra = $d.extra
    tags = @($d.tags)
}
$payload = [ordered]@{
    sessionID = $sessionId
    items = @($newItem)
    uri = if ($d.url) { $d.url } else { 'https://www.zotero.org/' }
} | ConvertTo-Json -Depth 15 -Compress

$saveItems = Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$api/connector/saveItems" -Headers @{
    'Content-Type' = 'application/json'
    'Zotero-Allowed-Request' = 'true'
    'X-Zotero-Connector-API-Version' = '3'
} -Body ([Text.Encoding]::UTF8.GetBytes($payload))
if ($saveItems.StatusCode -ne 201) {
    throw "saveItems HTTP $($saveItems.StatusCode): $($saveItems.Content)"
}

$attachmentStatus = 0
if (-not $CreateOnly) {
    $metadata = [ordered]@{
        sessionID = $sessionId
        parentItemID = $connectorItemId
        title = [IO.Path]::GetFileName($resolvedPdf)
        url = $SourceUrl
    } | ConvertTo-Json -Compress

    $request = [Net.HttpWebRequest]::Create("$api/connector/saveAttachment")
    $request.Method = 'POST'
    $request.ContentType = 'application/pdf'
    $request.ContentLength = $pdfBytes.Length
    $request.Headers.Add('X-Metadata', $metadata)
    $request.Headers.Add('Zotero-Allowed-Request', 'true')
    $stream = $request.GetRequestStream()
    $stream.Write($pdfBytes, 0, $pdfBytes.Length)
    $stream.Close()
    $response = $request.GetResponse()
    $attachmentStatus = [int]$response.StatusCode
    $response.Close()
    if ($attachmentStatus -ne 201) {
        throw "saveAttachment HTTP $attachmentStatus"
    }
}

Start-Sleep -Milliseconds 800
$afterJson = & curl.exe -sS "$api/api/users/0/items/top?sort=dateAdded&direction=desc&limit=100"
$afterRaw = $afterJson | ConvertFrom-Json
$after = @()
foreach ($item in $afterRaw) { $after += $item }
$duplicate = $after | Where-Object { $_.key -notin $beforeKeys -and $_.data.title -eq $d.title } | Select-Object -First 1
if (-not $duplicate) {
    throw 'Created duplicate item could not be identified'
}

$childrenJson = & curl.exe -sS "$api/api/users/0/items/$($duplicate.key)/children"
$children = @($childrenJson | ConvertFrom-Json)
$pdfChildren = @($children | Where-Object { $_.data.contentType -eq 'application/pdf' })

[pscustomobject]@{
    KeeperKey = $ParentKey
    DuplicateKey = $duplicate.key
    AttachmentStatus = $attachmentStatus
    PdfChildren = $pdfChildren.Count
    PdfChildKeys = @($pdfChildren | ForEach-Object { $_.key })
} | ConvertTo-Json -Depth 4
