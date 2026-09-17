$ErrorActionPreference = 'Continue'

$sources = @(
    @{ Key = 'TV2CRANX'; Url = 'https://ntrs.nasa.gov/api/citations/19680023501/downloads/19680023501.pdf' },
    @{ Key = 'WX77UXWE'; Url = 'https://journals.lww.com/ccmjournal/_layouts/15/oaks.journals/downloadpdf.aspx?an=00003246-198610000-00017' },
    @{ Key = 'N5ZN33ZD'; Url = 'https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1469-8986.1990.tb02171.x' },
    @{ Key = 'CANN65Q2'; Url = 'https://www.sciencedirect.com/science/article/pii/S0033062005800350/pdfft?isDTMRedir=true&download=true' },
    @{ Key = '9W4HCNS3'; Url = 'https://www.sciencedirect.com/science/article/pii/S0147956305800366/pdf' },
    @{ Key = 'BDQQ667F'; Url = 'https://asma.kglmeridian.com/downloadpdf/view/journals/asem/70/8/article-p780.pdf' },
    @{ Key = 'BX485JVA'; Url = 'https://link.springer.com/content/pdf/10.1023/A:1009982611386.pdf' },
    @{ Key = 'TU2SHRHJ'; Url = 'https://link.springer.com/content/pdf/10.1007/BF02344724.pdf' },
    @{ Key = 'CAGGEQ25'; Url = 'https://citeseerx.ist.psu.edu/document?doi=f4b6293d73d2775cc7184e87f49b3c541be2fc78&repid=rep1&type=pdf' },
    @{ Key = '2EAQ37CN'; Url = 'https://files.core.ac.uk/download/pdf/268438154.pdf' },
    @{ Key = '8JWJRAVJ'; Url = 'https://link.springer.com/content/pdf/10.1007/s10877-019-00330-y.pdf' },
    @{ Key = 'VK3PCFJV'; Url = 'https://journals.physiology.org/doi/pdf/10.1152/ajpheart.1979.237.4.H491' },
    @{ Key = 'FQM7ZRAS'; Url = 'https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1475-097X.1986.tb00622.x' },
    @{ Key = 'U3TUKU8E'; Url = 'https://europepmc.org/articles/PMC1216462?pdf=render' },
    @{ Key = '8JKRBJB7'; Url = 'https://www.sciencedirect.com/science/article/pii/000291499291235V/pdfft?isDTMRedir=true&download=true' },
    @{ Key = '2QCNIHMN'; Url = 'https://europepmc.org/articles/PMC1381447?pdf=render' },
    @{ Key = 'RB87XMRW'; Url = 'https://europepmc.org/articles/PMC1728690?pdf=render' },
    @{ Key = 'Z7ZBPV7W'; Url = 'https://link.springer.com/content/pdf/10.1007/s004210000226.pdf' },
    @{ Key = 'CAGGEQ25'; Url = 'https://journal.chestnet.org/article/S0012-3692%2816%2934828-0/pdf' },
    @{ Key = 'ISW2MKUD'; Url = 'https://link.springer.com/content/pdf/10.1007/s10557-005-1048-0.pdf' },
    @{ Key = 'HX3WCFV7'; Url = 'https://academic.oup.com/bja/article-pdf/95/5/603/964801/aei224.pdf' },
    @{ Key = 'G48PVZ2K'; Url = 'https://link.springer.com/content/pdf/10.1007/s00134-007-0828-3.pdf' },
    @{ Key = 'Q3TSQIRI'; Url = 'https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1399-6576.2007.01445.x' },
    @{ Key = '2EW8XB4B'; Url = 'https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1475-097X.2010.00977.x' },
    @{ Key = 'WBJWHA2M'; Url = 'https://link.springer.com/content/pdf/10.1007/s00392-011-0329-9.pdf' },
    @{ Key = 'JIW45UH5'; Url = 'https://www.scielo.br/j/abc/a/LGLbpXh9YT4mXwfN387sgLc/?format=pdf&lang=en' },
    @{ Key = 'QHKPBERQ'; Url = 'https://europepmc.org/articles/PMC5654291?pdf=render' },
    @{ Key = 'KINHB88N'; Url = 'https://europepmc.org/articles/PMC6169070?pdf=render' },
    @{ Key = 'ME49H3IJ'; Url = 'https://europepmc.org/articles/PMC7447428?pdf=render' },
    @{ Key = 'GGUPHZI9'; Url = 'https://link.springer.com/content/pdf/10.1007/s10877-024-01246-y.pdf' },
    @{ Key = 'EDX22DP4'; Url = 'https://academic.oup.com/bja/article-pdf/100/1/88/18252077/aem320.pdf' },
    @{ Key = 'PF2NDTSA'; Url = 'https://academic.oup.com/bja/article-pdf/100/4/517/693082/aen024.pdf' },
    @{ Key = 'KISTS8R8'; Url = 'https://fn.bmj.com/content/97/5/F340.full.pdf' },
    @{ Key = 'HQVE683T'; Url = 'https://osypka-asia.com/pdf/pediatric/electrical.pdf' },
    @{ Key = 'SS84XB5Z'; Url = 'https://europepmc.org/articles/PMC4261789?pdf=render' },
    @{ Key = '4NACTMC5'; Url = 'https://www.nature.com/articles/jp201665.pdf' },
    @{ Key = 'WABWSWPT'; Url = 'https://osypka-asia.com/pdf/pediatric/Electrical%20Cardiometry%20in%20children.pdf' },
    @{ Key = '9NHAZ7IC'; Url = 'https://iopscience.iop.org/article/10.1088/1361-6579/aac02b/pdf' },
    @{ Key = 'MCVHV3JZ'; Url = 'https://journals.lww.com/ccmjournal/_layouts/15/oaks.journals/downloadpdf.aspx?an=00003246-199002000-00019' },
    @{ Key = 'SXQNFSU4'; Url = 'https://journals.lww.com/ccmjournal/_layouts/15/oaks.journals/downloadpdf.aspx?an=00003246-199310000-00021' },
    @{ Key = 'BC68E8GK'; Url = 'https://link.springer.com/content/pdf/10.1007/BF02025304.pdf' },
    @{ Key = 'PEZ8IDN2'; Url = 'https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1527-5299.2004.03407.x' },
    @{ Key = 'NFFHGJVW'; Url = 'https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1751-7133.2008.00001.x' },
    @{ Key = '4A4BBND3'; Url = 'https://europepmc.org/articles/PMC3618318?pdf=render' },
    @{ Key = '4VZBDFRD'; Url = 'https://academic.oup.com/eurheartj/article-pdf/14/2/150/1584190/14-2-150.pdf' },
    @{ Key = 'W6SE58XU'; Url = 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7615984/pdf/EMS196063.pdf' },
    @{ Key = 'KRUICEIX'; Url = 'https://academic.oup.com/cardiovascres/article-pdf/118/17/3272/49127799/cvac013.pdf' },
    @{ Key = 'IAEHUPF7'; Url = 'https://orbi.uliege.be/bitstream/2268/290864/1/ehab368.pdf' },
    @{ Key = 'ZNJC9Q47'; Url = 'https://pure.eur.nl/files/215348682/ESICM_guidelines_on_circulatory_shock_and_hemodynamic_monitoring_2025.pdf' },
    @{ Key = 'QTSV7XNJ'; Url = 'https://link.springer.com/content/pdf/10.1007/s00134-014-3525-z.pdf' },
    @{ Key = 'RGRIR2DG'; Url = 'https://pure.rug.nl/ws/portalfiles/portal/179270684/Perioperative_Hemodynamic_Monitoring.pdf' }
)

$destination = $PSScriptRoot
$results = foreach ($source in $sources) {
    $path = Join-Path $destination ($source.Key + '.pdf')
    if (Test-Path -LiteralPath $path) {
        $bytes = [System.IO.File]::ReadAllBytes($path)
        if ($bytes.Length -ge 5 -and [Text.Encoding]::ASCII.GetString($bytes, 0, 5) -eq '%PDF-') {
            [pscustomobject]@{ Key = $source.Key; Status = 'already-valid'; Bytes = $bytes.Length; Url = $source.Url }
            continue
        }
    }

    & curl.exe -L --fail --silent --show-error --connect-timeout 20 --max-time 90 --retry 1 -A 'Mozilla/5.0' -o $path $source.Url
    $exitCode = $LASTEXITCODE
    $valid = $false
    $size = 0
    if (Test-Path -LiteralPath $path) {
        $bytes = [System.IO.File]::ReadAllBytes($path)
        $size = $bytes.Length
        $valid = $size -ge 5 -and [Text.Encoding]::ASCII.GetString($bytes, 0, 5) -eq '%PDF-'
    }
    if (-not $valid -and (Test-Path -LiteralPath $path)) {
        Remove-Item -LiteralPath $path -Force
    }
    [pscustomobject]@{ Key = $source.Key; Status = if ($valid) { 'downloaded' } else { "failed-$exitCode" }; Bytes = $size; Url = $source.Url }
}

$results | ConvertTo-Json -Depth 3
