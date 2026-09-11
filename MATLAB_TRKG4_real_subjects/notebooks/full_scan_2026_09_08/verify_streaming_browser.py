"""Read-only browser QA of a committed local streaming report."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    report = directory / 'report/full_scan.html'
    errors, requests = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--enable-unsafe-swiftshader'])
        page = browser.new_page(viewport={'width': 1480, 'height': 1040})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: requests.append(request.url)
                if request.url.startswith(('http://', 'https://')) else None)
        page.goto(report.as_uri(), wait_until='load', timeout=120000)
        page.wait_for_function("!document.getElementById('angle').disabled", timeout=120000)
        layers = page.locator('#angle option').count()
        assert layers == 42
        counts = page.evaluate("""() => {const d=document.getElementById('body').data;
            return {valid:d[1].x.length,invalid:d[2].x.length,missing:d[3].x.length,
                    outside:d[4].x.length,faces:d[0].i.length};} """)
        assert counts['faces'] == 155120
        page.screenshot(path=str(directory / 'report/browser_checked.png'), full_page=True)
        angle_counts = {}
        for angle in ('0', '20', '40', 'best'):
            page.select_option('#angle', angle)
            page.wait_for_function("!document.getElementById('angle').disabled", timeout=120000)
            angle_counts[angle] = page.evaluate("() => {const d=document.getElementById('body').data; return {valid:d[1].x.length,invalid:d[2].x.length,missing:d[3].x.length,outside:d[4].x.length};}")
            if angle == '20':
                page.screenshot(path=str(directory / 'report/browser_angle_zero.png'), full_page=True)
        assert not errors and not requests
        browser.close()
    result = {'passed': True, 'angle_layers': layers, 'best_layer': counts,
              'angles': angle_counts, 'javascript_errors': errors, 'external_requests': requests}
    (directory / 'report/browser_checks.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
