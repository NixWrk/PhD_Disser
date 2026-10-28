"""Check the assembled reader's links, tables and equation numbering."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote
from assemble_report import CHAPTERS, ROOT


def main():
    main_path = ROOT / 'ОТЧЁТ_ГПХ_РНЦХ.md'
    files = [main_path] + [ROOT / 'chapters' / name for name in CHAPTERS]
    bad_links, open_tables = [], []
    count = 0
    for file in files:
        text = file.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', text):
            target = target.strip('<>')
            if re.match(r'(?:[a-zA-Z][a-zA-Z0-9+.-]*:|#)', target):
                continue
            target = unquote(target.split('#')[0])
            count += 1
            if not (file.parent / target).is_file():
                bad_links.append({'file': str(file.relative_to(ROOT)), 'target': target})
        depth = 0
        for number, line in enumerate(text.splitlines(), 1):
            depth += len(re.findall(r'<details(?:\s|>)', line))
            if line.startswith('|') and depth < 1:
                open_tables.append({'file': str(file.relative_to(ROOT)), 'line': number})
            if re.search(r'<details\b[^>]*\bopen\b', line):
                open_tables.append({'file': str(file.relative_to(ROOT)), 'line': number, 'reason': 'open_details'})
            depth -= line.count('</details>')
    text = main_path.read_text(encoding='utf-8')
    tags = re.findall(r'\\tag\{([^}]+)\}', text)
    duplicates = sorted({x for x in tags if tags.count(x) > 1})
    result = {
        'status': 'documentation_checks_passed_not_scientific_validation'
            if not bad_links and not duplicates and not open_tables else 'failed',
        'chapter_count': len(CHAPTERS), 'word_count_whitespace': len(text.split()),
        'characters': len(text), 'numbered_equations': len(tags),
        'tables': len(re.findall(r'^\|\s*:?-', text, re.M)),
        'figures': len(re.findall(r'!\[', text)),
        'local_links_checked_in_main_and_sources': count, 'broken_links': bad_links,
        'duplicate_equation_numbers': duplicates, 'tables_not_collapsed': open_tables,
        'source_chapter_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files[1:]},
        'report_sha256': hashlib.sha256(main_path.read_bytes()).hexdigest(),
        'limits': ['Empirical CT cross-person errors are distinct from electrical SV/EF accuracy',
                   'No new FEM or synthetic observations in the empirical continuation',
                   'Historical dynamic CT and October electrical cohorts overlap in two people',
                   'Latest Nix-only chapter uses dependent phase blocks and three early RR1 points, not independent subject validation',
                   'Observed-window ranges and relative ranges are not validated clinical SV or EF',
                   'PulmoVista chapter includes new conditional real-CT/EIT inversion and TTRKG projection, not independent tissue validation',
                   'Nix conditional baseline residual is 32.44 percent; all-free solution rejected; correction-to-SV/EF chain remains unvalidated',
                   'Chapter 14 adds physical STL/FEM geometry probes and per-shape effective fits; fixed rho4 does not reject sphere',
                   'Whole-heart volume and effective resistivity changes require separate identifiability and dynamic validation',
                   'Relative pulse normalization improves a common held-out electrical metric conditionally; voltage residuals exceed 100 percent and volume accuracy is not established',
                   'Static/dynamic cardiac definitions and registration remain unresolved; equivalent whole-heart form precedes the chamber-function transition',
                   'Chapter 15 compares 28 implemented 3D approximations plus control on 12 automatic masks of one person; all masks touch the superior CT field boundary',
                   'Exact volume and centroid in constrained approximations use target CT inputs and are not reconstruction accuracy',
                   '432 single-anchor transformations still use target moments; geometric superiority does not establish electrical SV/EF accuracy',
                   'Chapter 16 adds 864 fixed/moving geometry probes, 99 motion-law checks and 42 conditional FEM probes; target CT volumes remain known inputs in the geometry tests',
                   'Target EIT supplies rho and delta-rho only; corrected TT supplies geometry, while existing separate recordings remain unsynchronized',
                   'Geometry-only EIT signals can leak into estimated tissue corrections; linear diagnostics do not establish measured tissue changes',
                   'Five-parameter rankings use explicit diagnostic steps and one TT montage; full nonlinear interactions and matched multi-montage comparison are not completed',
                   'Historical spherical radius changes also moved its center via z=r+h; earlier electrode sweeps held cardiac geometry fixed'],
        'figure_visual_review': 'Previous approximation/timing figures and latest Nix direction, guarded chamber predictions and formula-error PNGs inspected; chapter 14 shape/material, measurement-residual and composition/normalization PNGs inspected; chapter 15 cycle_0_overview, cycle_0_dynamics_3 and cycle_0_trajectory_3 PNGs inspected and final copies verified by SHA256; chapter 16 six PNGs inspected and reader copies verified by SHA256; remaining appendix figures not individually visually reviewed',
    }
    (ROOT / 'report_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_chapter_sha256', 'limits')}, ensure_ascii=False))
    return 0 if result['status'].startswith('documentation_checks_passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
