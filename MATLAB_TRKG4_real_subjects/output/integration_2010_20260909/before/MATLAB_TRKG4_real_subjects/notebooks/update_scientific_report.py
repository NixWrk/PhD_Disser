"""Rebuild scientific prose and HTML while preserving computed results.

No notebook cell or FEM solver is executed. Only declared display-wording
changes may modify code strings and saved Markdown outputs.
"""
from pathlib import Path
import argparse
import ast
import copy
import hashlib
import json
import os
import runpy

import nbformat
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'notebooks/TRKG4_inverse_inhale_results.ipynb'


def digest(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def computational_ast(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = '<string>'
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/scientific_report_20260909')
    parser.add_argument('--preview', action='store_true', help='Write review files without replacing canonical report')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_hash = digest(TARGET)
    original = nbformat.read(TARGET, as_version=4)
    original_code = [c for c in original.cells if c.cell_type == 'code']
    if not original_code or any(c.execution_count is None for c in original_code):
        raise RuntimeError('Executed canonical notebook required; generate/execute missing calculations separately')
    generated = out / 'generated.ipynb'
    old_target = os.environ.get('TRKG4_REPORT_TARGET')
    try:
        os.environ['TRKG4_REPORT_TARGET'] = str(generated)
        namespace = runpy.run_path(str(ROOT / 'notebooks/build_trkg4_inverse_report.py'))
    finally:
        if old_target is None:
            os.environ.pop('TRKG4_REPORT_TARGET', None)
        else:
            os.environ['TRKG4_REPORT_TARGET'] = old_target
    notebook = nbformat.read(generated, as_version=4)
    generated_code = [c for c in notebook.cells if c.cell_type == 'code']
    if len(original_code) != len(generated_code):
        raise RuntimeError('Code-cell structure changed; numerical results cannot be reused automatically')
    wording = namespace['scientific_wording']
    numerical_outputs = 0
    for previous, current in zip(original_code, generated_code):
        assert computational_ast(previous.source) == computational_ast(current.source), 'Computations changed'
        expected_source = previous.source
        for old, new in wording:
            expected_source = expected_source.replace(old, new)
        expected_source = namespace['scientific_code_format'](expected_source)
        assert current.source == expected_source, 'Undeclared code change'
        current.execution_count = previous.execution_count
        current.metadata = copy.deepcopy(previous.metadata)
        current.outputs = copy.deepcopy(previous.outputs)
        for before, after in zip(previous.outputs, current.outputs):
            if 'text/markdown' in after.get('data', {}):
                text = after.data['text/markdown']
                for old, new in wording:
                    text = text.replace(old, new)
                after.data['text/markdown'] = namespace['scientific_markdown'](text)
                # Every other MIME representation is kept unchanged.
                assert {k: v for k, v in before.data.items() if k != 'text/markdown'} == {
                    k: v for k, v in after.data.items() if k != 'text/markdown'}
            else:
                assert before == after
                numerical_outputs += 1
    assert not any(o.output_type == 'error' for c in generated_code for o in c.outputs)
    narrative = ROOT / 'notebooks/TRKG4_inverse_inhale_scientific.md'
    notebook.metadata['scientific_revision'] = {
        'date': '2026-09-09', 'source_notebook_sha256': original_hash,
        'narrative_sha256': digest(narrative),
        'scope': 'prose and display wording; numerical results reused without execution',
        'code_computational_ast_unchanged': True,
    }
    destination = out / 'review.ipynb' if args.preview else TARGET
    assert digest(TARGET) == original_hash, 'Canonical notebook changed concurrently'
    nbformat.validate(notebook)
    nbformat.write(notebook, destination)
    exporter = HTMLExporter(exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
    html, _ = exporter.from_notebook_node(notebook)
    html_path = destination.with_suffix('.html')
    soup = BeautifulSoup(html, 'html.parser')
    # Hide only notebook object representations; figures and tables stay intact.
    for block in soup.find_all('pre'):
        if block.get_text(strip=True).startswith('<matplotlib.legend.Legend at '):
            block.decompose()
    html = str(soup)
    html_path.write_text(html, encoding='utf-8')
    assert not soup.select('.jp-CodeCell .jp-InputArea'), 'Code leaked into reader HTML'
    markdown = '\n\n'.join(c.source for c in notebook.cells if c.cell_type == 'markdown')
    # Include all visible numerical tables and generated prose in the review inventory.
    reader_text = soup.get_text('\n', strip=True)
    (out / 'reader_visible_text.txt').write_text(reader_text, encoding='utf-8')
    (out / 'narrative_for_review.md').write_text(markdown, encoding='utf-8')
    report = {
        'code_cells': len(generated_code), 'markdown_cells': len(notebook.cells) - len(generated_code),
        'numerical_outputs_preserved_exactly': numerical_outputs,
        'code_computational_ast_unchanged': True, 'execution_performed': False,
        'html_code_inputs': 0, 'html_images': len(soup.find_all('img')),
        'html_tables': len(soup.find_all('table')),
        'files_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [destination, html_path, narrative]},
    }
    (out / 'revision_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
