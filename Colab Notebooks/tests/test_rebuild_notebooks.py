import json
from pathlib import Path
import tempfile
import unittest

import nbformat
from bs4 import BeautifulSoup

from tools.rebuild_notebooks import run_job


class RebuildNotebookTests(unittest.TestCase):
    def _write_notebook(self, path, source):
        notebook = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_markdown_cell("# РџСЂРѕРІРµСЂРєР° С‡РёС‚Р°С‚РµР»СЊСЃРєРѕРіРѕ РѕС‚С‡С‘С‚Р°"),
                nbformat.v4.new_code_cell(source),
            ],
        )
        path.write_text(nbformat.writes(notebook) + "\n", encoding="utf-8")

    def test_clean_execution_records_branch_and_code_free_html(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = root / "demo.ipynb"
            self._write_notebook(
                notebook,
                "import os\nprint('real_data_status: processed')\nassert os.environ['DEMO_INPUT'] == 'measured'",
            )
            work = root / "work"
            result = run_job(
                {
                    "path": "demo.ipynb",
                    "env": {
                        "DEMO_INPUT": "measured",
                        "KALMYKOV_RUN_REAL": "1",
                    },
                },
                root,
                work,
            )
            self.assertEqual(result["status"], "executed")
            self.assertEqual(
                result["branch_evidence"]["classification"],
                "configured_data_branch_with_status_evidence",
            )
            html = notebook.with_suffix(".html").read_text(encoding="utf-8")
            self.assertNotIn("os.environ['DEMO_INPUT']", html)
            soup = BeautifulSoup(html, "html.parser")
            self.assertFalse(soup.select(".jp-CodeCell .jp-InputArea"))
            self.assertIn("real_data_status: processed", html)

    def test_failure_keeps_source_and_refreshes_reader_html(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = root / "failing.ipynb"
            self._write_notebook(notebook, "raise RuntimeError('blocked input')")
            before = notebook.read_bytes()
            result = run_job({"path": "failing.ipynb"}, root, root / "work")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(notebook.read_bytes(), before)
            html = notebook.with_suffix(".html")
            self.assertTrue(html.is_file())
            self.assertNotIn("raise RuntimeError", html.read_text(encoding="utf-8"))
            self.assertEqual(result["html_source"], "fresh_render_from_unchanged_saved_notebook_after_failure")

    def test_invalid_notebook_is_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = root / "invalid.ipynb"
            notebook.write_text("{broken", encoding="utf-8")
            result = run_job({"path": "invalid.ipynb"}, root, root / "work")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["error_type"], "NotJSONError")


if __name__ == "__main__":
    unittest.main()
