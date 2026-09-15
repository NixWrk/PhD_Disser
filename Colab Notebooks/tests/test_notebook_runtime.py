import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from notebook_runtime import _prepare_config, configure_local_data


class NotebookRuntimeTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            name: os.environ.get(name)
            for name in (
                "KALMYKOV_EXP02_CONFIG",
                "KALMYKOV_NOTEBOOK_WORK",
                "KALMYKOV_RUN_REAL",
                "KALMYKOV_RUN_EXPLORATORY",
                "KALMYKOV_RUNTIME_EVENT",
            )
        }
        for name in self._saved:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _make_source(self, root):
        data_root = root / "data"
        derived_root = root / "derived"
        params_root = root / "params"
        timestamps_root = root / "timestamps"
        for path in (data_root, derived_root, params_root, timestamps_root):
            path.mkdir()
        (data_root / "record.csv").write_text("measured\n", encoding="utf-8")
        (derived_root / "existing.json").write_text("{}", encoding="utf-8")
        (params_root / "source.json").write_text("{}", encoding="utf-8")
        (timestamps_root / "source.json").write_text("{}", encoding="utf-8")
        notebook_root = root / "notebooks"
        config_root = notebook_root / "config"
        config_root.mkdir(parents=True)
        config = {
            "data_root": str(data_root),
            "derived_root": str(derived_root),
            "legacy_params_dir": str(params_root),
            "legacy_timestamps_dir": str(timestamps_root),
        }
        source = config_root / "exp02_paths.local.json"
        source.write_text(
            json.dumps(config, ensure_ascii=False),
            encoding="utf-8",
        )
        return notebook_root, source, data_root, derived_root

    def test_local_config_is_prepared_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook_root, source, data_root, derived_root = self._make_source(root)
            work = root / "work"
            selected = configure_local_data(notebook_root, work)
            prepared = Path(selected["KALMYKOV_EXP02_CONFIG"])
            config = json.loads(prepared.read_text(encoding="utf-8"))
            self.assertNotEqual(Path(config["derived_root"]).resolve(), derived_root.resolve())
            self.assertTrue(Path(config["derived_root"]).joinpath("existing.json").is_file())
            self.assertEqual(Path(config["data_root"]).resolve(), data_root.resolve())
            self.assertTrue(Path(config["legacy_params_dir"]).joinpath("source.json").is_file())
            self.assertEqual((derived_root / "existing.json").read_text(encoding="utf-8"), "{}")
            event = json.loads(Path(os.environ["KALMYKOV_RUNTIME_EVENT"]).read_text(encoding="utf-8"))
            self.assertEqual(event["branch"], "isolated_local_configs")
            self.assertEqual(event["configs"][0]["input_origins"]["derived_root"], str(derived_root.resolve()))

    def test_concurrent_preparation_publishes_one_complete_shared_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook_root, source, _, derived_root = self._make_source(root)
            work = root / "work"
            def prepare():
                return _prepare_config(source, work, "KALMYKOV_EXP02_CONFIG")
            with ThreadPoolExecutor(max_workers=4) as pool:
                paths = list(pool.map(lambda _: prepare(), range(4)))
            self.assertEqual({path.resolve() for path in paths}, {paths[0].resolve()})
            prepared = json.loads(paths[0].read_text(encoding="utf-8"))
            clone = Path(prepared["derived_root"])
            self.assertTrue((clone / "existing.json").is_file())
            self.assertNotEqual(clone, derived_root.resolve())

    def test_replay_directory_inside_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook_root, source, _, derived_root = self._make_source(root)
            with self.assertRaises(ValueError):
                _prepare_config(source, derived_root / "replay", "KALMYKOV_EXP02_CONFIG")


if __name__ == "__main__":
    unittest.main()
