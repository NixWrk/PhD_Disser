"""Runner regressions with tiny hashable files and mocked subprocesses.

The .nii.gz fixtures are plain bytes, not images. Neither the segmenter nor the
analyzer is executed; this suite requires only the Python standard library.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNNER_PATH = Path(__file__).parents[1] / "run_historical_cardiac_sequence.py"
PILOT = (0, 5, 10, 15)
MASK_NAMES = ["total_heart/heart.nii.gz"] + [
    f"heartchambers_highres/{name}.nii.gz" for name in (
        "heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
        "heart_atrium_right", "heart_ventricle_right", "aorta", "pulmonary_artery",
    )
]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class HistoricalRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.workflow = self.root / "workflow"
        self.workflow.mkdir()
        # Run the current source in an isolated tree, so its relative paths and
        # provenance checks cannot touch an active run or real medical data.
        self.runner_path = self.workflow / RUNNER_PATH.name
        self.runner_path.write_bytes(RUNNER_PATH.read_bytes())
        for filename in ("segment_totalseg_v2.py", "analyze_historical_cardiac_sequence.py"):
            (self.workflow / filename).write_text("# Hash-only fixture; never executed.\n", encoding="utf-8")
        spec = importlib.util.spec_from_file_location("historical_runner_under_test", self.runner_path)
        self.runner = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(self.runner)

        self.sequence = self.root / "sequence"
        phase_dir = self.sequence / "phases"
        phase_dir.mkdir(parents=True)
        self.phases = []
        for index in range(20):
            phase_id = f"phase_{index:02d}"
            path = phase_dir / f"{phase_id}.nii.gz"
            path.write_bytes(f"FAKE CT bytes {index}".encode())
            self.phases.append({"phase_id": phase_id, "phase_percent": index * 5,
                                "nifti": path.name, "sha256": file_hash(path)})
        self.conversion = phase_dir / "conversion_manifest.json"
        write_json(self.conversion, {"case_id": "test_case", "sequence_id": "test_sequence",
                                     "phases": self.phases})
        self.out = self.root / "run"
        self.config = {
            "sequence_dir": str(self.sequence), "output_dir": str(self.out),
            "segmentation_python": "mock-segmentation-python",
            "analysis_python": "mock-analysis-python", "gpu": "0",
            "notebook": str(self.out / "report.ipynb"),
            "expected_totalsegmentator_version": "2.18.0",
        }
        self.config_path = self.root / "config.json"
        write_json(self.config_path, self.config)
        self.state_path = self.out / "run_status.json"

    def phase_outputs(self, index, *, weights="original"):
        """Create the segmenter's eight expected files and a model manifest."""
        phase_id = self.phases[index]["phase_id"]
        masks = self.out / "automatic_masks_v2"
        files = []
        for name in MASK_NAMES:
            path = masks / phase_id / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"FAKE mask {phase_id} {name}".encode())
            files.append(path)
        manifest = {
            "backend_version": "2.18.0", "torch": "2.10.0+cu128", "cuda_runtime": "12.8",
            "models": {"whole_heart": {"task": "total"},
                       "anatomical_components": {"task": "heartchambers_highres"}},
            "model_files": [{"relative_tail": "Dataset301/fold_0/checkpoint_final.pth",
                             "sha256": hashlib.sha256(weights.encode()).hexdigest()}],
            "runs": [{"phase_id": phase_id, "status": "automatic_segmentation_pending_manual_review"}],
        }
        path = masks / f"segmentation_manifest_gpu0_p{index}_{index + 1}.json"
        write_json(path, manifest)
        files.append(path)
        return files, manifest

    def seed_completed_pilot(self, report_flag):
        phase_hashes = {}
        for index in PILOT:
            files, manifest = self.phase_outputs(index)
            phase_hashes[self.phases[index]["phase_id"]] = {
                str(path.relative_to(self.out)): file_hash(path) for path in files
            }
        state = {
            "status": "failed", "completed_phase_ids": [self.phases[i]["phase_id"] for i in PILOT],
            "phase_hashes": phase_hashes,
            "input_signatures": {
                "runner": file_hash(self.runner_path), "config": json_hash(self.config),
                "conversion": file_hash(self.conversion),
                "segmenter": file_hash(self.workflow / "segment_totalseg_v2.py"),
                "analyzer": file_hash(self.workflow / "analyze_historical_cardiac_sequence.py"),
            },
            "model_signature": json_hash({key: manifest[key] for key in (
                "backend_version", "torch", "cuda_runtime", "models", "model_files")}),
            "accuracy_evaluated": False, "reference_masks_available": False,
        }
        if report_flag is not None:
            state["pilot_report_written"] = report_flag
        write_json(self.state_path, state)
        return state

    def invoke(self, subprocess_mock):
        with mock.patch.object(sys, "argv", [str(self.runner_path), "--config", str(self.config_path)]), \
                mock.patch.object(self.runner.subprocess, "run", subprocess_mock):
            self.runner.main()

    def state(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def test_resume_empty_pilot_never_starts_another_segmentation(self):
        # Missing/failed reports and a stale success flag must all be rechecked.
        for flag in (None, False, True):
            with self.subTest(pilot_report_written=flag):
                initial = self.seed_completed_pilot(flag)

                def empty_report(command, **kwargs):
                    self.assertIn(str(self.workflow / "analyze_historical_cardiac_sequence.py"), command)
                    self.assertIn("--allow-partial", command)
                    write_json(self.out / "analysis" / "automatic_sequence_summary.json",
                               {"rows": [{"empty": True}]})

                process = mock.Mock(side_effect=empty_report)
                with self.assertRaisesRegex(RuntimeError, "Empty target mask"):
                    self.invoke(process)
                process.assert_called_once()
                self.assertFalse(any(str(self.workflow / "segment_totalseg_v2.py") in call.args[0]
                                     for call in process.call_args_list))
                self.assertEqual(self.state()["status"], "failed")
                self.assertEqual(self.state()["completed_phase_ids"], initial["completed_phase_ids"])

    def test_extra_phase_file_blocks_all_subprocesses(self):
        # Sorts between legitimate inputs and would shift the segmenter's index.
        (self.sequence / "phases" / "phase_00_extra.nii.gz").write_bytes(b"UNDECLARED fake phase")
        process = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "Phase files differ"):
            self.invoke(process)
        process.assert_not_called()
        self.assertFalse(self.state_path.exists())

    def test_changed_weights_on_next_phase_cannot_be_completed(self):
        segmented = []

        def segment(command, **kwargs):
            self.assertIn(str(self.workflow / "segment_totalseg_v2.py"), command)
            index = int(command[command.index("--phase-start") + 1])
            self.assertEqual(int(command[command.index("--phase-stop") + 1]), index + 1)
            segmented.append(index)
            self.phase_outputs(index, weights="original" if len(segmented) == 1 else "changed")

        process = mock.Mock(side_effect=segment)
        with self.assertRaisesRegex(RuntimeError, "Model weights or environment changed"):
            self.invoke(process)
        # 25% maps to phase index 5, not index 25 or the second consecutive file.
        self.assertEqual(segmented, [0, 5])
        self.assertEqual(process.call_count, 2)
        state = self.state()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["completed_phase_ids"], ["phase_00"])
        self.assertNotIn("phase_05", state["phase_hashes"])


if __name__ == "__main__":
    unittest.main()
