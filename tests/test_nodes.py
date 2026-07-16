import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from uuid import uuid4

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class FakeImage:
    def __init__(self, pixels):
        self.pixels = pixels

    def cpu(self):
        return self

    def numpy(self):
        return self.pixels


class LocalRunReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "output"
        self.output.mkdir()
        self.previous_folder_paths = sys.modules.get("folder_paths")
        self.folder_paths = types.ModuleType("folder_paths")
        self.folder_paths.get_output_directory = lambda: str(self.output)
        sys.modules["folder_paths"] = self.folder_paths

        self.module_name = f"local_run_receipts_test_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(
            self.module_name,
            ROOT / "__init__.py",
            submodule_search_locations=[str(ROOT)],
        )
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = self.module
        spec.loader.exec_module(self.module)
        self.nodes = sys.modules[f"{self.module_name}.nodes"]

    def tearDown(self):
        for name in list(sys.modules):
            if name == self.module_name or name.startswith(f"{self.module_name}."):
                del sys.modules[name]
        if self.previous_folder_paths is None:
            del sys.modules["folder_paths"]
        else:
            sys.modules["folder_paths"] = self.previous_folder_paths
        self.temp.cleanup()

    def build(self, parameters='{"sampler":"euler","steps":20}', seed=12):
        return self.module.BuildLocalRunKey().build("demo", "SD 1.5", seed, parameters, "")

    @staticmethod
    def image(red=0):
        pixels = np.zeros((4, 5, 3), dtype=np.float32)
        pixels[..., 0] = red
        return FakeImage(pixels)

    def test_key_is_stable_for_equivalent_parameter_json(self):
        key_a, record_a = self.build('{"steps":20,"sampler":"euler"}')
        key_b, record_b = self.build('{\n  "sampler": "euler",\n  "steps": 20\n}')
        self.assertEqual(key_a, key_b)
        self.assertEqual(record_a, record_b)
        self.assertTrue(key_a.startswith("lrr1_"))

    def test_key_changes_when_a_declared_value_changes(self):
        key_a, _ = self.build(seed=12)
        key_b, _ = self.build(seed=13)
        self.assertNotEqual(key_a, key_b)

    def test_invalid_parameters_are_rejected(self):
        with self.assertRaisesRegex(self.nodes.RunReceiptError, "valid JSON"):
            self.build("not json")
        with self.assertRaisesRegex(self.nodes.RunReceiptError, "JSON object"):
            self.build("[]")

    def test_create_and_repeat_are_idempotent(self):
        run_key, record = self.build()
        commit = self.module.CommitLocalRunImages()
        first = commit.commit_images([self.image(0.5)], run_key, record, "receipts", "image", {"2": {"class_type": "KSampler"}})
        second = commit.commit_images([self.image(0.5)], run_key, record, "receipts", "image", {"2": {"class_type": "KSampler"}})
        self.assertEqual(first["result"][0], "CREATED")
        self.assertEqual(second["result"][0], "ALREADY_IDENTICAL")
        run_directory = self.output / "receipts" / run_key
        receipt = json.loads((run_directory / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["run_key"], run_key)
        self.assertEqual(receipt["artifacts"][0]["filename"], "image-001.png")
        with Image.open(run_directory / "image-001.png") as saved:
            self.assertEqual(saved.size, (5, 4))
            self.assertEqual(saved.info, {})

    def test_changed_image_for_same_key_conflicts(self):
        run_key, record = self.build()
        commit = self.module.CommitLocalRunImages()
        commit.commit_images([self.image(0.1)], run_key, record, "receipts", "image")
        with self.assertRaisesRegex(self.nodes.RunReceiptError, "conflict"):
            commit.commit_images([self.image(0.9)], run_key, record, "receipts", "image")

    def test_incomplete_run_directory_is_never_overwritten(self):
        run_key, record = self.build()
        incomplete = self.output / "receipts" / run_key
        incomplete.mkdir(parents=True)
        with self.assertRaisesRegex(self.nodes.RunReceiptError, "incomplete"):
            self.module.CommitLocalRunImages().commit_images([self.image(0.1)], run_key, record, "receipts", "image")

    def test_unsafe_output_subfolder_is_rejected(self):
        run_key, record = self.build()
        with self.assertRaisesRegex(self.nodes.RunReceiptError, "safe relative"):
            self.module.CommitLocalRunImages().commit_images([self.image(0.1)], run_key, record, "../outside", "image")

    def test_batch_receipt_has_each_artifact(self):
        run_key, record = self.build()
        result = self.module.CommitLocalRunImages().commit_images([self.image(0.1), self.image(0.2)], run_key, record, "receipts", "batch")
        self.assertEqual(result["result"][0], "CREATED")
        receipt = json.loads((self.output / "receipts" / run_key / "receipt.json").read_text(encoding="utf-8"))
        self.assertEqual([item["filename"] for item in receipt["artifacts"]], ["batch-001.png", "batch-002.png"])

    def test_prompt_change_for_same_key_conflicts(self):
        run_key, record = self.build()
        commit = self.module.CommitLocalRunImages()
        commit.commit_images([self.image(0.2)], run_key, record, "receipts", "image", {"1": {"seed": 12}})
        with self.assertRaisesRegex(self.nodes.RunReceiptError, "stored receipt"):
            commit.commit_images([self.image(0.2)], run_key, record, "receipts", "image", {"1": {"seed": 13}})

    def test_commit_is_never_cached(self):
        self.assertTrue(math.isnan(self.module.CommitLocalRunImages.IS_CHANGED()))

    def test_prompt_hash_ignores_comfy_runtime_cache_marker(self):
        prompt = {"1": {"class_type": "EmptyImage", "inputs": {"width": 64}}}
        marked_prompt = {"1": {"class_type": "EmptyImage", "inputs": {"width": 64}, "is_changed": [math.nan]}}
        self.assertEqual(
            self.nodes._prompt_snapshot_hash(prompt),
            self.nodes._prompt_snapshot_hash(marked_prompt),
        )


if __name__ == "__main__":
    unittest.main()
