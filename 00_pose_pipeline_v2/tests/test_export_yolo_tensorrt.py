"""GPU-free tests for reproducible Ultralytics TensorRT export."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from export_yolo_tensorrt import (  # noqa: E402
    export_precision,
    sha256_file,
    validate_precisions,
)


class FakeYolo:
    """Record export parameters and create a small fake engine."""

    instances: list["FakeYolo"] = []

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.export_parameters: dict[str, object] | None = None
        self.__class__.instances.append(self)

    def export(self, **parameters):
        """Create the expected adjacent engine without using a GPU."""
        self.export_parameters = parameters
        engine = self.model_path.with_suffix(".engine")
        engine.write_bytes(b"fake-tensorrt-engine")
        return str(engine)


class FailingYolo:
    """Fail if instantiated, proving that existing engines are skipped."""

    def __init__(self, model_path: str) -> None:
        raise AssertionError(f"export must not run for {model_path}")


class TensorRtExportTest(unittest.TestCase):
    """Cover exact export arguments, manifests, and no-overwrite behavior."""

    def setUp(self) -> None:
        FakeYolo.instances.clear()

    def test_fp16_export_uses_exact_fixed_shape_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "yolov8m-pose.pt"
            source.write_bytes(b"source-weight")
            result = export_precision(
                source_model=source,
                output_root=root / "exports",
                precision="fp16",
                imgsz=640,
                batch=1,
                device=0,
                yolo_factory=FakeYolo,
                environment={"test": True},
            )
            self.assertEqual(result["status"], "exported")
            self.assertEqual(
                FakeYolo.instances[0].export_parameters,
                {
                    "format": "engine",
                    "imgsz": 640,
                    "batch": 1,
                    "dynamic": False,
                    "half": True,
                    "device": 0,
                },
            )
            copied = root / "exports/fp16/yolov8m-pose.pt"
            engine = root / "exports/fp16/yolov8m-pose.engine"
            manifest = json.loads(
                (root / "exports/fp16/manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["source_model"]["sha256"], sha256_file(source))
            self.assertEqual(manifest["copied_model"]["sha256"], sha256_file(copied))
            self.assertEqual(manifest["engine"]["sha256"], sha256_file(engine))
            self.assertEqual(manifest["engine"]["bytes"], len(b"fake-tensorrt-engine"))

    def test_fp32_sets_half_false(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.pt"
            source.write_bytes(b"weight")
            result = export_precision(
                source_model=source,
                output_root=root / "exports",
                precision="fp32",
                imgsz=512,
                batch=2,
                device=1,
                yolo_factory=FakeYolo,
                environment={"test": True},
            )
            self.assertEqual(result["status"], "exported")
            self.assertFalse(FakeYolo.instances[0].export_parameters["half"])
            self.assertEqual(FakeYolo.instances[0].export_parameters["imgsz"], 512)
            self.assertEqual(FakeYolo.instances[0].export_parameters["batch"], 2)
            self.assertEqual(FakeYolo.instances[0].export_parameters["device"], 1)

    def test_existing_engine_is_hashed_and_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.pt"
            source.write_bytes(b"weight")
            precision_dir = root / "exports/fp16"
            precision_dir.mkdir(parents=True)
            (precision_dir / "model.pt").write_bytes(b"weight")
            engine = precision_dir / "model.engine"
            engine.write_bytes(b"existing-engine")
            result = export_precision(
                source_model=source,
                output_root=root / "exports",
                precision="fp16",
                imgsz=640,
                batch=1,
                device=0,
                yolo_factory=FailingYolo,
                environment={"test": True},
            )
            self.assertEqual(result["status"], "skipped_existing")
            self.assertEqual(engine.read_bytes(), b"existing-engine")
            self.assertEqual(result["engine"]["sha256"], sha256_file(engine))

    def test_precision_validation_deduplicates_and_rejects_unknown(self) -> None:
        self.assertEqual(
            validate_precisions(["fp16", "fp32", "fp16"]),
            ["fp16", "fp32"],
        )
        with self.assertRaises(ValueError):
            validate_precisions(["int8"])


if __name__ == "__main__":
    unittest.main()
