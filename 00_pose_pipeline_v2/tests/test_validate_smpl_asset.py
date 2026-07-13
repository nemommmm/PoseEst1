"""Tests for SMPL asset validation helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from validate_smpl_asset import module_device  # noqa: E402


class ValidateSmplAssetTest(unittest.TestCase):
    """Cover parameter, buffer, and parameterless module device discovery."""

    def test_buffer_only_module(self) -> None:
        module = torch.nn.Module()
        module.register_buffer("template", torch.zeros(1))
        self.assertEqual(module_device(module, torch.ones(1)), torch.device("cpu"))

    def test_parameterless_module_uses_output(self) -> None:
        module = torch.nn.Module()
        self.assertEqual(module_device(module, torch.ones(1)), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
