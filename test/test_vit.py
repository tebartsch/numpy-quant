#!/usr/bin/env python
import pathlib
import unittest

import onnx


class TestMlp(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMlp, self).__init__(*args, **kwargs)
        onnx_model = onnx.load(pathlib.Path(__file__).parent / ".." / "models" / "vit.onnx")

    def test_todo(self):
        pass
