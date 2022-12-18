#!/usr/bin/env python
import io
import pathlib
import unittest
from time import time

import numpy as np
import onnx
import onnx.helper
import onnxruntime as ort
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification

from tinyquant.model import Model, Variable
from tinyquant.tensor import FTensor


def compare_all_nodes(onnx_model: onnx.ModelProto, input_data: dict[str, np.ndarray]):
    for value in onnx_model.graph.input:
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = value.name
        onnx_model.graph.output.append(intermediate_layer_value_info)
    for value in onnx_model.graph.value_info:
        intermediate_layer_value_info = onnx.helper.ValueInfoProto()
        intermediate_layer_value_info.name = value.name
        onnx_model.graph.output.append(intermediate_layer_value_info)

    onnx_bytes = io.BytesIO()
    onnx.save_model(onnx_model, onnx_bytes)
    ort_sess = ort.InferenceSession(onnx_bytes.getvalue())

    output_names = [o.name for o in onnx_model.graph.output]
    outputs = ort_sess.run(None, input_data)
    desired = {name: output for name, output in zip(output_names, outputs)}

    model = Model.from_onnx(onnx_model)
    model(list(FTensor(v) for v in input_data.values()))
    actual = {v.name: v.data.data for v in model.values if isinstance(v, Variable)}

    assert set(desired) == set(actual), "ONNX model and Tinyquant model should have the same nodes"
    for node in onnx_model.graph.node:
        if node.op_type in ["Constant", "ConstantOfShape"]:
            continue
        for output_name in node.output:
            mean_element_l1 = np.mean(np.abs(actual[output_name] - desired[output_name]))
            print(output_name, mean_element_l1)
            np.testing.assert_almost_equal(mean_element_l1, 0.0, decimal=4,
                                           err_msg=f"Mean of elementwise l1 norm for "
                                                   f"{output_name}: {mean_element_l1}")


class TestMlp(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestMlp, self).__init__(*args, **kwargs)
        self.onnx_model = onnx.load(pathlib.Path(__file__).parent / ".." / "models" / "vit_image_classifier.onnx")
        self.feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.torch_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

        dataset = load_dataset("huggingface/cats-image")
        image = dataset["test"]["image"][0]
        inputs_dict = self.feature_extractor(image, return_tensors="pt")
        self.cat_input_data = inputs_dict['pixel_values'].numpy()

    def test_vit_image_classifier_all_nodes(self):
        compare_all_nodes(self.onnx_model, {"inputs": self.cat_input_data})

    def test_vit_image_classifier_single_image(self):
        print("Preparing")

        onnx_bytes = io.BytesIO()
        onnx.save_model(self.onnx_model, onnx_bytes)
        ort_sess = ort.InferenceSession(onnx_bytes.getvalue())
        startime = time()
        desired_logits = ort_sess.run(None, {'inputs': self.cat_input_data})[0]
        onnx_time = time() - startime
        desired_label = self.torch_model.config.id2label[desired_logits.argmax(axis=-1)[0]]

        print("Create Model from ONNX")
        model = Model.from_onnx(self.onnx_model)

        print("Run inference")
        startime = time()
        actual_logits = model([FTensor(self.cat_input_data)])[0].data
        tinyquant_time = time() - startime
        actual_label = self.torch_model.config.id2label[actual_logits.argmax(axis=-1)[0]]

        print(desired_label)
        print(actual_label)

        print(f"ONNX Inference Time: {onnx_time:.2f}s")
        print(f"Tinyquant Inference Time: {tinyquant_time:.2f}s")

    def test_vit_quantization(self):
        model = Model.from_onnx(self.onnx_model)
        # qmodel = model.quantize([FTensor(self.cat_input_data)])
