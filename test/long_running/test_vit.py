#!/usr/bin/env python
import io
import os
import pathlib
import unittest
from time import time

import datasets
import numpy as np
import onnx
import onnx.helper
import onnxruntime as ort
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification

from extra.evaluate_profile_results import profile_results_plot
from numpy_quant.model import Model, Variable
from numpy_quant.tensor import FTensor


def copy_onnx_model(model: onnx.ModelProto):
    onnx_bytes = io.BytesIO()
    onnx.save(model, onnx_bytes)
    return onnx.load_from_string(onnx_bytes.getvalue())


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

    def test_vit_image_classifier_all_nodes(self):
        dataset = load_dataset("huggingface/cats-image")
        image = dataset["test"]["image"][0]
        inputs_dict = self.feature_extractor(image, return_tensors="pt")
        cat_input_data = inputs_dict['pixel_values'].numpy()

        onnx_model = copy_onnx_model(self.onnx_model)
        make_dim_param_fixed(onnx_model.graph, 'B', 1)

        compare_all_nodes(onnx_model, {"inputs": cat_input_data})

    def test_vit_image_classifier_single_image(self):
        print("Preparing")
        onnx_model = copy_onnx_model(self.onnx_model)
        make_dim_param_fixed(onnx_model.graph, 'B', 1)

        dataset = load_dataset("huggingface/cats-image")
        image = dataset["test"]["image"][0]
        inputs_dict = self.feature_extractor(image, return_tensors="np")
        cat_input_data = inputs_dict['pixel_values']
        onnx_bytes = io.BytesIO()
        onnx.save_model(onnx_model, onnx_bytes)
        ort_sess = ort.InferenceSession(onnx_bytes.getvalue())
        startime = time()
        desired_logits = ort_sess.run(None, {'inputs': cat_input_data})[0]
        onnx_time = time() - startime
        desired_label = self.torch_model.config.id2label[desired_logits.argmax(axis=-1)[0]]

        print("Create Model from ONNX")
        model = Model.from_onnx(onnx_model)

        print("Run inference")
        startime = time()
        actual_logits = model([FTensor(cat_input_data)])[0].data
        tinyquant_time = time() - startime
        actual_label = self.torch_model.config.id2label[actual_logits.argmax(axis=-1)[0]]

        print(desired_label)
        print(actual_label)

        print(f"ONNX Inference Time: {onnx_time:.2f}s")
        print(f"Tinyquant Inference Time: {tinyquant_time:.2f}s")

    def test_vit_quantization(self):
        image_folder = pathlib.Path(__file__).parent / ".." / "test_images" / "vit"
        os.makedirs(image_folder, exist_ok=True)

        onnx_model = copy_onnx_model(self.onnx_model)
        make_dim_param_fixed(onnx_model.graph, 'B', 1)

        image_dataset = datasets.load_dataset('Maysee/tiny-imagenet', split='train')

        n_images = 100000  # https://huggingface.co/datasets/Maysee/tiny-imagenet
        n_use_images = 2
        step = n_images // n_use_images

        image_list = []
        label_list = []

        print("Preprocessing Image")
        for i in range(0, n_images, step):
            data_dict = image_dataset[i]
            image = data_dict['image'].convert(mode='RGB')
            image.save(image_folder / f"{i}.jpg")
            image_list.append(self.feature_extractor(image, return_tensors="np")['pixel_values'])
            label_list.append(data_dict['label'])

        inputs = np.concatenate(image_list, axis=0)
        labels = np.array(label_list, dtype=np.int64)

        print("Create Model from ONNX")
        model = Model.from_onnx(onnx_model)
        qmodel = model.quantize([FTensor(inputs)], bit_width=8)

        print("Run float32 inference")
        startime = time()
        outputs, profile_results = model([FTensor(inputs)], profile=True)
        desired_logits = outputs[0].data
        float32_time = time() - startime
        print(f"Float32 Inference Time: {float32_time:.2f}s")
        del model

        desired_label = self.torch_model.config.id2label[desired_logits.argmax(axis=-1)[0]]
        print(desired_label)

        print("Run int8 inference")
        startime = time()
        outputs, q_profile_results = qmodel([FTensor(inputs)], profile=True)
        actual_logits = outputs[0].data

        int8_time = time() - startime
        print(f"Tinyquant Inference Time: {int8_time:.2f}s")
        del qmodel

        actual_label = self.torch_model.config.id2label[actual_logits.argmax(axis=-1)[0]]
        print(actual_label)

        profile_results_plot(profile_results, q_profile_results)
        self.assertEqual(actual_label, desired_label)
