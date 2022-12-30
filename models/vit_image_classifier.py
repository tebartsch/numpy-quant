import io
import pathlib

import onnx
import onnx.shape_inference
import torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification

base_path = pathlib.Path(__file__).parent


def vit_image_classifier():
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    inputs = feature_extractor(image, return_tensors="pt")

    onnx_bytes = io.BytesIO()
    torch.onnx.export(
        model,
        tuple(inputs.values()),
        f=onnx_bytes,
        input_names=['inputs'],
        output_names=['logits'],
        dynamic_axes={'inputs': {0: 'B'}},
        do_constant_folding=True,
        opset_version=17,
    )
    onnx_model = onnx.load_from_string(onnx_bytes.getvalue())
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    return onnx_model


if __name__ == "__main__":
    onnx.save(vit_image_classifier(), base_path / "vit_image_classifier-no-weights.onnx",
              location=base_path / "vit_image_classifier")
    onnx.save(vit_image_classifier(), base_path / "vit_image_classifier_no_weights.onnx",
              location=base_path / "vit_image_classifier_no_weights.data",
              save_as_external_data=True)
