import io
import pathlib

import onnx
import onnx.shape_inference
import torch
from datasets import load_dataset
from transformers import ViTImageProcessor, ViTForImageClassification

from test import vit_self_attention

base_path = pathlib.Path(__file__).parent


def vit_image_classifier_self_attention():
    torch_vit_image_classifier = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    vit_config = torch_vit_image_classifier.vit.config
    image_size, patch_size = vit_config.image_size, vit_config.patch_size
    num_patches = (image_size // patch_size)**2
    embeddings_size = num_patches + 1
    onnx_model = vit_self_attention(1,
                                    embeddings_size,
                                    vit_config.hidden_size,
                                    vit_config.num_attention_heads)
    return onnx_model


def vit_image_classifier(batch_axis_dynamic: bool):
    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]
    torch_vit_image_classifier = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    inputs = feature_extractor(image, return_tensors="pt")

    if batch_axis_dynamic:
        dynamic_axis = {'inputs': {0: 'B'}}
    else:
        dynamic_axis = {}

    onnx_bytes = io.BytesIO()
    torch.onnx.export(
        torch_vit_image_classifier,
        tuple(inputs.values()),
        f=onnx_bytes,
        input_names=['inputs'],
        output_names=['logits'],
        dynamic_axes=dynamic_axis,
        do_constant_folding=True,
        opset_version=17,
    )
    onnx_model = onnx.load_from_string(onnx_bytes.getvalue())
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    return onnx_model


if __name__ == "__main__":
    onnx.save(vit_image_classifier(batch_axis_dynamic=True),
              base_path / "vit" / "vit_image_classifier.onnx",
              location=base_path / "vit" / "vit_image_classifier")
    # Store onnx files without weights to view them with netron
    onnx.save(vit_image_classifier(batch_axis_dynamic=False),
              base_path / "vit" / "vit_image_classifier_no_weights.onnx",
              location=base_path / "vit" / "vit_image_classifier_no_weights.data",
              save_as_external_data=True)
    onnx.save(vit_image_classifier_self_attention(),
              base_path / "vit" / "vit_image_classifier_self_attention_no_weights.onnx",
              location=base_path / "vit" / "vit_image_classifier_self_attention_no_weights.data",
              save_as_external_data=True)
