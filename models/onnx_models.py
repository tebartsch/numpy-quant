import io
import pathlib
from itertools import zip_longest

import onnx
import onnx.shape_inference
import onnx.numpy_helper
import numpy as np
import torch
from datasets import load_dataset
from transformers import ViTConfig, ViTImageProcessor
from transformers.models.vit.modeling_vit import ViTSelfAttention, ViTModel, ViTEmbeddings, ViTPooler, ViTLayer, \
    ViTForImageClassification

base_path = pathlib.Path(__file__).parent


def shapes_broadcastable(shape_a: tuple[int, ...], shape_b: tuple[int, ...]):
    return all((m == n) or (m == 1) or (n == 1) for m, n in zip(shape_a[::-1], shape_b[::-1]))


def gemm(k: int, m: int, n: int, random_seed: int):
    rng = np.random.default_rng(random_seed)

    input_name, output_name = "input", "output"
    input = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [k, m])
    output = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [k, n])

    weight_data = rng.normal(size=(m, n)).astype(np.float32)
    bias_data = rng.normal(size=n).astype(np.float32)

    weight_name = "weight"
    weight = onnx.numpy_helper.from_array(weight_data, weight_name)

    bias_name = "bias"
    bias = onnx.numpy_helper.from_array(bias_data, bias_name)

    node = onnx.helper.make_node(
        name="Gemm",
        op_type="Gemm",
        inputs=[input_name, weight_name, bias_name],
        outputs=[output_name],
    )

    graph_def = onnx.helper.make_graph(
        nodes=[node],
        name="Gemm",
        inputs=[input],
        outputs=[output],
        initializer=[weight, bias],
    )

    onnx_model = onnx.helper.make_model(graph_def, producer_name="numpy-quant-test")
    onnx_model.opset_import[0].version = 13

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)

    return onnx_model


def matmul(a_shape: tuple[int, ...], b_shape: tuple[int, ...]):
    assert len(a_shape) > 2, "Input shape a must habe at least 2 dimensions"
    assert len(b_shape) > 2, "Input shape b must habe at least 2 dimensions"
    assert a_shape[-1] == b_shape[-2], (f"shapes a_shape={a_shape}, b_shape={b_shape} have no matching last and "
                                        f"second last dimension.")
    assert shapes_broadcastable(a_shape[:-2], b_shape[:-2]), (
        f"shapes a_shape[:-2]={a_shape[:-2]}, b_shape[:-2]={b_shape[:-2]} are not broadcastable.")

    out_shape = (np.broadcast_shapes(a_shape[:-2], b_shape[:-2]) + a_shape[-2:-1] + b_shape[-1:])

    input_a_name, input_b_name, output_name = "input_a", "input_b", "output"
    input_a = onnx.helper.make_tensor_value_info(input_a_name, onnx.TensorProto.FLOAT, list(a_shape))
    input_b = onnx.helper.make_tensor_value_info(input_b_name, onnx.TensorProto.FLOAT, list(b_shape))
    output = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, list(out_shape))

    node = onnx.helper.make_node(
        name="MatMul",
        op_type="MatMul",
        inputs=[input_a_name, input_b_name],
        outputs=[output_name],
    )

    graph_def = onnx.helper.make_graph(
        nodes=[node],
        name="MatMul",
        inputs=[input_a, input_b],
        outputs=[output],
    )

    onnx_model = onnx.helper.make_model(graph_def, producer_name="numpy-quant-test")
    onnx_model.opset_import[0].version = 13

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)

    return onnx_model


def conv(b: int,
         c: int,
         inp_shape: (int, int),
         out_c: int,
         kernel_shape: (int, int),
         pads: (int, int, int, int),
         strides: (int, int),
         random_seed: int):
    rng = np.random.default_rng(random_seed)

    out_width = (inp_shape[0] - kernel_shape[0] + pads[0] + pads[2]) // strides[0] + 1
    out_height = (inp_shape[1] - kernel_shape[1] + pads[1] + pads[3]) // strides[1] + 1

    input_name, output_name = "input", "output"
    input = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [b, c, *inp_shape])
    output = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT,
                                                [b, out_c, out_width, out_height])

    weight_data = rng.normal(size=(out_c, c, *kernel_shape)).astype(np.float32)
    bias_data = rng.normal(size=out_c).astype(np.float32)

    weight_name = "weight"
    weight = onnx.numpy_helper.from_array(weight_data, weight_name)

    bias_name = "bias"
    bias = onnx.numpy_helper.from_array(bias_data, bias_name)

    node = onnx.helper.make_node(
        name="Conv",
        op_type="Conv",
        inputs=[input_name, weight_name, bias_name],
        outputs=[output_name],
        kernel_shape=kernel_shape,
        pads=pads,
        strides=strides,
    )

    graph_def = onnx.helper.make_graph(
        nodes=[node],
        name="Conv",
        inputs=[input],
        outputs=[output],
        initializer=[weight, bias],
    )

    onnx_model = onnx.helper.make_model(graph_def, producer_name="numpy-quant-test")
    onnx_model.opset_import[0].version = 13

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)

    return onnx_model


def expand():
    input_name, shape_name, output_name = "input", "shape", "output"
    input = onnx.helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [1, 1, 8])
    shape = onnx.helper.make_tensor_value_info(shape_name, onnx.TensorProto.INT64, [3])
    output = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [1, 1, 8])

    node = onnx.helper.make_node(
        name="Expand",
        op_type="Expand",
        inputs=[input_name, shape_name],
        outputs=[output_name],
    )

    graph_def = onnx.helper.make_graph(
        nodes=[node],
        name="Expand",
        inputs=[input, shape],
        outputs=[output],
    )

    onnx_model = onnx.helper.make_model(graph_def, producer_name="numpy-quant-test")
    onnx_model.opset_import[0].version = 13

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)

    return onnx_model


def vit_embedding(batch_size: int, image_size: int, patch_size: int, hidden_size: int):
    vit_config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
    )
    pytorch_model = ViTEmbeddings(vit_config)
    onnx_model_bytes = io.BytesIO()
    torch.onnx.export(
        pytorch_model,
        torch.zeros((batch_size, 3, image_size, image_size)),
        f=onnx_model_bytes,
        input_names=['inputs'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=17,
    )
    onnx_model = onnx.load_from_string(onnx_model_bytes.getvalue())
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    return onnx_model


def vit_self_attention(batch_size: int, embeddings_size: int, hidden_size: int, num_attention_heads: int):
    vit_config = ViTConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    pytorch_model = ViTSelfAttention(vit_config)

    onnx_model_bytes = io.BytesIO()
    torch.onnx.export(
        pytorch_model,
        torch.zeros((batch_size, embeddings_size, hidden_size)),
        f=onnx_model_bytes,
        input_names=['inputs'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=17,
    )
    onnx_model = onnx.load_from_string(onnx_model_bytes.getvalue())
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    return onnx_model


def vit_layer(batch_size: int, image_size: int, patch_size: int, intermediate_size: int,
              hidden_size: int, num_attention_heads: int):
    vit_config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
    )
    pytorch_model = ViTLayer(vit_config)

    onnx_model_bytes = io.BytesIO()
    torch.onnx.export(
        pytorch_model,
        torch.zeros((batch_size, (image_size // patch_size) ** 2 + 1, hidden_size)),
        f=onnx_model_bytes,
        input_names=['inputs'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=17,
    )
    onnx_model = onnx.load_from_string(onnx_model_bytes.getvalue())
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    return onnx_model


def vit_pooler(batch_size: int, image_size: int, patch_size: int, hidden_size: int):
    vit_config = ViTConfig(
        batch_size=batch_size,
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
    )
    pytorch_model = ViTPooler(vit_config)

    onnx_model_bytes = io.BytesIO()
    torch.onnx.export(
        pytorch_model,
        torch.zeros((batch_size, (image_size // patch_size) ** 2 + 1, hidden_size)),
        f=onnx_model_bytes,
        input_names=['inputs'],
        output_names=['pooler_output'],
        do_constant_folding=True,
        opset_version=17,
    )
    onnx_model = onnx.load_from_string(onnx_model_bytes.getvalue())
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    return onnx_model


def vit(batch_size: int, image_size: int, patch_size: int, intermediate_size: int,
        hidden_size: int, num_attention_heads: int):
    vit_config = ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
    )
    pytorch_model = ViTModel(vit_config)

    onnx_model_bytes = io.BytesIO()
    torch.onnx.export(
        pytorch_model,
        torch.zeros((batch_size, 3, image_size, image_size)),
        f=onnx_model_bytes,
        input_names=['inputs'],
        output_names=['hidden_states'],
        do_constant_folding=True,
        opset_version=17,
    )
    onnx_model = onnx.load_from_string(onnx_model_bytes.getvalue())
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    return onnx_model


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
    onnx.save(gemm(k=3, m=4, n=2, random_seed=0),
              base_path / "test" / "gemm.onnx")
    onnx.save(matmul(a_shape=(2, 1, 4, 3), b_shape=(1, 3, 3, 5)),
              base_path / "test" / "matmul.onnx")
    onnx.save(conv(b=2, c=3, inp_shape=(9, 10), out_c=2,
                   kernel_shape=(3, 2), pads=(0, 2, 2, 1), strides=(2, 1),
                   random_seed=0),
              base_path / "test" / "conv.onnx")
    onnx.save(expand(), base_path / "test" / "expand.onnx")
    onnx.save(vit_self_attention(batch_size=1, embeddings_size=10, hidden_size=16, num_attention_heads=4),
              base_path / "test" / "vit_self_attention.onnx")
    onnx.save(vit_embedding(batch_size=1, image_size=16, patch_size=4, hidden_size=8),
              base_path / "test" / "vit_embedding.onnx")
    onnx.save(vit_layer(batch_size=1, image_size=16, patch_size=4, intermediate_size=22,
                        hidden_size=8, num_attention_heads=4),
              base_path / "test" / "vit_layer.onnx")
    onnx.save(vit_pooler(batch_size=1, image_size=16, patch_size=4, hidden_size=8),
              base_path / "test" / "vit_pooler.onnx")
    onnx.save(vit(batch_size=2, image_size=16, patch_size=4, intermediate_size=22,
                  hidden_size=8, num_attention_heads=2),
              base_path / "test" / "vit.onnx")

    onnx.save(vit_image_classifier(), base_path / "vit_image_classifier.onnx")
