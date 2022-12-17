import onnx
import onnx.shape_inference
import onnx.numpy_helper
import numpy as np


def gemm(k: int, m: int, n: int, random_seed: int):
    rng = np.random.default_rng(random_seed)

    input_name, output_name = "input_name", "output_name"
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

    onnx_model = onnx.helper.make_model(graph_def, producer_name="tinyquant-test")
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

    input_name, output_name = "input_name", "output_name"
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

    onnx_model = onnx.helper.make_model(graph_def, producer_name="tinyquant-test")
    onnx_model.opset_import[0].version = 13

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)

    return onnx_model
