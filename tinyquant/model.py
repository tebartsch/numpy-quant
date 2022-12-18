"""
Represent a neural network imported from ONNX and implement inference.
"""
from collections import OrderedDict
from copy import copy
from typing import List, Any, Union
import numpy as np
import onnx
import onnx.mapping
import onnx.numpy_helper

from tinyquant.numpy_helper import conv2d
from tinyquant.tensor import Tensor, FTensor, quantize_tensor, quantize_tensor_min_max, fconv2d, ITensor
from tinyquant.quantize import quant_parameters


class Constant:
    def __init__(self, name: str, outputs: List['Node'], data: Tensor = None):
        self.name = name
        self.outputs = outputs
        self.data = data

    def __repr__(self):
        return f"Constant({self.name})"


class Variable:
    def __init__(self, name: str, inputs: List['Node'], outputs: List['Node'], data: Tensor = None):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.data = data

    def __repr__(self):
        return f"Variable({self.name})"


Value = Union[Constant, Variable]


class Node:
    def __init__(self,
                 name: str, op: str,
                 attrs: dict[str, Any],
                 inputs: List[Value],
                 outputs: List[Value]):
        self.name = name
        self.op = op
        self.attrs = attrs
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        return f"Node({self.name})"


def convert_onnx_dtype_to_numpy_dtype(x):
    np_dtype = onnx.helper.tensor_dtype_to_np_dtype(x.type)
    value = onnx.helper.get_attribute_value(x)
    if isinstance(value, onnx.TensorProto):
        return onnx.numpy_helper.to_array(value)
    else:
        return value


class Model:
    def __init__(self, nodes: list[Node], values: list[Value], inputs: List[Value], outputs: List[Value]):
        self.nodes = nodes
        self.values = values
        self.inputs = inputs
        self.outputs = outputs

    @classmethod
    def from_onnx(cls, onnx_model: onnx.ModelProto):
        graph = onnx_model.graph

        value_dict: dict[str, Value] = {}
        for onnx_tensor in graph.initializer:
            arr = np.array(onnx.numpy_helper.to_array(onnx_tensor))
            value_dict[onnx_tensor.name] = Constant(onnx_tensor.name, outputs=[], data=FTensor(arr))

        inputs: List[Value] = []
        for onnx_tensor in graph.input:
            value_dict[onnx_tensor.name] = Variable(onnx_tensor.name, inputs=[], outputs=[])
            inputs.append(value_dict[onnx_tensor.name])

        nodes: dict[str, Node] = {}
        for onnx_node in graph.node:
            node = Node(
                name=onnx_node.name,
                op=onnx_node.op_type,
                attrs={a.name: convert_onnx_dtype_to_numpy_dtype(a) for a in onnx_node.attribute},
                inputs=[value_dict[input_name] for input_name in onnx_node.input],
                outputs=[]  # Filled below
            )

            for input_tensor_name in onnx_node.input:
                if input_tensor_name not in value_dict:
                    value_dict[input_tensor_name] = Variable(name=input_tensor_name, inputs=[], outputs=[node])
                else:
                    value_dict[input_tensor_name].outputs.append(node)
            for output_tensor_name in onnx_node.output:
                if output_tensor_name not in value_dict:
                    value_dict[output_tensor_name] = Variable(name=output_tensor_name, inputs=[node], outputs=[])
                else:
                    value_dict[output_tensor_name].inputs.append(node)

            node.outputs = [value_dict[output_name] for output_name in onnx_node.output]

            nodes[onnx_node.name] = node

        outputs: List[Value] = []
        for onnx_tensor in graph.output:
            outputs.append(value_dict[onnx_tensor.name])

        return cls(list(nodes.values()), list(value_dict.values()), inputs, outputs)

    def __call__(self, inputs: List[Tensor]):

        # Set input values
        for tensor, variable in zip(inputs, self.inputs):
            variable.data = tensor

        # Iterate through nodes updating all variables in the model.
        for node in self.nodes:
            if node.op == 'Add':
                a = node.inputs[0].data
                b = node.inputs[1].data
                y = a + b
                node.outputs[0].data = y
            elif node.op == 'Constant':
                value = node.attrs['value']
                if value.dtype == np.float32:
                    node.outputs[0].data = FTensor(value)
                elif value.dtype == np.int64:
                    node.outputs[0].data = ITensor(value)
                else:
                    cls = value.dtype.__class__
                    raise ValueError(f"Constant value type {cls.__module__}.{cls.__qualname__} not supported.")
            elif node.op == 'Conv':
                x = node.inputs[0].data
                w = node.inputs[1].data
                b = node.inputs[2].data
                y = fconv2d(x, w, b, tuple(node.attrs['pads']), tuple(node.attrs['strides']))
                node.outputs[0].data = y
            elif node.op == 'Div':
                a = node.inputs[0].data
                b = node.inputs[1].data
                y = a.div(b)
                node.outputs[0].data = y
            elif node.op in ['Gemm', 'QGemm']:
                x = node.inputs[0].data
                w = node.inputs[1].data
                b = node.inputs[2].data
                if 'transA' in node.attrs and node.attrs['transA']:
                    x = x.T
                if 'transB' in node.attrs and node.attrs['transB']:
                    w = w.T
                if node.op == 'Gemm':
                    y = x.matmul(w) + b
                if node.op == 'QGemm':
                    y = x.matmul(w) + b
                    y = y.requantize(node.attrs['output_bit_width'], node.attrs['output_scale'],
                                     node.attrs['output_zero_point'])
                node.outputs[0].data = y
            elif node.op in ['MatMul']:
                a = node.inputs[0].data
                b = node.inputs[1].data
                y = a.matmul(b)
                node.outputs[0].data = y
            elif node.op == 'Relu':
                x = node.inputs[0].data
                y = x.relu()
                node.outputs[0].data = y
            elif node.op == 'Reshape':
                x = node.inputs[0].data
                shape = node.inputs[1].data
                y = x.reshape(shape)
                node.outputs[0].data = y
            elif node.op == 'Sigmoid':
                x = node.inputs[0].data
                y = x.sigmoid()
                node.outputs[0].data = y
            elif node.op == 'Softmax':
                x = node.inputs[0].data
                y = x.softmax()
                node.outputs[0].data = y
            elif node.op == 'TinyQuantize':
                x = node.inputs[0].data
                y = quantize_tensor(x, node.attrs['bit_width'], node.attrs['scale'], node.attrs['zero_point'])
                node.outputs[0].data = y
            elif node.op == 'TinyDequantize':
                x = node.inputs[0].data
                if 'zero_point' in node.attrs:
                    y = FTensor((x.data - node.attrs['zero_point']) * node.attrs['scale'])
                else:
                    y = FTensor(x.data * node.attrs['scale'])
                node.outputs[0].data = y
            elif node.op == 'Transpose':
                x = node.inputs[0].data
                axes = node.attrs['perm']
                y = x.transpose(axes)
                node.outputs[0].data = y
            else:
                raise ValueError(f"ONNX operand {node.op} not supported.")

        output_tensors: List[Tensor] = []
        for out_var in self.outputs:
            output_tensors.append(out_var.data)

        return output_tensors

    def quantize_model(self, calibration_inputs: list[Tensor], bit_width=8):
        self(calibration_inputs)
        node_dict = {node.name: node for node in self.nodes}
        value_dict = {value.name: value for value in self.values}
        value_data_data_dict = {val.name: val.data.data for val in self.values}
        value_min_dict = {name: np.mean(data.reshape((data.shape[0], -1)).min())
                          for name, data in value_data_data_dict.items()}
        value_max_dict = {name: np.mean(data.reshape((data.shape[0], -1)).max())
                          for name, data in value_data_data_dict.items()}

        def get_quantization_params(value: Value):
            scale, zero_point = quant_parameters(value_min_dict[value.name],
                                                 value_max_dict[value.name],
                                                 bit_width=bit_width,
                                                 asymmetric=True)
            return {
                "scale": scale,
                "zero_point": zero_point,
                "bit_width": bit_width,
            }

        qnodes_dict: OrderedDict[str, Node] = OrderedDict()
        qvalues_dict: dict[str, Value] = {}

        qparams_per_value: dict[str, (int, int)] = {}
        for value in self.inputs:
            qvalues_dict[value.name] = value
            qparams_per_value[value.name] = quant_parameters(value_min_dict[value.name],
                                                             value_max_dict[value.name],
                                                             bit_width=bit_width,
                                                             asymmetric=True)

        for node in self.nodes:
            qattributes: dict[str, np.ndarray] = {}
            # Copy attributes
            for trans in ('transA', 'transB'):
                if trans in node.attrs:
                    qattributes[trans] = node.attrs[trans]
            if node.op == "Gemm":
                # Quantize the two matrix input values of the general matrix multiplication and store quantization
                # parameters to qattributes
                for key, index in [("mat1", 0), ("mat2", 1)]:
                    input_value = node.inputs[index]
                    if isinstance(input_value, Constant):
                        qdata = quantize_tensor_min_max(input_value.data, bit_width, asymmetric=False)
                        qvalues_dict[input_value.name] = Constant(input_value.name, [], qdata)
                        qparams_per_value[input_value.name] = qdata.scale, qdata.zero_point
                    else:
                        qvalues_dict[input_value.name] = Variable(input_value.name, [], [], None)
                    quant_params = get_quantization_params(input_value)
                    scale, zero_point = quant_params['scale'], quant_params['zero_point']
                    qattributes[f"{key}_scale"] = scale
                    if zero_point is not None:
                        qattributes[f"{key}_zero_point"] = zero_point
                # Quantize bias and store corresponding quantization params to qattributes
                bias = node.inputs[2]
                qparams1 = qparams_per_value[node.inputs[0].name]
                qparams2 = qparams_per_value[node.inputs[1].name]
                bias_scale = qparams1[0] * qparams2[0]
                qparams_per_value[bias.name] = bias_scale, None
                qbias = quantize_tensor(bias.data, 4 * bit_width, bias_scale, None)
                qvalues_dict[bias.name] = Constant(bias.name, [], qbias)
                qattributes[f"bias_scale"] = bias_scale
                qattributes[f"bias_bit_width"] = 4 * bit_width
                # Store output quantization params to qattributes
                quant_params = get_quantization_params(node.outputs[0])
                output_scale, output_zero_point = quant_params['scale'], quant_params['zero_point']
                qattributes[f"output_scale"] = output_scale
                if output_zero_point is not None:
                    qattributes[f"output_zero_point"] = output_zero_point
                qattributes[f"output_bit_width"] = bit_width
                qnodes_dict[node.name] = Node(node.name, "QGemm", qattributes, [], [])
                out_val = node.outputs[0]
                qparams_per_value[out_val.name] = (output_scale, output_zero_point)
                qvalues_dict[out_val.name] = Variable(out_val.name, [], [], None)
            elif node.op in ["Relu", "Sigmoid"]:
                out_val = node.outputs[0]
                qvalues_dict[out_val.name] = Variable(out_val.name, [], [], None)
                output_scale, output_zero_point = qparams_per_value[node.inputs[0].name]
                qparams_per_value[out_val.name] = (output_scale, output_zero_point)
                qattributes["output_scale"] = output_scale
                qattributes["output_zero_point"] = output_zero_point
                qnodes_dict[node.name] = Node(node.name, node.op, qattributes, [], [])
            else:
                raise ValueError(f"Node {node.op} support in quantization.")

        # Connect nodes to their corresponding input and output values according to the model which is quantized
        for name, qnode in qnodes_dict.items():
            qnode.inputs = [qvalues_dict[i.name] for i in node_dict[name].inputs]
            qnode.outputs = [qvalues_dict[o.name] for o in node_dict[name].outputs]

        # Connect value to their corresponding input and output values according to the model which is quantized
        for name, qvalue in qvalues_dict.items():
            if isinstance(qvalue, Variable):
                qvalue.inputs = [qnodes_dict[i.name] for i in value_dict[name].inputs]
            qvalue.outputs = [qnodes_dict[o.name] for o in value_dict[name].outputs]

        qoutputs = [qvalues_dict[o.name] for o in self.outputs]
        qinputs = [qvalues_dict[i.name] for i in self.inputs]

        def insert_quantization_node(value: Value, nodes: list[Node]):
            attrs = get_quantization_params(input)
            qnode = Node(f"{value.name}/quantize", "TinyQuantize", attrs, [input], [])
            qinput = Variable(f"{qnode.name}_output_0", inputs=[qnode], outputs=copy(input.outputs))
            qnode.outputs = [qinput]
            for o in input.outputs:
                ind = o.inputs.index(input)
                o.inputs[ind] = qinput
            split = min([nodes.index(o) for o in input.outputs])
            input.outputs = [qnode]
            return qinput, nodes[:split] + [qnode] + nodes[split:]

        def insert_dequantization_node(value: Value, nodes: list[Node]):
            attrs = get_quantization_params(output)
            qnode = Node(f"{value.name}/dequantize", "TinyDequantize", attrs, [], [output])
            qoutput = Variable(f"{qnode.name}_input_0", inputs=copy(output.inputs), outputs=[qnode])
            qnode.inputs = [qoutput]
            for i in output.inputs:
                ind = i.outputs.index(output)
                i.outputs[ind] = qoutput
            split = max([nodes.index(i) for i in output.inputs]) + 1
            output.inputs = [qnode]
            return qoutput, nodes[:split] + [qnode] + nodes[split:]

        # Insert input quantization nodes into quantized graph
        for input in qinputs:
            qinput, new_nodes = insert_quantization_node(input, list(qnodes_dict.values()))
            qvalues_dict[qinput.name] = qinput
            qnodes_dict = {n.name: n for n in new_nodes}
        # Insert output dequantization nodes into quantized graph
        for output in qoutputs:
            qoutput, new_nodes = insert_dequantization_node(input, list(qnodes_dict.values()))
            qvalues_dict[qoutput.name] = qoutput
            qnodes_dict = {n.name: n for n in new_nodes}

        return Model(list(qnodes_dict.values()), list(qvalues_dict.values()), qinputs, qoutputs)
