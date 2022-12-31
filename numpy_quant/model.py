"""
Represent a neural network imported from ONNX and implement inference.
"""
from collections import OrderedDict
from copy import copy
from time import time
from typing import List, Any, Union
import numpy as np
import onnx
import onnx.mapping
import onnx.numpy_helper

from numpy_quant.tensor import Tensor, FTensor, quantize_tensor, quantize_tensor_min_max, fconv2d, ITensor, concat, where, \
    QTensor
from numpy_quant.numpy_quantization import quant_parameters


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
    value = onnx.helper.get_attribute_value(x)
    if isinstance(value, onnx.TensorProto):
        return onnx.numpy_helper.to_array(value)
    else:
        return value


def onnx_operator_implementation(op: str, inputs: list[Tensor], attrs: dict[str, object]) -> list[Tensor]:
    if op == 'Add':
        a = inputs[0]
        b = inputs[1]
        y = a + b
        return [y]
    elif op == 'Concat':
        x_list = [x for x in inputs]
        axis = attrs['axis']
        return [concat(x_list, axis=axis)]
    elif op == 'Constant':
        value = attrs['value']
        if value.dtype == np.float32:
            return [FTensor(value)]
        elif value.dtype == np.int64:
            return [ITensor(value)]
        else:
            cls = value.dtype.__class__
            raise ValueError(f"Constant value type {cls.__module__}.{cls.__qualname__} not supported.")
    elif op == 'ConstantOfShape':
        shape = inputs[0]
        value = attrs['value']
        y_data = np.full(tuple(shape.data), fill_value=value, dtype=value.dtype)
        if value.dtype == np.float32:
            return [FTensor(y_data)]
        elif value.dtype == np.int64:
            return [ITensor(y_data)]
        else:
            cls = value.dtype.__class__
            raise ValueError(f"Constant value type {cls.__module__}.{cls.__qualname__} not supported.")
    elif op == 'Conv':
        x = inputs[0]
        w = inputs[1]
        b = inputs[2]
        y = fconv2d(x, w, b, tuple(attrs['pads']), tuple(attrs['strides']))
        return [y]
    elif op == 'Div':
        a = inputs[0]
        b = inputs[1]
        y = a.div(b)
        return [y]
    elif op == 'Equal':
        a = inputs[0]
        b = inputs[1]
        return [a == b]
    elif op == 'Erf':
        x = inputs[0]
        return [x.erf()]
    elif op == 'Expand':
        x = inputs[0]
        shape = inputs[1]
        return [x.expand(shape)]
    elif op == 'Gather':
        x = inputs[0]
        indices = inputs[1]
        y = x.take(indices, axis=attrs['axis'])
        return [y]
    elif op == 'Gemm':
        x = inputs[0]
        w = inputs[1]
        b = inputs[2]
        if 'transA' in attrs and attrs['transA']:
            x = x.T
        if 'transB' in attrs and attrs['transB']:
            w = w.T
        y = x.matmul(w) + b
        return [y]
    elif op in ['Identity']:
        return [inputs[0].copy()]
    elif op == 'LayerNormalization':
        # See formulas: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean
        x = inputs[0]
        scale = inputs[1]
        bias = inputs[2]

        # stage 1
        mean = x.mean(axis=attrs['axis'], keepdims=True)
        d = x + (-mean)
        dd = d * d
        var = dd.mean(axis=attrs['axis'], keepdims=True)
        vareps = var + attrs['epsilon']
        stddev = vareps.sqrt()
        normalized = d * stddev.inv()

        # stage 2
        y = normalized * scale + bias

        return [y]
    elif op in ['MatMul']:
        a = inputs[0]
        b = inputs[1]
        y = a.matmul(b)
        return [y]
    elif op in ['Mul']:
        a = inputs[0]
        b = inputs[1]
        y = a * b
        return [y]
    elif op == 'ReduceMean':
        x = inputs[0]
        return [x.mean(attrs['axis'], keepdims=attrs['keepdims'])]
    elif op == 'Relu':
        x = inputs[0]
        y = x.relu()
        return [y]
    elif op == 'Reshape':
        x = inputs[0]
        shape = inputs[1]
        y = x.reshape(shape)
        return [y]
    elif op == 'Sigmoid':
        x = inputs[0]
        y = x.sigmoid()
        return [y]
    elif op == "Shape":
        x = inputs[0]
        return [x.shape]
    elif op == "Slice":
        x = inputs[0]
        starts = inputs[1].data
        ends = inputs[2].data
        axes = inputs[3].data
        slices = [slice(None, None, None)] * x.shape.size
        for s, e, a in zip(starts, ends, axes):
            slices[a] = slice(s, e)
        return [x.__getitem__(tuple(slices))]
    elif op == 'Softmax':
        x = inputs[0]
        y = x.softmax(axis=attrs['axis'])
        return [y]
    elif op == 'Tanh':
        x = inputs[0]
        return [x.tanh()]
    elif op == 'Transpose':
        x = inputs[0]
        axes = attrs['perm']
        y = x.transpose(axes)
        return [y]
    elif op == 'Unsqueeze':
        x = inputs[0]
        axes = inputs[1]
        return x.expand_dims(axis=axes)
    elif op == 'Where':
        condition = inputs[0]
        x = inputs[1]
        y = inputs[2]
        return [where(condition, x, y)]
    else:
        raise ValueError(f"ONNX operand {op} not supported.")


class Model:
    def __init__(self, nodes: list[Node], values: list[Value], inputs: List[Variable], outputs: List[Variable]):
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

    def __call__(self, inputs: List[np.ndarray], profile=False):

        time_per_op_types = {op: 0.0 for op in {n.op for n in self.nodes}}

        # Set input values
        for array, variable in zip(inputs, self.inputs):
            if array.dtype == np.float32:
                variable.data = FTensor(array)
            elif array.dtype == np.int64:
                variable.data = ITensor(array)
            else:
                raise ValueError(f"Array dtype {array.dtype} not supported")

        # Iterate through nodes updating all variables in the model.
        for node in self.nodes:
            inputs = [i.data for i in node.inputs]

            stime = time()
            outputs = onnx_operator_implementation(node.op, inputs, node.attrs)
            time_per_op_types[node.op] += time() - stime

            for o, tensor in zip(node.outputs, outputs):
                o.data = tensor

        output_tensors: List[np.ndarray] = []
        for out_var in self.outputs:
            output_tensors.append(out_var.data.data)

        profile_results = time_per_op_types
        if profile:
            return output_tensors, profile_results
        else:
            return output_tensors

    def quantize(self, calibration_inputs: list[np.ndarray], bit_width=8):
        self(calibration_inputs)
        node_dict = {node.name: node for node in self.nodes}
        value_dict = {value.name: value for value in self.values}
        value_data_data_dict = {val.name: val.data.data for val in self.values}
        value_min_dict = {name: np.mean(data.reshape((data.shape[0], -1) if data.shape else (-1,)).min())
                          for name, data in value_data_data_dict.items()}
        value_max_dict = {name: np.mean(data.reshape((data.shape[0], -1) if data.shape else (-1,)).max())
                          for name, data in value_data_data_dict.items()}

        def get_quantization_params(value: Value, assymetric: bool):
            scale, zero_point = quant_parameters(value_min_dict[value.name],
                                                 value_max_dict[value.name],
                                                 bit_width=bit_width,
                                                 asymmetric=assymetric)
            return QuantizationParams(scale, zero_point)

        qnodes_dict: OrderedDict[str, Node] = OrderedDict()
        qvalues_dict: dict[str, Value] = {}

        qparams_per_value: dict[str, QuantizationParams] = {}
        for value in self.inputs:
            qvalues_dict[value.name] = value
            scale, zero_point = quant_parameters(value_min_dict[value.name],
                                                 value_max_dict[value.name],
                                                 bit_width=bit_width,
                                                 asymmetric=isinstance(value, Variable))
            qparams_per_value[value.name] = QuantizationParams(scale, zero_point)

        for value in self.values:
            if isinstance(value, Constant):
                scale, zero_point = quant_parameters(value_min_dict[value.name],
                                                     value_max_dict[value.name],
                                                     bit_width=bit_width,
                                                     asymmetric=isinstance(value, Variable))
                qvalues_dict[value.name] = Constant(value.name, [],
                                                    quantize_tensor(value.data, bit_width, scale, zero_point))
                qparams_per_value[value.name] = QuantizationParams(scale, zero_point)

        for node in self.nodes:
            if node.op == "MatMul":
                # Store output quantization params
                qnodes_dict[node.name] = Node(node.name, "MatMul", node.attrs, [], [])
                out_val = node.outputs[0]
                qparams_per_value[out_val.name] = get_quantization_params(node.outputs[0], assymetric=True)
                qvalues_dict[out_val.name] = Variable(out_val.name, [], [], None)
            if node.op == "Gemm":
                # Quantize the two matrix input values of the general matrix multiplication and store quantization
                # parameters to qattributes
                for input_value in node.inputs[:2]:
                    if isinstance(input_value, Variable):
                        qvalues_dict[input_value.name] = Variable(input_value.name, [], [], None)
                        qparams_per_value[input_value.name] = get_quantization_params(
                            input_value, isinstance(input_value, Variable))
                # Quantize bias and store corresponding quantization params to qattributes
                bias = node.inputs[2]
                qparams1 = qparams_per_value[node.inputs[0].name]
                qparams2 = qparams_per_value[node.inputs[1].name]
                bias_scale = qparams1.scale * qparams2.scale
                qparams_per_value[bias.name] = QuantizationParams(bias_scale, None)
                qbias = quantize_tensor(bias.data, 4 * bit_width, bias_scale, None)
                qvalues_dict[bias.name] = Constant(bias.name, [], qbias)
                # Store output quantization params
                qnodes_dict[node.name] = Node(node.name, "Gemm", node.attrs, [], [])
                out_val = node.outputs[0]
                qparams_per_value[out_val.name] = get_quantization_params(node.outputs[0], assymetric=True)
                qvalues_dict[out_val.name] = Variable(out_val.name, [], [], None)
            if node.op == "Add" and (isinstance(node.inputs[0], Constant) or isinstance(node.inputs[1], Constant)):
                if isinstance(node.inputs[0], Constant):
                    bias_ind = 0
                    x_ind = 1
                else:
                    x_ind = 0
                    bias_ind = 1
                x_name = node.inputs[x_ind].name
                bias_name = node.inputs[bias_ind].name
                bias = node.inputs[bias_ind].data
                input_qparams = qparams_per_value[x_name]
                bias_scale = input_qparams.scale
                qbias_data = quantize_tensor(bias.data, 4 * bit_width, bias_scale, None)
                qbias = Constant(bias_name, [], qbias_data)
                qvalues_dict[qbias.name] = qbias
                qparams_per_value[bias_name] = QuantizationParams(bias_scale, None)
                # Store output quantization params
                qnodes_dict[node.name] = Node(node.name, "Add", node.attrs, [], [])
                out_val = node.outputs[0]
                qparams_per_value[out_val.name] = get_quantization_params(node.outputs[0], assymetric=True)
                qvalues_dict[out_val.name] = Variable(out_val.name, [], [], None)
            elif node.op in ['Identity', 'Relu']:
                out_val = node.outputs[0]
                qvalues_dict[out_val.name] = Variable(out_val.name, [], [], None)
                qparams_per_value[out_val.name] = qparams_per_value[node.inputs[0].name]
                qnodes_dict[node.name] = Node(node.name, node.op, node.attrs, [], [])
            else:
                out_val = node.outputs[0]  # TODO only work for node with one output
                qvalues_dict[out_val.name] = Variable(out_val.name, [], [], None)
                qparams_per_value[out_val.name] = get_quantization_params(node.outputs[0], assymetric=True)
                qnodes_dict[node.name] = Node(node.name, node.op, node.attrs, [], [])

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

        return QModel(list(qnodes_dict.values()), list(qvalues_dict.values()), qinputs, qoutputs,
                      bit_width, qparams_per_value)


class QuantizationParams:
    def __init__(self, scale: np.float32, zero_point: Union[np.int64, None]):
        self.scale = scale
        self.zero_point = zero_point


class QModel(Model):
    def __init__(self, nodes: list[Node], values: list[Value], inputs: List[Variable], outputs: List[Variable],
                 bit_width: int, quant_params: dict[str, QuantizationParams]):
        """
        quant_params: value name -> quantization parameter
        """
        super(QModel, self).__init__(nodes, values, inputs, outputs)
        self.bit_width = bit_width
        self.quant_params = quant_params

    def __call__(self, inputs: List[np.ndarray], profile=False):
        # Set input values
        for array, variable in zip(inputs, self.inputs):
            qparams = self.quant_params[variable.name]
            if array.dtype == np.float32:
                variable.data = quantize_tensor(FTensor(array), self.bit_width, qparams.scale, qparams.zero_point)
            elif array.dtype == np.int64:
                variable.data = ITensor(array)
            else:
                raise ValueError(f"Array dtype {array.dtype} not supported")

        time_per_op_types = {op: 0.0 for op in {n.op for n in self.nodes}}
        time_per_op_types["TinyqQuant"] = 0.0
        time_per_op_types["TinyqDequant"] = 0.0

        # Iterate through nodes updating all variables in the model.
        for node in self.nodes:
            if node.op == "MatMul":
                inputs_data = []
                for i in node.inputs:
                    if isinstance(i.data, FTensor):
                        qparams = self.quant_params[i.name]

                        stime = time()
                        req_input = quantize_tensor(i.data, self.bit_width, qparams.scale, qparams.zero_point)
                        time_per_op_types["TinyqQuant"] += time() - stime

                        inputs_data.append(req_input)
                    else:
                        inputs_data.append(i.data)
            elif node.op == "Gemm":
                inputs_data = []
                for i in node.inputs:
                    if isinstance(i.data, FTensor):
                        qparams = self.quant_params[i.name]
                        stime = time()
                        req_input = quantize_tensor(i.data, self.bit_width, qparams.scale, qparams.zero_point)
                        time_per_op_types["TinyqQuant"] += time() - stime

                        inputs_data.append(req_input)
                    else:
                        inputs_data.append(i.data)
            else:
                inputs_data = []
                for i in node.inputs:
                    if isinstance(i.data, QTensor):
                        stime = time()
                        deq_input = i.data.dequantize()
                        time_per_op_types["TinyqDequant"] += time() - stime

                        inputs_data.append(deq_input)
                    else:
                        inputs_data.append(i.data)

            stime = time()
            outputs_data = onnx_operator_implementation(node.op, inputs_data, node.attrs)
            time_per_op_types[node.op] += time() - stime

            for o, tensor in zip(node.outputs, outputs_data):
                if node.op == "Gemm":
                    qparams = self.quant_params[node.outputs[0].name]
                    requant_data = tensor.requantize(self.bit_width, qparams.scale, qparams.zero_point)
                    o.data = requant_data
                else:
                    o.data = tensor

        output_tensors: List[Tensor] = []
        for out_var in self.outputs:
            if isinstance(out_var.data, FTensor):
                output_tensors.append(out_var.data.data)
            elif isinstance(out_var.data, QTensor):
                output_tensors.append(out_var.data.dequantize().data)
            else:
                raise ValueError

        profile_results = time_per_op_types
        if profile:
            return output_tensors, profile_results
        else:
            return output_tensors
