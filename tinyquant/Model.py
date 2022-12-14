"""
Represent a neural network imported from ONNX and implement inference.
"""


from typing import Union, List
import numpy as np
import onnx
import onnx.mapping
import onnx.numpy_helper

from tinyquant.Tensor import Tensor, FTensor


class Value:
    pass


class Constant(Value):
    def __init__(self, name: str, outputs: List['Node'], data: Tensor = None):
        self.name = name
        self.outputs = outputs
        self.data = data

    def __repr__(self):
        return f"Constant({self.name})"


class Variable(Value):
    def __init__(self, name: str, inputs: List['Node'], outputs: List['Node'], data: Tensor = None):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.data = data

    def __repr__(self):
        return f"Variable({self.name})"


class Node:
    def __init__(self,
                 name: str, op: str,
                 inputs: List[Value],
                 outputs: List[Value]):
        self.name = name
        self.op = op
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        return f"Node({self.name})"


class Model:
    def __init__(self, onnx_model: onnx.ModelProto):
        graph = onnx_model.graph

        value_dict: dict[str, Union[Value]] = {}
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

        graph_outputs: List[Value] = []
        for onnx_tensor in graph.output:
            graph_outputs.append(value_dict[onnx_tensor.name])

        self.input_vars = inputs
        self.output_vars = graph_outputs
        self.values = list(value_dict.values())
        self.nodes = list(nodes.values())

    def __call__(self, inputs: List[Tensor]):

        # Set input values
        for tensor, variable in zip(inputs, self.input_vars):
            variable.data = tensor

        # Iterate through nodes updating all variables in the model.
        for node in self.nodes:
            if node.op == 'Relu':
                x = node.inputs[0].data
                y = x.relu()
                node.outputs[0].data = y
            elif node.op == 'Sigmoid':
                x = node.inputs[0].data
                y = (1.0 + (-x).exp()).inv()
                node.outputs[0].data = y
            elif node.op == 'Gemm':
                x = node.inputs[0].data
                w = node.inputs[1].data
                b = node.inputs[2].data
                y = w.dot(x.T).T + b.reshape(tuple([1]*(len(w.shape)-1) + [b.shape[0]]))
                node.outputs[0].data = y
            else:
                raise ValueError(f"ONNX operand {node.op} not supported.")

        output_tensors: List[Tensor] = []
        for out_var in self.output_vars:
            output_tensors.append(out_var.data)

        return output_tensors
