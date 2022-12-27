from numpy_quant.model import Model


def summarize(model: Model):
    """Print summary of given model."""

    node_names = [node.name for node in model.nodes]
    node_inputs = [node.inputs for node in model.nodes]
    node_outputs = [node.outputs for node in model.nodes]

    n_column_0 = max(len(n) for n in node_names)
    n_column_1 = max(max((len(t.name) for t in i), default=0) for i in node_inputs)
    n_column_2 = max(max((len(t.name) for t in i), default=0) for i in node_outputs)

    n_column_0 = max(n_column_0, len("Node"))
    n_column_1 = max(n_column_1, len("Inputs"))
    n_column_2 = max(n_column_2, len("Outputs"))

    ret_str = "="*n_column_0 + "=+=" + "="*n_column_1 + "=+=" + "="*n_column_2 + "\n"
    ret_str += f"{'Node':<{n_column_0}} | {'Inputs':<{n_column_1}} | {'Outputs':<{n_column_2}}\n"
    ret_str += "="*n_column_0 + "=+=" + "="*n_column_1 + "=+=" + "="*n_column_2 + "\n"

    horizontal_sep = "-"*n_column_0 + "-+-" + "-"*n_column_1 + "-+-" + "-"*n_column_2 + "\n"
    for node in model.nodes:
        inp_ind, out_ind = 0, 0
        ret_str += f"{node.name:<{n_column_0}}"
        while True:
            if not (inp_ind == 0 and out_ind == 0):
                ret_str += " "*n_column_0
            if inp_ind < len(node.inputs):
                ret_str += f" | {node.inputs[inp_ind].name:<{n_column_1}}"
                inp_ind += 1
            else:
                ret_str += " | " + " "*n_column_1
            if out_ind < len(node.outputs):
                ret_str += f" | {node.outputs[out_ind].name:<{n_column_2}}"
                out_ind += 1
            else:
                ret_str += " | " + " "*n_column_2
            ret_str += "\n"
            if not (inp_ind < len(node.inputs) or out_ind < len(node.outputs)):
                break
        ret_str += horizontal_sep
    return ret_str
