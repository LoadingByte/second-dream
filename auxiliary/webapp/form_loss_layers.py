from numbers import Number
from typing import Mapping

import numpy as np
import regex

_FLOAT_REGEX = r"[\-+]?\d+(?:.[\d]+)?"
_LINE_REGEX = regex.compile(
    rf"^\s*([^\s]+)(?:\s+({_FLOAT_REGEX})|(?:\s+#(\d+(?:,\d+)*)[:=\s]+({_FLOAT_REGEX}))+)\s*$")


def format_loss_layers(loss_layers):
    def _weight_to_string(weight):
        if isinstance(weight, Number):
            return str(weight)
        else:
            return " ".join("#" + ",".join(map(str, n)) + f":{w}" for n, w in weight.items())

    return "\n".join(layer_name + " " + _weight_to_string(weight) for layer_name, weight in loss_layers.items())


def parse_loss_layers(string):
    def _parse_num(num_str):
        try:
            return int(num_str)
        except ValueError:
            return float(num_str)

    loss_layers = {}
    errors = []

    for line_no, line in enumerate(string.splitlines()):
        match = _LINE_REGEX.match(line)
        if match is None:
            errors.append(f"Line {line_no + 1}: Invalid syntax")
            continue

        layer_name = match[1]
        if match[2] is not None:
            weight = _parse_num(match[2])
            loss_layers[layer_name] = weight
        else:
            neuron_weights = {tuple(int(dim) for dim in n.split(",")): _parse_num(w)
                              for n, w in zip(match.captures(3), match.captures(4))}
            loss_layers[layer_name] = neuron_weights

    return loss_layers, errors


def validate_loss_layers(loss_layers, model):
    errors = []

    layer_dict = {layer.name: layer for layer in model.keras_model.layers}
    for line_no, (layer_name, user_weights) in enumerate(loss_layers.items()):
        if layer_name not in layer_dict.keys():
            errors.append(f"Line {line_no + 1}: Unknown layer {layer_name}")
        elif isinstance(user_weights, Mapping):
            layer_shape = np.array([comp for comp in layer_dict[layer_name].output_shape if comp is not None])

            for neuron in user_weights.keys():
                if len(layer_shape) != len(neuron):
                    errors.append(f"Line {line_no + 1}: Layer {layer_name} has dimension {len(layer_shape)}, "
                                  f"you provided for {len(neuron)}")
                elif any(neuron >= layer_shape):
                    errors.append(f"Line {line_no + 1}: Maximum neuron is " +
                                  ",".join(map(str, layer_shape - 1)) + ", you provided " +
                                  ",".join(map(str, neuron)))

    return errors
