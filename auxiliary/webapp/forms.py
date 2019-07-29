from collections import Mapping
from itertools import takewhile
from numbers import Number

import numpy as np
import regex
from flask_wtf import FlaskForm
from markupsafe import Markup
from wtforms import Field, SelectField, IntegerField, FloatField, RadioField, SubmitField, FormField
from wtforms.validators import ValidationError, StopValidation
from wtforms.widgets import TextArea

import second_dream
from . import app
from .data import MODELS, ModelId, layer_names, timeser_len, max_testset_timeser_idx
from .dreamer import BASE_TIMESER_PRESETS


def _first(coll):
    return next(iter(coll))


# === Model form ===


_DEFAULT_MODEL = _first(MODELS.keys())
if "DEFAULT_MODEL" in app.config:
    try:
        _DEFAULT_MODEL = next(m for m in MODELS.keys() if (m.model_type, m.dataset_name) == app.config["DEFAULT_MODEL"])
    except StopIteration:
        app.logger.error("DEFAULT_MODEL %s from config file could not be found in models directory",
                         app.config["DEFAULT_MODEL"])


class ModelForm(FlaskForm):
    model = SelectField("Model",
                        choices=[(id, f"{id.model_type} on {id.dataset_name}") for id in MODELS.keys()],
                        default=_DEFAULT_MODEL,
                        coerce=ModelId.from_string)
    update = SubmitField("Update")


# === Base time series form ===

def _abort_if_not_selected(form, field):
    if form.source.data not in field.name:
        field.errors[:] = []
        raise StopValidation()


class TimeserField(Field):
    _NUMERIC_CHARS = r"\d+\-\.e"
    _REGEX = regex.compile(rf"^(?:[^{_NUMERIC_CHARS}]*([{_NUMERIC_CHARS}]+))*[^{_NUMERIC_CHARS}]*$")

    widget = TextArea()

    def _value(self):
        if self.process_errors:
            return self.raw_data[0]
        else:
            return " ".join(map(str, self.data))

    def process_formdata(self, valuelist):
        if self.data is None:
            self.data = np.empty(0)

        if valuelist:
            # Note that np.array can still fail and raise a ValueError, e.g., when only "e" is input.
            self.data = np.array(TimeserField._REGEX.match(valuelist[0]).captures(1), dtype=float)


class BaseTimeserForm(FlaskForm):
    testset_timeser_idx = IntegerField("Time series from model test set",
                                       default=0, validators=[_abort_if_not_selected])
    preset = SelectField("Preset",
                         choices=list(BASE_TIMESER_PRESETS.items()),
                         default=_first(BASE_TIMESER_PRESETS.keys()),
                         validators=[_abort_if_not_selected])
    custom = TimeserField("Custom", validators=[_abort_if_not_selected])

    _base_timeser_len = None
    _max_timeser_idx = None

    def validate_testset_timeser_idx(self, field):
        if field.data is not None and self._max_timeser_idx is not None:
            if not 0 <= field.data <= self._max_timeser_idx:
                raise ValidationError(f"Must be in range 0\u2026{self._max_timeser_idx}")

    def validate_custom(self, field):
        if self._base_timeser_len is not None:
            if len(field.data) != self._base_timeser_len:
                raise ValidationError(f"Must be of length {self._base_timeser_len}, not {len(field.data)}")

    def tell_model_id(self, model_id):
        self._base_timeser_len = timeser_len(model_id)
        self._max_timeser_idx = max_testset_timeser_idx(model_id)

        self.testset_timeser_idx.label.text = f"Time series from test set (0\u2026{self._max_timeser_idx})"
        self.custom.label.text = f"Custom (length {self._base_timeser_len})"


_base_timeser_options = [(v, v) for v in vars(BaseTimeserForm) if v[0] != "_"]
setattr(BaseTimeserForm, "source",
        RadioField(choices=_base_timeser_options, default=_base_timeser_options[0][0]))


# === Dream form ===

class LossLayersField(Field):
    _FLOAT_REGEX = r"[\-+]?\d+(?:.[\d]+)?"
    _LINE_REGEX = regex.compile(
        rf"^\s*([^\s]+)(?:\s+({_FLOAT_REGEX})|(?:\s+#(\d+(?:,\d+)*)[:=\s]+({_FLOAT_REGEX}))+)\s*$")

    widget = TextArea()

    def _value(self):
        def _weight_to_string(weight):
            if isinstance(weight, Number):
                return str(weight)
            else:
                return " ".join("#" + ",".join(map(str, n)) + f":{w}" for n, w in weight.items())

        if self.process_errors:
            return self.raw_data[0]
        else:
            return "\n".join(layer_name + " " + _weight_to_string(weight) for layer_name, weight in self.data.items())

    def process_formdata(self, valuelist):
        def _parse_num(num_str):
            try:
                return int(num_str)
            except ValueError:
                return float(num_str)

        if valuelist or self.data is None:
            self.data = {}

        if valuelist:
            for line_no, line in enumerate(valuelist[0].splitlines()):
                match = LossLayersField._LINE_REGEX.match(line)
                if match is None:
                    self.process_errors.append(f"Line {line_no + 1}: Invalid syntax")
                    continue

                layer_name = match[1]
                if match[2] is not None:
                    weight = _parse_num(match[2])
                    self.data[layer_name] = weight
                else:
                    neuron_weights = {tuple(int(dim) for dim in n.split(",")): _parse_num(w)
                                      for n, w in zip(match.captures(3), match.captures(4))}
                    self.data[layer_name] = neuron_weights


class DreamForm(FlaskForm):
    pass

    _model_id = None

    def validate_loss_layers(self, field):
        layer_dict = {layer.name: layer for layer in MODELS[self._model_id].keras_model.layers}
        for line_no, (layer_name, user_weights) in enumerate(field.data.items()):
            if layer_name not in layer_dict.keys():
                field.errors.append(f"Line {line_no + 1}: Unknown layer {layer_name}")
            elif isinstance(user_weights, Mapping):
                layer_shape = np.array([comp for comp in layer_dict[layer_name].output_shape if comp is not None])

                for neuron in user_weights.keys():
                    if len(layer_shape) != len(neuron):
                        field.errors.append(f"Line {line_no + 1}: Layer {layer_name} has dimension {len(layer_shape)}, "
                                            f"you provided for {len(neuron)}")
                    elif any(neuron >= layer_shape):
                        field.errors.append(f"Line {line_no + 1}: Maximum neuron is " +
                                            ",".join(map(str, layer_shape - 1)) + ", you provided " +
                                            ",".join(map(str, neuron)))

    def tell_model_id(self, model_id):
        self._model_id = model_id


# Extracts the parameter description for one hyperparam from the dream function docstring.
def _hyperparam_desc(hyperparam):
    def count_indent(string):
        return len(string) - len(string.lstrip())

    doc_lines = second_dream.dream.__doc__.splitlines()
    start_idx = _first(i for i, line in enumerate(doc_lines) if hyperparam + ":" in line)
    start_indent = count_indent(doc_lines[start_idx])
    param_lines = [doc_lines[start_idx]]
    param_lines.extend(takewhile(lambda line: count_indent(line) > start_indent, doc_lines[start_idx + 1:]))
    return regex.sub(r"\s+", " ", " ".join(param_lines).replace(hyperparam + ":", "")).strip()


for param, default in second_dream.DEFAULT_HYPERPARAMS.items():
    if isinstance(default, int):
        field = IntegerField(param, description=_hyperparam_desc(param), default=default)
    else:
        field = FloatField(param, description=_hyperparam_desc(param), default=default)
    setattr(DreamForm, param, field)

_DEFAULT_LAYER = layer_names(_DEFAULT_MODEL)[-1]

DreamForm.loss_layers = LossLayersField("Loss layers",
                                        description=Markup(
                                            "Each line adds one layer whose L2 activation we maximize, alongside a "
                                            "relative weight. Because the sum of all squared activations of the "
                                            "neuron in the layer is maximized, few high activations will win over "
                                            "lots of small activations.<br/>"
                                            f"<kbd>{_DEFAULT_LAYER} 0.5</kbd> maximizes the layer {_DEFAULT_LAYER} "
                                            "with weight 0.5.<br/>"
                                            f"<kbd>{_DEFAULT_LAYER} #0:1 #3:1.5</kbd> only maximizes neurons 0 and 3 "
                                            f"from the layer {_DEFAULT_LAYER} with weights 1 and 1.5, "
                                            "respectively.<br/>"
                                            "Note that depending on the shape of the layer, you may need more "
                                            "coordinates to identify a single neuron. For example, <kbd>#5,8:1.5</kbd> "
                                            "assigns weight 1.5 to neuron 5,8."),
                                        # By default, use the whole last layer of the default (= first) model.
                                        default={_DEFAULT_LAYER: 1})
DreamForm.dream = SubmitField("Start Dreaming!")


# === Main form ===

class MainForm(FlaskForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.model.validate(None):
            model_id = self.model.model.data
            self.base_timeser.tell_model_id(model_id)
            self.dream.tell_model_id(model_id)

    model = FormField(ModelForm, "Model")
    base_timeser = FormField(BaseTimeserForm, "Base time series")
    dream = FormField(DreamForm, "Dream")
