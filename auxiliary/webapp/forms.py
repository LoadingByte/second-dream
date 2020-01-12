from itertools import takewhile

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
from .dreamer import BASE_TIMESER_PRESETS, PLOT_TYPES
from .form_loss_layers import format_loss_layers, parse_loss_layers, validate_loss_layers


def _first(coll):
    return next(iter(coll))


# === Model form ===


# By default, use the first model.
_DEFAULT_MODEL_ID = _first(MODELS)
if "DEFAULT_MODEL" in app.config:
    try:
        _DEFAULT_MODEL_ID = next(m for m in MODELS
                                 if (m.model_type, m.dataset_name) == app.config["DEFAULT_MODEL"])
    except StopIteration:
        raise ValueError(f"DEFAULT_MODEL {app.config['DEFAULT_MODEL']} from config file "
                         "could not be found in models directory")


class ModelForm(FlaskForm):
    model = SelectField("Model",
                        choices=[(id, f"{id.model_type} on {id.dataset_name}") for id in MODELS],
                        default=_DEFAULT_MODEL_ID,
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
    widget = TextArea()

    def _value(self):
        if self.process_errors:
            return self.raw_data[0]
        else:
            return format_loss_layers(self.data)

    def process_formdata(self, valuelist):
        if valuelist:
            self.data, self.process_errors = parse_loss_layers(valuelist[0])
        elif self.data is None:
            self.data = {}


class DreamForm(FlaskForm):
    _model_id = None

    def validate_loss_layers(self, field):
        field.errors += validate_loss_layers(field.data, MODELS[self._model_id])

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

# By default, use the whole last layer of the default model.
_DEFAULT_LOSS_LAYERS = {layer_names(_DEFAULT_MODEL_ID)[-1]: 1}
if "DEFAULT_LOSS_LAYERS" in app.config:
    _DEFAULT_LOSS_LAYERS, errors = parse_loss_layers(app.config["DEFAULT_LOSS_LAYERS"])
    errors += validate_loss_layers(_DEFAULT_LOSS_LAYERS, MODELS[_DEFAULT_MODEL_ID])
    if errors:
        raise ValueError("The DEFAULT_LOSS_LAYERS specified in the config file are invalid:" +
                         "".join("\n * " + error for error in errors))
_DEMO_LAYER = next(iter(_DEFAULT_LOSS_LAYERS))
DreamForm.loss_layers = LossLayersField("Loss layers",
                                        description=Markup(
                                            "Each line adds one layer whose L2 activation we maximize, alongside a "
                                            "relative weight. Because the sum of all squared activations of the "
                                            "neuron in the layer is maximized, few high activations will win over "
                                            "lots of small activations.<br/>"
                                            f"<kbd>{_DEMO_LAYER} 0.5</kbd> maximizes the layer {_DEMO_LAYER} "
                                            "with weight 0.5.<br/>"
                                            f"<kbd>{_DEMO_LAYER} #0:1 #3:1.5</kbd> only maximizes neurons 0 and 3 "
                                            f"from the layer {_DEMO_LAYER} with weights 1 and 1.5, "
                                            "respectively.<br/>"
                                            "Note that depending on the shape of the layer, you may need more "
                                            "coordinates to identify a single neuron. For example, <kbd>#5,8:1.5</kbd> "
                                            "assigns weight 1.5 to neuron 5,8."),
                                        default=_DEFAULT_LOSS_LAYERS)

DreamForm.plot_type = SelectField("Plot type",
                                  choices=list(PLOT_TYPES.items()),
                                  default=_first(PLOT_TYPES.keys()))
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
