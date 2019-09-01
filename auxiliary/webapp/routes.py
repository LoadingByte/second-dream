from bokeh.resources import INLINE
from flask import render_template, request, Response
from markupsafe import Markup

from second_dream import DEFAULT_HYPERPARAMS
from . import app
from .data import true_class
from .dreamer import Overloaded, visualize_model, dream_and_visualize
from .forms import MainForm


@app.route("/")
@app.route("/index")
def index():
    form = MainForm(request.args)

    context = {}

    if form.model.validate(None):
        model_id = form.model.model.data
        model_svg = visualize_model(model_id)
        context["model_svg"] = Markup(model_svg)

    if form.validate() and form.dream.dream.data:
        model_id = form.model.model.data
        hyperparams = {field.short_name: field.data for field in form.dream
                       if field.short_name in DEFAULT_HYPERPARAMS.keys()}

        try:
            dream_script, dream_div, dream_video, base_prediction, final_dream_prediction = dream_and_visualize(
                model_id,
                form.base_timeser.source.data, form.base_timeser.testset_timeser_idx.data,
                form.base_timeser.preset.data, form.base_timeser.custom.data,
                hyperparams, form.dream.loss_layers.data,
                form.dream.plot_type.data)

            if dream_script is not None:
                context["dream_script"] = Markup(dream_script)
                context["dream_div"] = Markup(dream_div)
            elif dream_video is not None:
                context["dream_video"] = Markup(dream_video)
            context["base_prediction"] = base_prediction
            context["final_dream_prediction"] = final_dream_prediction

            if form.base_timeser.source.data == "testset_timeser_idx":
                context["true_class"] = true_class(model_id, form.base_timeser.testset_timeser_idx.data)
        except Overloaded as e:
            context["overloaded"] = e.fault

    return render_template("index.html", form=form, **context)


BOKEH_JS = "\n".join(INLINE.js_raw)


@app.route("/bokeh.js")
def bokeh_js():
    return Response(BOKEH_JS, mimetype="text/javascript")
