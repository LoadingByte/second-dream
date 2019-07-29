from concurrent.futures.thread import ThreadPoolExecutor
from io import StringIO

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import second_dream
from . import app
from .data import MODELS, TESTSETS

BASE_TIMESER_PRESETS = {
    "zero": "Zero",
    "global_mean": "Global Mean",
    "local_mean": "Local Mean",
    "global_noise": "Global Noise",
    "local_noise": "Local Noise"
}

_DREAMER_THREAD = ThreadPoolExecutor(max_workers=app.config["DREAM_THREADS"], thread_name_prefix="Dreamer")


class Overloaded(Exception):

    def __init__(self, fault):
        self.fault = fault


def visualize_model(model_id):
    model = MODELS[model_id]
    dot = keras.utils.vis_utils.model_to_dot(model.keras_model, show_shapes=True, show_layer_names=True, rankdir="TB")
    svg = dot.create_svg(encoding="utf-8").decode("utf-8")

    return _postprocess_svg(svg)


def dream_and_visualize(model_id,
                        base_source, base_testset_timeser_idx, base_preset, base_custom,
                        hyperparams, loss_layers):
    # Limit the number of waiting users.
    if _DREAMER_THREAD._work_queue.qsize() >= app.config["MAX_CONCURRENT_REQUESTS"] - 1:
        raise Overloaded("no_capacity")

    base_u_timeser = _get_base_u_timeser(model_id, base_source, base_testset_timeser_idx, base_preset, base_custom)

    # Delegate the actual neural network work to the designated dreamer thread.
    future = _DREAMER_THREAD.submit(_do_dream_and_predict, model_id, base_u_timeser, hyperparams, loss_layers)
    dream_u_timeser, base_prediction, dream_prediction = future.result()

    # Plot
    plt.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(base_u_timeser, label="Base")
    ax.plot(dream_u_timeser, label="Dream")
    ax.legend()
    fig.tight_layout()

    svg_io = StringIO()
    fig.savefig(svg_io, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = svg_io.getvalue()

    svg = svg.replace("white-space:pre;", "")  # fix whole website from breaking apart
    svg = _postprocess_svg(svg)

    return svg, base_prediction, dream_prediction


def _do_dream_and_predict(model_id, base_u_timeser, hyperparams, loss_layers):
    def predict(u_timeser):
        return model.keras_model.predict(np.reshape(u_timeser, (1, -1, 1)))[0]

    model = MODELS[model_id]
    # First restore the environment the model was trained in; without this, we get errors!
    with model.graph.as_default():
        with model.session.as_default():
            # Do the deep dreaming. Timeout if it takes too long.
            try:
                dream_u_timeser = second_dream.dream(model.keras_model, loss_layers, base_u_timeser, **hyperparams,
                                                     fetch_loss_and_grads=model.instrument)
            except tf.errors.DeadlineExceededError:
                raise Overloaded("timeout")

            # To finish of, quickly let the model predict the base and dream time series
            base_prediction = predict(base_u_timeser)
            dream_prediction = predict(dream_u_timeser)

            # Due to some lazyness in Keras, only now, after one execution of every model function (instrumentation and
            # predict), the graph has been fully built. So now, we can finally finalize the graph.
            # This is of course optional and the webapp would function without this. However, it catches memory leaks.
            if not model.graph.finalized:
                model.graph.finalize()

            return dream_u_timeser, base_prediction, dream_prediction


def _get_base_u_timeser(model_id, source, testset_timeser_idx, preset, custom):
    X_test = TESTSETS[model_id.testset_name][0]
    base_len = X_test.shape[1]

    if source == "testset_timeser_idx":
        return TESTSETS[model_id.testset_name][0][testset_timeser_idx, :]
    elif source == "preset":
        if preset == "zero":
            return np.zeros(base_len)
        elif preset == "global_mean":
            return np.repeat(np.average(X_test), base_len)
        elif preset == "local_mean":
            return np.mean(X_test, axis=0)
        elif preset == "global_noise":
            return np.random.normal(loc=np.average(X_test), scale=np.std(np.array(X_test)), size=base_len)
        elif preset == "local_noise":
            return np.array([np.random.normal(loc=m, scale=s) for m, s
                             in zip(np.mean(X_test, axis=0), np.std(X_test, axis=0))])
    elif source == "custom":
        return custom


def _postprocess_svg(svg):
    # Remove XML headers
    return svg[svg.find("<svg"):]
