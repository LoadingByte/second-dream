import os
from collections import namedtuple, OrderedDict
from os.path import join, splitext

import keras
import numpy as np
import tensorflow as tf

import second_dream
from . import app

MODELS = OrderedDict()
TESTSETS = OrderedDict()


class ModelId(namedtuple("ModelId", "model_type dataset_name testset_name")):
    def __str__(self):
        return self.model_type + "-" + self.dataset_name

    @classmethod
    def create(cls, model_type, dataset_name):
        return cls(model_type, dataset_name, dataset_name + "_TEST")

    @classmethod
    def from_string(cls, string):
        return string if isinstance(string, cls) else cls.create(*string.split("-"))


LoadedModel = namedtuple("Model", "keras_model instrument graph session")


def layer_names(model_id):
    return [l.name for l in MODELS[model_id].keras_model.layers]


def timeser_len(model_id):
    return TESTSETS[model_id.testset_name][0].shape[1]


def max_testset_timeser_idx(model_id):
    return len(TESTSETS[model_id.testset_name][1]) - 1


def true_class(model_id, testset_timeser_idx):
    y_test = TESTSETS[model_id.testset_name][1]
    unique_classes = np.unique(y_test)
    return np.argwhere(unique_classes == y_test[testset_timeser_idx])[0, 0]


def _init():
    global MODELS
    global TESTSETS

    models_dir = app.config["MODELS_DIR"]
    testsets_dir = app.config["TESTSETS_DIR"]

    app.logger.info("Loading testsets...")
    for filename in os.listdir(testsets_dir):
        testset_name = splitext(filename)[0]
        app.logger.info(f"Loading testset {testset_name}...")

        testset = _load_dataset(join(testsets_dir, filename))
        TESTSETS[testset_name] = testset
    app.logger.info("Finished loading testsets!")

    app.logger.info("Loading models...")
    for model_type in os.listdir(models_dir):
        for filename in os.listdir(join(models_dir, model_type)):
            dataset_name = splitext(filename)[0]
            model_id = ModelId.create(model_type, dataset_name)
            app.logger.info(f"Loading model {model_type} for dataset {dataset_name}...")

            MODELS[model_id] = None
            graph = tf.Graph()
            with graph.as_default():
                session = tf.Session()
                with session.as_default():
                    keras_model = keras.models.load_model(join(models_dir, model_type, filename))
                    instrument = second_dream.instrument_model(
                        keras_model,
                        backend_kwargs={"options": tf.RunOptions(timeout_in_ms=app.config["DREAM_TIMEOUT"])})
                    MODELS[model_id] = LoadedModel(keras_model, instrument, graph, session)
    app.logger.info("Finished loading models!")


def _load_dataset(file):
    data = np.loadtxt(file, delimiter="\t")
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    return X, y


_init()
