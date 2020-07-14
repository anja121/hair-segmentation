import json
import sys
import os

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from nets.semseg_mobilenet import build_model


def read_config_file(path):

    try:
        with open(path) as config_file:
            conf = json.load(config_file)
            return conf
    except IOError:
        sys.exit()


def get_callbacks(conf, folder_name):
    callbacks = []

    if conf["checkpoint"]:
        checkpoint_dir = "checkpoints/" + folder_name

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_callback = ModelCheckpoint(checkpoint_dir + "/hair-segmenation" +
                                              '-{epoch:03d}-{loss:03f}.h5',
                                              verbose=1,
                                              monitor='loss',
                                              save_best_only=True,
                                              mode='auto')
        callbacks.append(checkpoint_callback)

    if conf["logs"]:
        logdir = "logs/" + folder_name

        tensorboard_callback = TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)

    return callbacks


def get_model(input_shape, merge_type="concat"):
    model = build_model(shape=(input_shape, input_shape, 3), merge_type=merge_type)
    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

    return model
