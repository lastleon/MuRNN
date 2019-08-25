import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Dense, CuDNNLSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from tensorboard import default
from tensorboard import program
import logging

import datetime
import numpy as np
import random

import json
import pickle
from os import mkdir, system, listdir
from os.path import exists
from dataprocessing import DataProcessor

import argparse

from distutils.dir_util import copy_tree

### helper functions

def get_datetime_str():
    return '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())

def mkdir_safely(path):
    if not exists(path):
        mkdir(path)

def max(array):
    curr_index = -1
    curr_value = -np.inf
    for i in range(len(array)):
        if array[i] > curr_value:
            curr_index = i
            curr_value = array[i]
    return curr_index, curr_value


class MuRNN:
    
    def __init__(self, data_dir):
        
        mkdir_safely("./models/")

        self.dp = None
        self.data_path = data_dir
        
        self.model = None
        self.model_path = None
        self.SEQUENCE_LENGTH = 70

        self.timesignature = get_datetime_str()
        
    
    def new_model(self):
        # make new model directory
        self.model_path = "./models/model-" + self.timesignature + "/"
        mkdir_safely(self.model_path)

        self.dp = DataProcessor(self.data_path)

        # make model
        data_input = Input(batch_shape=(None, None, 3), name="input")

        x = CuDNNLSTM(1500, return_sequences=False)(data_input)
        x = Dropout(0.2)(x)

        note_picker = Dense(len(self.dp.note_vocab), activation="softmax", name="note_output")(x)
        duration_picker = Dense(len(self.dp.duration_vocab), activation="softmax", name="duration_output")(x)
        offset_picker = Dense(len(self.dp.offset_vocab), activation="softmax", name="offset_output")(x)

        self.model = Model(inputs=[data_input], outputs=[note_picker, duration_picker, offset_picker])
        self.compile()

    def train(self, steps_per_epoch, epochs, save_every_epoch=False, run_tensorboard_server=False):

        with open(self.model_path + "model.json", "w") as file:
            file.write(self.model.to_json())
        with open(self.model_path + "variables.json", "w") as file:
            file.write('{ "SEQUENCE_LENGTH" : ' + str(self.SEQUENCE_LENGTH) + ', ' +
                       '"TIMESIGNATURE" : "' + self.timesignature + '" }')
        
        tensorboard = TensorBoard(log_dir=self.model_path + "logs/", write_grads=True, write_images=True)
        early_stopping = EarlyStopping(min_delta=0.0002, patience=5)

        callbacks = [tensorboard, early_stopping]

        if save_every_epoch:
            checkpointer = ModelCheckpoint(self.model_path + "weights-{epoch:04d}.hdf5", save_weights_only=True)
            callbacks.append(checkpointer)
        
        if run_tensorboard_server:
            self.start_tensorboard()

        self.model.fit_generator(self.dp.train_generator_with_padding(self.SEQUENCE_LENGTH), steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, callbacks=callbacks)
        self.model.save_weights(self.model_path + "weights.hdf5")


    def load_model(self, model_dir_name, weights_filename="weights.hdf5"):

        self.model_path = "./models/" + model_dir_name + "/"

        with open(self.model_path + "model.json", "r") as model_json:
            self.model = model_from_json(model_json.read())

        with open(self.model_path + "variables.json", "r") as variable_json:
            variables = json.load(variable_json)
            self.SEQUENCE_LENGTH = int(variables["SEQUENCE_LENGTH"])

            if "TIMESIGNATURE" in variables.keys():
                self.timesignature = variables["TIMESIGNATURE"]
            else:
                # for backwards-compatibility
                self.timesignature = model_dir_name.replace("model-","")
            
            self.dp = DataProcessor(self.data_path)

        self.model.load_weights(self.model_path + weights_filename)

        self.compile()
    
    def make_song(self, length):
        np.set_printoptions(threshold=np.inf)

        song = []
        sequence = np.ones((1, self.SEQUENCE_LENGTH, 3))

        random_note = random.randrange(0, len(self.dp.note_vocab))
        sequence[0][-1][0] =  float(random_note) / float(len(self.dp.note_vocab))

        random_duration = random.randrange(0, len(self.dp.duration_vocab))
        sequence[0][-1][1] =  float(random_duration) / float(len(self.dp.duration_vocab))

        random_offset = random.randrange(0, len(self.dp.offset_vocab))
        sequence[0][-1][2] =  float(random_offset) / float(len(self.dp.offset_vocab))

        song.append((self.dp.num_to_note(random_note), self.dp.num_to_duration(random_duration), self.dp.num_to_offset(random_offset)))

        for i in range(length-1):
            note_prediction, duration_prediction, offset_prediction = self.model.predict(sequence)

            note_index = max(note_prediction[0])[0]
            duration_index = max(duration_prediction[0])[0]
            offset_index = max(offset_prediction[0])[0]
            
            song.append((self.dp.num_to_note(note_index), self.dp.num_to_duration(duration_index), self.dp.num_to_offset(offset_index)))

            sequence = np.roll(sequence, -sequence.shape[2])

            sequence[0][-1][0] = note_index / len(self.dp.note_vocab)
            sequence[0][-1][1] = duration_index / len(self.dp.duration_vocab)
            sequence[0][-1][2] = offset_index / len(self.dp.offset_vocab)

        mkdir_safely(self.model_path + "songs/")
        
        songpath = self.model_path + "songs/song-" + get_datetime_str() + ".mu"

        with open(songpath, "wb") as f:
            pickle.dump(song, f)

        return songpath
        
    
    def compile(self):
        optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.95)

        self.model.compile(loss="categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"])

    def start_tensorboard(self):
        log = logging.getLogger("werkzeug").setLevel(logging.ERROR)
        tb = program.TensorBoard(default.get_plugins())
        tb.configure(argv=[None, "--logdir", self.model_path + "logs/"])
        url = tb.launch()
        print('\nTensorBoard at %s \n' % url)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="MuRNN")

    parser.add_argument("dataset_directory",
                        help="The path to the dataset on which the model should be trained")
    parser.add_argument("steps_per_epoch",
                        type=int, 
                        help="The number of steps in each epoch of training")
    parser.add_argument("epochs",
                        type=int,
                        help="The number of training epochs")
    parser.add_argument("--run_tensorboard",
                        action="store_true",
                        help="Use this flag to run a tensorboard server on port 6006")
    parser.add_argument("--save_every_epoch",
                        action="store_true",
                        help="Use this flag to save the weights of the model for each epoch")
    
    args = parser.parse_args()

    model = MuRNN(args.dataset_directory)

    model.new_model()
        
    model.train(args.steps_per_epoch, args.epochs, save_every_epoch=args.steps_per_epoch, run_tensorboard_server=args.run_tensorboard)

    copy_tree(model.model_path, "../storage/model-" + model.timesignature + "/")

"""
    TEMP DISCLAIMER:
    The music dataset was downloaded from http://www.piano-midi.de/ and is licensed under
    the cc-by-sa Germany License (https://creativecommons.org/licenses/by-sa/3.0/de/deed.en)
"""