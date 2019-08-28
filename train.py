import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Dense, CuDNNLSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback, Callback

from tensorboard import default
from tensorboard import program
import logging

import datetime
import numpy as np
import random

import json
import pickle
from os import mkdir, system, listdir
from os.path import exists, join
from dataprocessing import DataProcessor

import argparse

from distutils.dir_util import copy_tree
from utils import mkdir_safely, get_datetime_str

def max(array):
    curr_index = -1
    curr_value = -np.inf
    for i in range(len(array)):
        if array[i] > curr_value:
            curr_index = i
            curr_value = array[i]
    return curr_index, curr_value

class StateResetterCallback(Callback):

    def __init__(self, dataprocessor, model):
        super(StateResetterCallback, self).__init__()

        self.dataprocessor = dataprocessor
        self.model = model

    def on_batch_end(self, batch, logs={}):
        if self.dataprocessor.next_batch_is_new_song:
            self.model.reset_states()

class MuRNN:
    
    def __init__(self):

        self.dp = None
        self.data_path = None
        
        self.model = None
        self.model_path = None

        self.SEQUENCE_LENGTH = -1
        self.TIMESIGNATURE = get_datetime_str()
        self.STATEFUL = None

        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        
    
    def new_model(self, data_dir, sequence_length=50, stateful=False, target_dir="./models/"):

        mkdir_safely(target_dir)
        
        self.data_path = data_dir

        self.SEQUENCE_LENGTH = sequence_length
        self.STATEFUL = stateful

        # make new model directory
        self.model_path = target_dir + "model-" + self.TIMESIGNATURE + "/"
        mkdir_safely(self.model_path)
        mkdir_safely(join(self.model_path, "songs/"))
        
        self.dp = DataProcessor(self.data_path)
        
        # make model
        if stateful:
            data_input = Input(batch_shape=(1, None, 5), name="input")
        else:
            data_input = Input(batch_shape=(None, None, 5), name="input")

        x = CuDNNLSTM(500, return_sequences=False, stateful=stateful)(data_input)
        x = Dropout(0.3)(x)

        x = Dense(700, activation="relu")(x)
        x = Dropout(0.3)(x)

        x = Dense(700, activation="relu")(x)
        x = Dropout(0.3)(x)

        note_picker = Dense(len(self.dp.note_vocab), activation="softmax", name="note_output")(x)
        duration_picker = Dense(len(self.dp.duration_vocab), activation="softmax", name="duration_output")(x)
        offset_picker = Dense(len(self.dp.offset_vocab), activation="softmax", name="offset_output")(x)
        volume_picker = Dense(1, activation="sigmoid", name="volume_output")(x)
        tempo_picker = Dense(1, activation="sigmoid", name="tempo_output")(x)

        self.model = Model(inputs=[data_input], outputs=[note_picker, duration_picker, offset_picker, volume_picker, tempo_picker])
        self.compile()

    def train(self, steps_per_epoch, epochs, save_every_epoch=False, limit=DataProcessor.default_limit):
        # save everything that can be saved before training
        with open(join(self.model_path, "model.json"), "w") as file:
            file.write(self.model.to_json())
        with open(join(self.model_path, "variables.json"), "w") as file:
            file.write('{ "SEQUENCE_LENGTH" : ' + str(self.SEQUENCE_LENGTH) + ',\n' +
                       '"TIMESIGNATURE" : "' + self.TIMESIGNATURE + '",\n' +
                       '"STATEFUL" : "' + str(self.STATEFUL) + '" }')
        
        with open(join(self.model_path, "dataprocessor.pkl"), "wb") as file:
            pickle.dump(self.dp, file)

        # training
        callbacks = []

        # Tensorboard
        callbacks.append(TensorBoard(log_dir=self.model_path + "logs/", write_grads=True, write_images=True))
        #EarlyStopping
        callbacks.append(EarlyStopping(monitor="loss", min_delta=0.0002, patience=5))

        if self.STATEFUL:
            # StateResetterCallback
            #callbacks.append(StateResetterCallback(self.dp, self.model))
            pass
        
        if save_every_epoch:
            # ModelCheckpoint
            callbacks.append(ModelCheckpoint(self.model_path + "weights-{epoch:04d}.hdf5", save_weights_only=True))
        
        self.model.fit_generator(self.dp.train_generator_with_padding(self.SEQUENCE_LENGTH, (1 if self.STATEFUL else limit)), 
                                steps_per_epoch=steps_per_epoch, 
                                epochs=epochs, 
                                verbose=1, 
                                callbacks=callbacks)
        
        self.model.save_weights(self.model_path + "weights.hdf5")


    def load_model(self, model_dir_path, weights_filename="weights.hdf5"):

        self.model_path = model_dir_path

        with open(join(self.model_path, "model.json"), "r") as model_json:
            self.model = model_from_json(model_json.read())

        with open(join(self.model_path, "variables.json"), "r") as variable_json:
            variables = json.load(variable_json)
            self.SEQUENCE_LENGTH = int(variables["SEQUENCE_LENGTH"])

            if "TIMESIGNATURE" in variables.keys():
                self.TIMESIGNATURE = variables["TIMESIGNATURE"]
            else:
                # for backwards-compatibility
                self.TIMESIGNATURE = model_dir_name.replace("model-","")
            
            if "STATEFUL" in variables.keys():
                self.STATEFUL = bool(variables["STATEFUL"])
            else:
                self.STATEFUL = False
            
            with open(join(self.model_path, "dataprocessor.pkl"), "rb") as file:
                self.dp = pickle.load(file)

        self.model.load_weights(join(self.model_path, weights_filename))

        self.compile()
    
    def make_song(self, length):
        song = []
        sequence = np.ones((1, self.SEQUENCE_LENGTH, 5))

        random_note = random.randrange(0, len(self.dp.note_vocab))
        sequence[0][-1][0] =  float(random_note) / float(len(self.dp.note_vocab))

        random_duration = random.randrange(0, len(self.dp.duration_vocab))
        sequence[0][-1][1] =  float(random_duration) / float(len(self.dp.duration_vocab))

        random_offset = random.randrange(0, len(self.dp.offset_vocab))
        sequence[0][-1][2] =  float(random_offset) / float(len(self.dp.offset_vocab))

        random_volume = random.random()
        sequence[0][-1][3] = random_volume
        
        random_tempo = random.random()
        sequence[0][-1][4] = random_tempo

        song.append((self.dp.num_to_note(random_note),
                     self.dp.num_to_duration(random_duration),
                     self.dp.num_to_offset(random_offset),
                     random_volume,
                     random_tempo))

        for i in range(length-1):
            note_prediction, duration_prediction, offset_prediction, volume_prediction, tempo_prediction = self.model.predict(sequence)

            note_index = max(note_prediction[0])[0]
            duration_index = max(duration_prediction[0])[0]
            offset_index = max(offset_prediction[0])[0]

            volume_prediction = volume_prediction[0][0]
            tempo_prediction = tempo_prediction[0][0]
            
            song.append((self.dp.num_to_note(note_index),
                         self.dp.num_to_duration(duration_index),
                         self.dp.num_to_offset(offset_index),
                         volume_prediction,
                         tempo_prediction))

            sequence = np.roll(sequence, -sequence.shape[2])

            sequence[0][-1][0] = note_index / len(self.dp.note_vocab)
            sequence[0][-1][1] = duration_index / len(self.dp.duration_vocab)
            sequence[0][-1][2] = offset_index / len(self.dp.offset_vocab)
            sequence[0][-1][3] = volume_prediction
            sequence[0][-1][4] = tempo_prediction

        return song
        
    
    def compile(self):
        #optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.95)
        self.model.compile(
            loss={"note_output" : "categorical_crossentropy",
                  "duration_output" : "categorical_crossentropy",
                  "offset_output" : "categorical_crossentropy",
                  "volume_output" : "mean_absolute_error",
                  "tempo_output" : "mean_absolute_error"},
            loss_weights=self.get_lossweights(),
            optimizer="adam",
            metrics=["accuracy"])
    
    def get_lossweights(self, smoothing=0.4):

        output_sizes = [len(self.dp.note_vocab),
                        len(self.dp.duration_vocab),
                        len(self.dp.offset_vocab),
                        1,
                        1]
        output_weights = [0.4, 0.15, 0.15, 0.15, 0.15]
        

        weighted_average = float(sum([tup[0]*tup[1] for tup in zip(output_sizes, output_weights)])) / float(sum(output_weights))

        output_names = ["note_output", "duration_output", "offset_output", "volume_output", "tempo_output"]

        return dict(zip(output_names, [((weighted_average / float(size))*(1-smoothing)) + smoothing for size in output_sizes]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="MuRNN")

    parser.add_argument("dir",
                        type=str,
                        help="Set the path to the dataset on which the model will be trained")
    parser.add_argument("-steps_per_epoch",
                        type=int, 
                        default=400,
                        help="Specify the number of steps in each epoch of training,\ndefaults to 400")
    parser.add_argument("-epochs",
                        type=int,
                        default=10,
                        help="Specify the number of training epochs,\ndefaults to 10")
    parser.add_argument("-seq_len",
                        type=int,
                        default=50,
                        help="Specify the sequence length,\ndefaults to 50")
    parser.add_argument("-limit",
                        type=int,
                        default=DataProcessor.default_limit,
                        help="Set a batchsize limit as to not exceed memory capabilities of the GPU,\ndefaults to " + str(DataProcessor.default_limit))
    parser.add_argument("-target_dir",
                        type=str,
                        default="./models/",
                        help="Specify a target directory in which your model-directory will be saved,\ndefaults to './models/'")
    parser.add_argument("--stateful",
                        action="store_true",
                        help="Use this flag to indicate the network is stateful\nNote that batch size will automatically be 1")
    parser.add_argument("--save_every_epoch",
                        action="store_true",
                        help="Use this flag to save the weights of the model for each epoch")
    
    args = parser.parse_args()

    model = MuRNN()

    model.new_model(args.dir, sequence_length=args.seq_len, stateful=args.stateful, target_dir=args.target_dir)
        
    model.train(args.steps_per_epoch, args.epochs, save_every_epoch=args.steps_per_epoch, limit=args.limit)

"""
    TEMP DISCLAIMER:
    The music dataset was downloaded from http://www.piano-midi.de/ and is licensed under
    the cc-by-sa Germany License (https://creativecommons.org/licenses/by-sa/3.0/de/deed.en)
"""