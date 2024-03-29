import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Dense, CuDNNLSTM, Dropout, Bidirectional, LeakyReLU, TimeDistributed
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback

import datetime
import numpy as np
import random

import json
import pickle
from os import mkdir, system, listdir
from os.path import exists, join, basename, dirname
from dataprocessing import DataProcessor

import argparse

from utils import mkdir_safely, get_datetime_str

def max(array):
    curr_index = -1
    curr_value = -np.inf
    for i in range(len(array)):
        if array[i] > curr_value:
            curr_index = i
            curr_value = array[i]
    return curr_index, curr_value

def choose_entry(array, alpha):
    data = []
    for i in range(len(array)):
        data.append((array[i], i))
    
    data.sort(key=lambda tup: tup[0], reverse=True)

    return data[random.randint(0, np.floor(alpha * (len(data)-1)))][1]

class EpochEndCallback(Callback):

    def __init__(self, murrn):
        super(EpochEndCallback, self).__init__()
        self.murrn = murrn

    def on_epoch_end(self, batch, logs={}):
        # save epochs trained
        variables = None
        with open(join(self.murrn.model_path, "variables.json"), "r") as file:
            variables = json.load(file)
        variables["EPOCHS_TRAINED"] = int(variables["EPOCHS_TRAINED"]) + 1
        with open(join(self.murrn.model_path, "variables.json"), "w") as file:
            file.write(json.dumps(variables, file))

        # make sample songs
        for i in range(2):
            DataProcessor.retrieve_midi_from_loaded_data(self.murrn.make_song(), target_dir=self.murrn.model_path, filename="sample-"+str(variables["EPOCHS_TRAINED"])+"-"+str(i))

class MuRNN:
    
    def __init__(self):

        self.dp = None
        self.data_path = None
        
        self.model = None
        self.model_path = None

        self.SEQUENCE_LENGTH = -1
        self.TIMESIGNATURE = get_datetime_str()
        self.EPOCHS_TRAINED = 0
        
    
    def new_model(self, data_dir, sequence_length=50, target_dir="./models/"):

        mkdir_safely(target_dir)
        
        self.data_path = data_dir

        self.SEQUENCE_LENGTH = sequence_length

        # make new model directory
        self.model_path = target_dir + "model-" + self.TIMESIGNATURE + "/"
        mkdir_safely(self.model_path)
        mkdir_safely(join(self.model_path, "songs/"))
        
        # make new dataprocessor
        self.dp = DataProcessor(self.data_path)
        
        # make model
        data_input = Input(batch_shape=(None, None, 6), name="input")

        x = TimeDistributed(Dense(10))(data_input)
        x = LeakyReLU(alpha=0.3)(x)

        x = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(256, return_sequences=False))(x)

        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.3)(x)

        x = Dense(512)(x)
        x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.3)(x)

        # notes
        note_picker = Dense(len(self.dp.note_vocab), activation="softmax", name="note_output")(x)
        
        # durations
        duration_picker = Dense(len(self.dp.duration_vocab), activation="softmax", name="duration_output")(x)

        # offsets
        offset_picker = Dense(len(self.dp.offset_vocab), activation="softmax", name="offset_output")(x)

        # volumes
        volume_picker = Dense(1, activation="sigmoid", name="volume_output")(x)

        # tempos
        tempo_picker = Dense(1, activation="sigmoid", name="tempo_output")(x)

        # belongs to prev chord
        prev_chord_picker = Dense(2, activation="softmax", name="belongs_to_prev_chord_output")(x)

        self.model = Model(inputs=[data_input], outputs=[note_picker, duration_picker, offset_picker, volume_picker, tempo_picker, prev_chord_picker])
        self.compile()

    def train(self, steps_per_epoch, epochs, save_every_epoch=False, limit=DataProcessor.default_limit):
        # save everything that can be saved before training
        with open(join(self.model_path, "model.json"), "w") as file:
            file.write(self.model.to_json())
        with open(join(self.model_path, "variables.json"), "w") as file:
            variables = {
                "SEQUENCE_LENGTH" : self.SEQUENCE_LENGTH,
                "TIMESIGNATURE" : self.TIMESIGNATURE,
                "EPOCHS_TRAINED" : self.EPOCHS_TRAINED,
                "DATASET_NAME" : basename(dirname(self.dp.dir_path))
            }
            file.write(json.dumps(variables))
        
        with open(join(self.model_path, "dataprocessor.pkl"), "wb") as file:
            pickle.dump(self.dp, file)

        #### training
        callbacks = []

        # Tensorboard
        callbacks.append(TensorBoard(log_dir=self.model_path + "logs/", write_grads=True, write_images=True))
        #EarlyStopping
        callbacks.append(EarlyStopping(monitor="loss", min_delta=0.0002, patience=5))
        # EpochEndCallback
        callbacks.append(EpochEndCallback(self))
        
        if save_every_epoch:
            # ModelCheckpoint
            callbacks.append(ModelCheckpoint(self.model_path + "weights-{epoch:04d}.hdf5", save_weights_only=True))
        
        # actual training takes place here
        self.model.fit_generator(self.dp.train_generator_with_padding(self.SEQUENCE_LENGTH, limit), 
                                steps_per_epoch=steps_per_epoch, 
                                epochs=self.EPOCHS_TRAINED + epochs,
                                verbose=1, 
                                callbacks=callbacks,
                                initial_epoch = self.EPOCHS_TRAINED)
        
        self.model.save_weights(self.model_path + "weights.hdf5")

    # load already trained model
    def load_model(self, model_dir_path, weights_filename="weights.hdf5"):

        self.model_path = model_dir_path

        # load model information
        with open(join(self.model_path, "model.json"), "r") as model_json:
            self.model = model_from_json(model_json.read())

        # load saved variables
        with open(join(self.model_path, "variables.json"), "r") as variable_json:
            variables = json.load(variable_json)
            self.SEQUENCE_LENGTH = int(variables["SEQUENCE_LENGTH"])

            if "TIMESIGNATURE" in variables.keys():
                self.TIMESIGNATURE = variables["TIMESIGNATURE"]
            else:
                # for backwards-compatibility
                self.TIMESIGNATURE = model_dir_name.replace("model-","")
            
            if "EPOCHS_TRAINED" in variables.keys():
                self.EPOCHS_TRAINED = int(variables["EPOCHS_TRAINED"])
            else:
                # for backwards-compatibility
                self.EPOCHS_TRAINED = 0
            
        # load the dataprocessor
        with open(join(self.model_path, "dataprocessor.pkl"), "rb") as file:
                self.dp = pickle.load(file)


        self.model.load_weights(join(self.model_path, weights_filename))

        self.compile()
    
    def make_song(self, alpha=0.0, length=200):
        song = []
        sequence = np.ones((1, self.SEQUENCE_LENGTH, 6))

        # start off with one randomly generated note
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

        sequence[0][-1][5] = 0.0

        song.append((self.dp.num_to_note(random_note),
                     self.dp.num_to_duration(random_duration),
                     self.dp.num_to_offset(random_offset),
                     random_volume,
                     random_tempo,
                     0.0))

        # then continuously predict the next note
        for i in range(length-1):
            note_prediction, duration_prediction, offset_prediction, volume_prediction, tempo_prediction, belongs_to_prev_chord_prediction = self.model.predict(sequence)

            """
            note_index = max(note_prediction[0])[0]
            duration_index = max(duration_prediction[0])[0]
            offset_index = max(offset_prediction[0])[0]
            """

            note_index = choose_entry(note_prediction[0], alpha)
            duration_index = choose_entry(duration_prediction[0], alpha)
            offset_index = choose_entry(offset_prediction[0], alpha)

            volume_prediction = volume_prediction[0][0]
            tempo_prediction = tempo_prediction[0][0]
            
            """
            belongs_to_prev_chord_index = max(belongs_to_prev_chord_prediction[0])[0]
            """
            belongs_to_prev_chord_index = choose_entry(belongs_to_prev_chord_prediction[0], alpha)
            
            song.append((self.dp.num_to_note(note_index),
                         self.dp.num_to_duration(duration_index),
                         self.dp.num_to_offset(offset_index),
                         volume_prediction,
                         tempo_prediction,
                         float(belongs_to_prev_chord_index)))

            sequence = np.roll(sequence, -sequence.shape[2])

            sequence[0][-1][0] = note_index / len(self.dp.note_vocab)
            sequence[0][-1][1] = duration_index / len(self.dp.duration_vocab)
            sequence[0][-1][2] = offset_index / len(self.dp.offset_vocab)
            sequence[0][-1][3] = volume_prediction
            sequence[0][-1][4] = tempo_prediction
            sequence[0][-1][5] = float(belongs_to_prev_chord_index)

        return song
        
    def compile(self):
        opt = tf.keras.optimizers.Adadelta()
        self.model.compile(
            loss={"note_output" : "categorical_crossentropy",
                  "duration_output" : "categorical_crossentropy",
                  "offset_output" : "categorical_crossentropy",
                  "volume_output" : "mse",
                  "tempo_output" : "mse",
                  "belongs_to_prev_chord_output" : "binary_crossentropy"},
            loss_weights=self.get_lossweights(),
            optimizer=opt,
            metrics=["accuracy"])
    
    def get_lossweights(self, smoothing=0.4):
        output_sizes = [len(self.dp.note_vocab),
                        len(self.dp.duration_vocab),
                        len(self.dp.offset_vocab),
                        1,
                        1,
                        2]
        
        ## Hyperparameter
        weights = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]

        output_names = ["note_output", "duration_output", "offset_output", "volume_output", "tempo_output", "belongs_to_prev_chord_output"]

        output_weights = []

        for i in range(len(output_sizes)):
            a = float(weights[i] * sum(output_sizes)) / float(output_sizes[i] * sum(weights))

            output_weights.append(smoothing * (a - 1) + 1)

        return dict(zip(output_names, output_weights))


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
    parser.add_argument("-continue_training",
                        type=str,
                        default=None,
                        help="Continue training the specified model")
    parser.add_argument("-target_dir",
                        type=str,
                        default="./models/",
                        help="Specify a target directory in which your model-directory will be saved,\ndefaults to './models/'")
    parser.add_argument("--save_every_epoch",
                        action="store_true",
                        help="Use this flag to save the weights of the model for each epoch")
    
    args = parser.parse_args()

    model = MuRNN()

    if args.continue_training != None:
        model.load_model(args.continue_training)
    else:
        model.new_model(args.dir, sequence_length=args.seq_len, target_dir=args.target_dir)
        
    model.train(args.steps_per_epoch, args.epochs, save_every_epoch=args.steps_per_epoch, limit=args.limit)
