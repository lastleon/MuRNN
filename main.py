import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, CuDNNLSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import datetime
import numpy as np
import random

import json
import pickle
from os import mkdir
from os.path import exists
from dataprocessing import DataProcessor

### helper functions

def get_datetime_str():
    return '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())

def mkdir_safely(path):
    if not exists(path):
        os.mkdir(path)




class Model:
    
    def __init__(self, data_dir):
        
        mkdir_safely("./models/")

        self.dp = None
        self.data_path = data_dir
        
        self.model = None
        self.model_path = None
        self.SEQUENCE_LENGTH = 50

        self.timesignature = get_datetime_str()
        
    
    def new_model(self):
        # make new model directory
        self.model_path = "./models/model-" + self.timesignature + "/"
        mkdir_safely(self.model_path)

        self.dp = DataProcessor(self.data_path)

        # make model
        self.model = Sequential()

        self.model.add(CuDNNLSTM(500, input_shape=(None, 1), return_sequences=False))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(1500, activation="relu"))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(len(self.dp.vocab), activation="softmax"))

        print(self.model.summary(90))

        self.compile()

    def train(self, save_every_epoch=False):

        tensorboard = TensorBoard(log_dir=self.model_path + "logs/", write_grads=True, write_images=True)

        callbacks = [tensorboard]

        if save_every_epoch:
            checkpointer = ModelCheckpoint(self.model_path + "weights-{epoch:04d}.hdf5", save_weights_only=True)
            callbacks.append(checkpointer)

        self.model.fit_generator(self.dp.train_generator_with_padding(self.SEQUENCE_LENGTH), steps_per_epoch=500, epochs=25, verbose=1, callbacks=callbacks)
        self.model.save_weights(self.model_path + "weights.hdf5")

        with open(self.model_path + "model.json", "w") as file:
            file.write(self.model.to_json())
        with open(self.model_path + "variables.json", "w") as file:
            file.write('{ "SEQUENCE_LENGTH" : ' + str(self.SEQUENCE_LENGTH) + ', ' +
                       '"TIMESIGNATURE" : "' + self.timesignature + '" }')

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
        sequence = np.ones((1, self.SEQUENCE_LENGTH, 1))
        random_note = random.randint(0, len(self.dp.vocab))
        sequence[0][-1][0] =  random_note / len(self.dp.vocab)
        song.append(self.dp.num_to_note(random_note))

        for i in range(length-1):
            prediction = self.model.predict(sequence)[0]
            
            index_of_prediction = np.where(prediction == np.amax(prediction))[0][0]
            note = self.dp.num_to_note(index_of_prediction)
            
            song.append(note)

            sequence = np.roll(sequence, -sequence.shape[2])

            sequence[0][-1][0] = index_of_prediction / len(self.dp.vocab)

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


if __name__ == '__main__':
    model = Model(".\\test_dataset\\")

    model.new_model()
    model.train()

    model.dp.retrieve_midi_from_processed_file(model.make_song(300))
