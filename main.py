import tensorflow as tf
import datetime
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, CuDNNLSTM, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard

import json
import numpy as np
import os
from dataprocessing import DataProcessor
import pickle
import random

import matplotlib.pyplot as plt


def get_datetime_str():
    return '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())

def mkdir_safely(path):
    if not os.path.exists(path):
        os.mkdir(path)

def max(array):
    curr_index = -1
    curr_value = -np.inf
    for i in range(len(array)):
        if array[i] > curr_value:
            curr_index = i
            curr_value = array[i]
    return (curr_index, curr_value)



class Model:
    
    def __init__(self, data_dir):
        
        mkdir_safely("./models/")

        self.dp = None
        self.data_path = data_dir
        self.SEQUENCE_LENGTH = 30
        self.model = None
        
        self.timesignature = get_datetime_str()
    
    def new_model(self):

        self.dp = DataProcessor(self.data_path)

        self.model = Sequential()

        self.model.add(CuDNNLSTM(500, input_shape=(None, 1), return_sequences=False))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(1500, activation="relu"))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(len(self.dp.vocab), activation="softmax"))

        print(self.model.summary(90))


        self.compile()

    def train(self, save_every_epoch=False):
        path = "./models/model-" + self.timesignature + "/"

        mkdir_safely(path)

        tensorboard = TensorBoard(log_dir=path + "logs", write_grads=True, write_images=True)

        callbacks = [tensorboard]

        if save_every_epoch:
            checkpointer = ModelCheckpoint(path + "weights-{epoch:02d}.hdf5", save_weights_only=True)
            callbacks.append(checkpointer)

        #self.model.fit_generator(self.dp.train_generator_with_padding(self.SEQUENCE_LENGTH), steps_per_epoch=500, epochs=25, verbose=1, callbacks=callbacks)
        self.model.fit_generator(self.dp.train_generator_with_padding(self.SEQUENCE_LENGTH), steps_per_epoch=1, epochs=2, verbose=1, callbacks=callbacks)
        self.model.save_weights(path + "weights.hdf5")

        with open(path + "model.json", "w") as file:
            file.write(self.model.to_json())
        with open(path + "variables.json", "w") as file:
            file.write('{ "SEQUENCE_LENGTH" : ' + str(self.SEQUENCE_LENGTH) + ', ' +
                       '"TIMESIGNATURE" : "' + self.timesignature + '" }')

    # path to directory
    def load_model(self, model_dir, weights_filename="weights.hdf5"):

        path = "./models/" + model_dir + "/"
        with open(path + "model.json", "r") as model_json:
            self.model = model_from_json(model_json.read())

        with open(path + "variables.json", "r") as variable_json:
            variables = json.load(variable_json)
            self.SEQUENCE_LENGTH = int(variables["SEQUENCE_LENGTH"])

            if "TIMESIGNATURE" in variables.keys():
                self.timesignature = variables["TIMESIGNATURE"]
            else:
                # for backwards-compatibility
                self.timesignature = model_dir.replace("model-","")
            
            self.dp = DataProcessor(self.data_path)

        self.model.load_weights(path + weights_filename)

        self.compile()
    
    def make_song(self, length):
        np.set_printoptions(threshold=np.inf)

        song = []
        sequence = np.ones((1, self.SEQUENCE_LENGTH, 1))
        random_note = random.randint(0, len(self.dp.vocab))
        sequence[0][-1][0] =  random_note / len(self.dp.vocab)
        song.append(self.dp.num_to_note(random_note))

        for i in range(length-1):
            #input("press any button...")
            prediction = self.model.predict(sequence)[0]
            
            index_of_prediction = np.where(prediction == np.amax(prediction))[0][0]
            note = self.dp.num_to_note(index_of_prediction)
            
            song.append(note)

            sequence = np.roll(sequence, -sequence.shape[2])

            sequence[0][-1][0] = index_of_prediction / len(self.dp.vocab)

            #print(sequence)
            #print(index_of_prediction)

        mkdir_safely("./models/model-" + self.timesignature + "/songs/")
        
        songpath = "./models/model-" + self.timesignature + "/songs/song-" + get_datetime_str() + ".mu"

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
    # something wrong with retrieve v2
    model.train()
    #model.load_model("model-2019-08-16_10-20-51")
    model.dp.retrieve_midi_from_processed_file(model.make_song(300))

############# MODEL HAS TO BE TRANSFORMED AFTER TRAINING,
############# SO IT CAN RUN ON BOTH CPU AND GPU