import tensorflow as tf
import datetime
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, CuDNNLSTM, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

import json
import numpy as np
import os
from dataprocessing import DataProcessor


def get_datetime_str():
    return '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())

def mkdir_safely(path):
    if not os.path.exists(path):
        os.mkdir(path)



class Model:
    
    def __init__(self):

        mkdir_safely("./models/")

        self.SEQUENCE_LENGTH = 300
        self.dp = DataProcessor("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")
        self.model = None
        
        self.timesignature = get_datetime_str()
    
    def new_model(self):
        self.model = Sequential()

        self.model.add(CuDNNLSTM(88, input_shape=(None, 88), return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(CuDNNLSTM(88))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(88, activation="relu"))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(88, activation="sigmoid"))

        print(self.model.summary(90))


        self.compile()

    def train(self, save_every_epoch=False):
        path = "./models/model-" + self.timesignature + "/"

        mkdir_safely(path)

        if save_every_epoch:
            checkpointer = ModelCheckpoint(path + "weights.hdf5")
            self.model.fit_generator(self.dp.train_generator_no_padding(self.SEQUENCE_LENGTH), steps_per_epoch=500, epochs=40, verbose=1, callbacks=[checkpointer])
        else:
            self.model.fit_generator(self.dp.train_generator_no_padding(self.SEQUENCE_LENGTH), steps_per_epoch=500, epochs=40, verbose=1)
            self.model.save_weights(path + "weights.hdf5")

        with open(path + "model.json", "w") as file:
            file.write(self.model.to_json())
        with open(path + "variables.json", "w") as file:
            file.write('{ "SEQUENCE_LENGTH" : ' + str(self.SEQUENCE_LENGTH) + ' }')

    # path to directory
    def load_model(self, model_dir):
        path = "./models/" + model_dir + "/"
        with open(path + "model.json", "r") as model_json:
            self.model = model_from_json(model_json.read())

        with open(path + "variables.json", "r") as variable_json:
            self.SEQUENCE_LENGTH = int(json.load(variable_json)["SEQUENCE_LENGTH"])
        self.model.load_weights(path + "weights.hdf5")
        self.timesignature = model_dir.replace("model-", "")

        self.compile()
    
    # chord must be list of numbers between 0 and 87
    # TODO: sanitize input
    def make_song(self, length, chord=[50]):

        song_string = ""
        # np.set_printoptions(threshold=np.inf)

        song = np.zeros((length, 88))
        sequence = np.zeros((1, self.SEQUENCE_LENGTH, 88))

        for note in chord:
            sequence[0][-1][note] = 1.0
            song[0][note] = 1.0

        song_string += self.dp.one_hot_vec_to_string(song[0])
        
        
        for i in range(length-1):

            prediction = self.model.predict(sequence, batch_size=1)[0]
            
            for j in range(len(prediction)):

                if prediction[j] < 0.33:
                    prediction[j] = 0
                elif prediction[j] < 0.66:
                    prediction[j] = 0.5
                else:
                    prediction[j] = 1.0
            
            song[i+1] = prediction[:]

            sequence = np.roll(sequence, -88)
            sequence[0][-1] = prediction[:]

            song_string += self.dp.one_hot_vec_to_string(prediction[:])

        for i in range(length):
            print(np.array_equal(song[i], np.zeros(song[i].shape)))

        mkdir_safely("./models/model-" + self.timesignature + "/songs/")
        
        songpath = "./models/model-" + self.timesignature + "/songs/song-" + get_datetime_str() + ".txt"

        with open(songpath, "w") as file:
            file.write(song_string)
            file.flush()

        return songpath
    
    def compile(self):
        optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

        self.model.compile(loss="mean_absolute_error",
            optimizer=optimizer,
            metrics=["accuracy"])
        
if __name__ == '__main__':
    model = Model()
    
    model.new_model()
    model.train()
    #model.load_model("model-2019-08-02_18-44-34")
    
    model.dp.retrieve_midi_from_processed_file_v3(model.make_song(300))

############# MODEL HAS TO BE TRANSFORMED AFTER TRAINING,
############# SO IT CAN RUN ON BOTH CPU AND GPU