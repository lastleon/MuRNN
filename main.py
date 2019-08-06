import tensorflow as tf
import datetime
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, CuDNNLSTM, LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

import json
import numpy as np
import os
from dataprocessing import DataProcessor
import pickle
import random


def get_datetime_str():
    return '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())

def mkdir_safely(path):
    if not os.path.exists(path):
        os.mkdir(path)



class Model:
    
    def __init__(self):
        
        mkdir_safely("./models/")

        self.dp = None
        self.data_path = ".\\clean_dataset\\"
        self.SEQUENCE_LENGTH = 300
        self.model = None
        
        self.timesignature = get_datetime_str()
    
    def new_model(self):

        self.dp = DataProcessor(self.data_path)

        self.model = Sequential()
        # TODO: change model specs like input shape, ...

        self.model.add(CuDNNLSTM(1, input_shape=(None, 1), return_sequences=True))
        self.model.add(Dropout(0.2))

        self.model.add(CuDNNLSTM(1))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(2500, activation="relu"))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(len(self.dp.vocab), activation="softmax"))

        print(self.model.summary(90))


        self.compile()

    def train(self, save_every_epoch=False):
        path = "./models/model-" + self.timesignature + "/"

        mkdir_safely(path)

        if save_every_epoch:
            checkpointer = ModelCheckpoint(path + "weights.hdf5")
            self.model.fit_generator(self.dp.train_generator_no_padding(self.SEQUENCE_LENGTH), steps_per_epoch=500, epochs=10, verbose=1, callbacks=[checkpointer])
        else:
            self.model.fit_generator(self.dp.train_generator_no_padding(self.SEQUENCE_LENGTH), steps_per_epoch=500, epochs=40, verbose=1)
            #self.model.fit_generator(self.dp.train_generator_test(), steps_per_epoch=500, epochs=40, verbose=1)
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
            variables = json.load(variable_json)
            self.SEQUENCE_LENGTH = int(variables["SEQUENCE_LENGTH"])
            self.dp = DataProcessor(self.data_path)
        self.model.load_weights(path + "weights.hdf5")
        self.timesignature = model_dir.replace("model-", "")

        self.compile()
    
    def make_song(self, length):
        # np.set_printoptions(threshold=np.inf)

        song = []
        sequence = np.ones((1, self.SEQUENCE_LENGTH, 1)) * -1
        random_note = random.randint(0, len(self.dp.vocab))
        sequence[0][-1][0] =  random_note / len(self.dp.vocab)
        song.append(self.dp.num_to_note(random_note))

        #print(sequence)

        for i in range(length-1):
            #input("press any button...")
            prediction = self.model.predict(sequence, batch_size=1)[0]
            
            index_of_prediction = np.where(prediction == np.amax(prediction))[0][0]
            note = self.dp.num_to_note(index_of_prediction)
            
            song.append(note)

            sequence = np.roll(sequence, -1)
            sequence[0][-1][0] = index_of_prediction / len(self.dp.vocab)

            #print(sequence)
            #print(index_of_prediction)

        mkdir_safely("./models/model-" + self.timesignature + "/songs/")
        
        songpath = "./models/model-" + self.timesignature + "/songs/song-" + get_datetime_str() + ".mu"

        with open(songpath, "wb") as f:
            pickle.dump(song, f)

        return songpath
        
    
    def compile(self):
        optimizer = tf.keras.optimizers.SGD(lr=0.01, momentum=0.001)

        self.model.compile(loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"])
        
if __name__ == '__main__':
    model = Model()
    #model.new_model()
    # something wrong with retrieve v2
    #model.train()
    model.load_model("model-2019-08-06_17-54-15")
    
    model.dp.retrieve_midi_from_processed_file(model.make_song(20))

############# MODEL HAS TO BE TRANSFORMED AFTER TRAINING,
############# SO IT CAN RUN ON BOTH CPU AND GPU