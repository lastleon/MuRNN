from dataprocessing import DataProcessor
from decimal import Decimal
import music21
import numpy as np
import tensorflow as tf

#d = DataProcessor("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")
#d.process_files("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")

#d.retrieve_midi_from_processed_file_v2("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\megalovania.txt")

#print(next(d.train_generator())[1].shape)

data = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = data.load_data()

print(y_train.shape)