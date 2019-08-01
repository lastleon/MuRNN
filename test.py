from dataprocessing import DataProcessor
from decimal import Decimal
import music21
import numpy as np
import tensorflow as tf

d = DataProcessor("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")
#d.process_files("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")

#d.retrieve_midi_from_processed_file_v2("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\megalovania.txt")

#print(next(d.train_generator())[1].shape)

np.set_printoptions(threshold=np.inf)

next(d.train_generator_no_padding(5))
