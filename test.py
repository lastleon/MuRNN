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

#next(d.train_generator_no_padding(5))

a = np.zeros((3,3))
a[2][2] = 1
a[1][2] = 2
a[0][2] = 3
print(a)
a = np.roll(a, -1*3*2)
print(a)

