from dataprocessing import DataProcessor
from decimal import Decimal
import music21
import numpy as np
#import tensorflow as tf

d = DataProcessor("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")

#d.make_conversion_dictionaries()
#print(len(d.get_vocab()))

#print(".".join(["brah", "yag"]))
#d.create_processed_file("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\megalovania.mid")
#d.process_files("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")

#print(d.load_processed_file("D:\\leont\\Documents\\Schule\\W-Seminar\\NN\\models\\model-2019-08-06_17-54-15\\songs\\song-2019-08-06_18-46-29.mu"))

#d.retrieve_midi_from_processed_file("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\megalovania.mu")

#print(next(d.train_generator())[1].shape)

#stream = music21.converter.parse("../test_v2/megalovania.mid").flat.sorted

#print(stream.show("text"))
"""
bruh = np.zeros((1,2,4))
bruh[0][-1][2] = 1
bruh = np.roll(bruh, -4)

print(bruh)

"""
np.set_printoptions(threshold=np.inf)
print(next(d.train_generator_with_padding())[0])

#### wait a fuckin sec






#print(next(d.train_generator_no_padding(sequence_length=2))[0])

#print(next(d.train_generator_no_padding()))

#d.retrieve_midi_from_processed_file_v3("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\what_is_this_thing.txt")

#bruh = [("a", 1), ("b", 2), ("c", 3)]

#print(list(zip(*bruh)))

"""

#next(d.train_generator_no_padding(5))

a = np.zeros((3,3))
a[2][2] = 1
a[1][2] = 2
a[0][2] = 3
print(a)
a = np.roll(a, -1*3*2)
print(a)

"""