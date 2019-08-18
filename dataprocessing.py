import music21

import numpy as np
import random # may not be random enough
from decimal import Decimal

from os.path import isdir, isfile, join, splitext
from os import listdir
import pickle
import glob

class DataProcessor():

    def __init__(self, dir_path):

        # check if path to directory is valid
        if isdir(dir_path):
            self.dir_path = dir_path
        else:
            raise Exception("Provided path is faulty...")

        self.process_files()


    def process_files(self):
        
        # files are saved here (not full path, but names + extension only)
        files = []

        complete_filelist = listdir(self.dir_path)

        # if a file is of type mid(i), then its
        # name is saved in 'files'
        for file in complete_filelist:
            f_name, f_extension = splitext(file)

            if f_extension == ".midi" or f_extension == ".mid":
                files.append(file)

        # track how many files are created
        created_files_counter = 0

        # iterate over all of the mid(i) files
        for file in files.copy():
            # if a there is no .mu file to a midi file, it will be created 
            # (if possible)
            if not splitext(file)[0] + ".mu" in complete_filelist:
                # if not possible, the filename is removed from 'files'
                if self.create_processed_file(join(self.dir_path, file)):
                    print("File '" + splitext(file)[0] + ".mu' was created...")
                    created_files_counter += 1
                else:
                    files.remove(file)

        if len(files) == 0:
            raise FileNotFoundError("There are no fitting files in this directory...")
        else:
            self.files = files
            self.vocab = self.get_vocab()
            self.notes_to_num_dict, self.num_to_notes_dict = self.make_conversion_dictionaries()
            print(str(created_files_counter) + (" new file was created" if created_files_counter == 1 else " new files were created"))
            print("Finished!")
    
    def create_processed_file(self, f_path):
        ### STRUCTURE
        # [
        #   "C#4 4.0",
        #   "D#4,C4,E4 1/3",
        #   ... 
        # ]
        if isfile(f_path) and (splitext(f_path)[1] == '.midi' or splitext(f_path)[1] == '.mid') and not splitext(f_path)[0].endswith("retrieved"):
            # stream of notes flattened and sorted
            m_stream = music21.converter.parse(f_path).flat.sorted
            
            notes = []

            for note in m_stream.notes:

                note_duration = note.duration.quarterLength
                    
                pitches = note.pitches
                pitch_names = []
                
                # add pitchnames to list, if note is a chord (has more than one pitch) then they are first sorted
                # if a pitch is ambiguous (e.g. C# <-> D-), then the (alphabetically) first name is chosen
                for i in range(len(pitches)):
                    if not (pitches[i].accidental == None or pitches[i].accidental.modifier == ""):
                        pitch_versions = [pitches[i].nameWithOctave, pitches[i].getEnharmonic().nameWithOctave]
                        pitch_versions.sort()
                        pitch_names.append(pitch_versions[0])
                    else:
                        pitch_names.append(pitches[i].nameWithOctave)
                
                pitch_names.sort()
                pitch_name_string = ",".join(pitch_names)
                notes.append(pitch_name_string + " " + str(note_duration))

            with open(splitext(f_path)[0] + ".mu", "wb") as f:
                pickle.dump(notes, f)
            return True
        return False

    def train_generator_no_padding(self, sequence_length=10):

        #############   PRODUCED DATA:
        #### x_train:
        # SHAPE: (batch_size, sequence_length, features)
        # [batch_size]      number of shifts + 1 (because of the initial state), example (with sequence_length=2):
        #                   [C  E] D  E  A    // + 1
        #                    C [E  D] E  A    // shift 1
        #                    C  E [D  E] A    // shift 2
        #                    C  E  D [E  A]   // shift 3
        #                    -----------------//--------
        #                                          4 
        # [sequence_length] number of timesteps in a single datapoint in a batch, example:
        #                   [C  E] D  E  A  --> sequence_length = 2
        #                   [C  E  D] E  A  --> sequence_length = 3
        #                   [C  E  D  E] A  --> sequence_length = 4
        #                   ...
        # [features]        note converted to num and normalized with vocab size --> always 1

        ############# FAILSAFE
        # --> GPU-memory capabilities aren't exceeded
        # if the computed batch_size is greater than LIMIT,
        # then the data is split into three sub-arrays of 
        # about the same size, but where each array-length is
        # less than LIMIT. These sub-arrays are yielded seperately,
        # only after the last sub-array has been yielded a new song
        # is loaded

        LIMIT = 200

        remainder = []

        while True:
            if len(remainder) == 0:
                # random file chosen and loaded
                filename = splitext(random.choice(self.files))[0]
                music_data = self.load_processed_file(join(self.dir_path, filename + ".mu"))
                
                # sequence length shouldn't be larger than the whole music data array
                if len(music_data) < sequence_length:
                    sequence_length = len(music_data)

                # batch_size is number of shifts of the training data array + 1
                batch_size = len(music_data) - sequence_length + 1

                ## shape of training data is (batch_size, timesteps, features)
                # x_train example with batch_size = 2 and timesteps = 3: 
                # [ 
                #   [0.123, 0.125, 0.520],
                #   [0.125, 0.520, 1.000]
                # ]
                # ---> encoded notes
                #
                # y_train example with batch_size = 2 and len(vocab) = 5:
                # [
                #   [0.000, 0.000, 1.000, 0.000, 0.000],
                #   [0.000, 0.000, 0.000, 0.000, 1.000]
                # ]
                # ---> probability of a note
                
                x_train = np.ones((batch_size, sequence_length, 1))
                y_train = np.zeros((batch_size, len(self.vocab)))

                for i in range(batch_size): 
                    for j in range(sequence_length):
                        note = music_data[i+j]    
                        # numerical representation of note/chord, normalized
                        x_train[i][j][0] = float(self.note_to_num(note)) / float(len(self.vocab))

                    if i == batch_size-1:
                        y_train[i] = np.zeros(len(self.vocab))
                    else:
                        note = music_data[i+sequence_length]
                        y_train[i][self.note_to_num(note)] = 1.0
                
                if batch_size > LIMIT:
                    remainder = list(zip(np.array_split(x_train, np.ceil(len(x_train)/LIMIT)), np.array_split(y_train, np.ceil(len(y_train)/LIMIT))))
                    x_train, y_train = remainder.pop()

            else:
                x_train, y_train = remainder.pop()
                
            yield x_train, y_train
                    
                    
    def train_generator_with_padding(self, sequence_length=10):

        #############   PRODUCED DATA:
        #### x_train:
        # SHAPE: (batch_size, sequence_length, features)
        # [batch_size]      numeber of shifts in data (which is the length), example (with sequence_length=2):
        #                   [0  E] D  E  A    // + 1
        #                    0 [E  D] E  A    // shift 1
        #                    0  E [D  E] A    // shift 2
        #                    0  E  D [E  A]   // shift 3
        #                    -----------------//--------
        #                                          4
        # [sequence_length] number of timesteps in a single datapoint in a batch, example:
        #                   [C  E] D  E  A  --> sequence_length = 2
        #                   [C  E  D] E  A  --> sequence_length = 3
        #                   [C  E  D  E] A  --> sequence_length = 4
        #                   ...
        # [features]        note converted to num and normalized with vocab size --> always 1

        ############# FAILSAFE
        # --> description in train_generator_no_padding

        LIMIT = 350

        remainder = []

        while True:
            if len(remainder) == 0:
                # random file chosen and loaded
                filename = splitext(random.choice(self.files))[0]
                music_data = self.load_processed_file(join(self.dir_path, filename + ".mu"))

                # batch_size is number of shifts of the training data array
                batch_size = len(music_data)

                ## shape of training data is (batch_size, timesteps, features)
                # x_train example with batch_size = 2 and timesteps = 3: 
                # [ 
                #   [0.000, 0.000, 0.520],
                #   [0.000, 0.520, 0.500]
                # ]
                # ---> encoded notes with padding
                #
                # y_train example with batch_size = 2 and len(vocab) = 5:
                # [
                #   [0.000, 0.000, 1.000, 0.000, 0.000],
                #   [0.000, 0.000, 0.000, 0.000, 1.000]
                # ]
                # ---> probability of a note
                
                x_train = np.ones((batch_size, sequence_length, 1))
                y_train = np.zeros((batch_size, len(self.vocab)))

                for i in range(batch_size):
                    # shift batches in x_train one step to the left
                    x_train = np.roll(x_train, -x_train.shape[1]*x_train.shape[2])
                    # add new note
                    x_train[-1] = np.roll(x_train[-2], -x_train.shape[2])
                    x_train[-1][-1][0] = float(self.note_to_num(music_data[i])) / float(len(self.vocab))

                    if i != batch_size-1:
                        y_train[i][self.note_to_num(music_data[i+1])] = 1.0
                
                if batch_size > LIMIT:
                    remainder = list(zip(np.array_split(x_train, np.ceil(len(x_train)/LIMIT)), np.array_split(y_train, np.ceil(len(y_train)/LIMIT))))
                    x_train, y_train = remainder.pop()

            else:
                x_train, y_train = remainder.pop()
                
            yield x_train, y_train

    # STRUCTURE OF THE LOADED FILE
    # --> see create_processed_file
    def load_processed_file(self, f_path):
        if isfile(f_path):
            with open(f_path, "rb") as f:
                return pickle.load(f)
        else:
            print("File with path '" + f_path + "' could not be loaded...")
            return None    

    def retrieve_midi_from_processed_file(self, f_path):
        if isfile(f_path) and splitext(f_path)[1] == ".mu":
            with open(f_path, "rb") as f:
                data = pickle.load(f)
                stream = music21.stream.Stream()
                for datapoint in data:
                    pitch, duration = datapoint.split(" ")
                    
                    if len(duration.split("/")) > 1:
                        duration = float(Decimal(duration.split("/")[0]) / Decimal(duration.split("/")[1]))
                    else:
                        duration = float(duration)

                    if "," in pitch:
                        note = music21.chord.Chord(pitch.split(","))
                    else:
                        note = music21.note.Note(pitch)
                    note.duration = music21.duration.Duration(duration)
                    stream.append(note)
                stream.write('midi', fp=(splitext(f_path)[0]+"-retrieved.midi"))
        else:
            print("File with path '" + f_path + "' could not be retrieved...")

    def make_conversion_dictionaries(self):
        notes_to_num = dict((note,num) for num,note in enumerate(self.vocab))
        num_to_notes = dict((num,note) for num,note in enumerate(self.vocab))

        return notes_to_num, num_to_notes

    # shortcut to convert from note to number
    def note_to_num(self, note):
        return self.notes_to_num_dict[note]
    # shortcut to convert from number to note
    def num_to_note(self, num):
        return self.num_to_notes_dict[num]

    def get_vocab(self):
        vocab = set()
        for file in glob.glob(self.dir_path + "*.mu"):
            with open(file, "rb") as f:
                data = pickle.load(f)
                for note in data:
                    vocab.add(note)
        
        return tuple(vocab)
