import music21

import numpy as np
import random # may not be random enough
from decimal import Decimal

from os.path import isdir, isfile, join, splitext
from os import listdir
import pickle
import glob

import datetime

#### Helper functions

def get_datetime_str():
    return '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())

class DataProcessor:

    default_limit = 1000
    twelfth = Decimal("0.08333333333333333333") # 1/12 

    def __init__(self, dir_path):

        self.next_batch_is_new_song = None

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

            self.note_vocab, self.duration_vocab, self.offset_vocab = self.get_vocab()

            # NOTE
            self.notes_to_num_dict, self.num_to_notes_dict = self.make_note_conversion_dictionaries()
            # DURATION
            self.durations_to_num_dict, self.num_to_durations_dict = self.make_duration_conversion_dictionaries()
            # OFFSET
            self.offsets_to_num_dict, self.num_to_offsets_dict = self.make_offset_conversion_dictionaries()
            
            print("Note vocab: " + str(len(self.note_vocab)) + "  Duration vocab: " + str(len(self.duration_vocab))
                + "  Offset vocab: " + str(len(self.offset_vocab)) 
                + "  TOTAL: " + str(len(self.note_vocab)+len(self.duration_vocab)+len(self.offset_vocab)))
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

            previous_offset = 0.0

            for note in m_stream.notes:
                
                note_duration = float(self.quarterLength_to_int_representation(str(note.duration.quarterLength)))

                note_offset = float(self.quarterLength_to_int_representation(str(note.offset)) - previous_offset)
                previous_offset = float(self.quarterLength_to_int_representation(str(note.offset)))
                    
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
            
                notes.append((pitch_name_string, note_duration, note_offset))
            
            with open(splitext(f_path)[0] + ".mu", "wb") as f:
                pickle.dump(notes, f)
            return True
        return False

    def train_generator_no_padding(self, sequence_length=10):
        print(
        """
            DO NOT USE THIS, THIS DOES NOT WORK YET

        """)


        raise NotImplementedError
        
        
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


        remainder = []

        while True:
            if len(remainder) == 0:
                # random file chosen and loaded
                filename = splitext(random.choice(self.files))[0]
                music_data = DataProcessor.load_processed_file(join(self.dir_path, filename + ".mu"))
                
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
                
                if batch_size > self.LIMIT:
                    remainder = list(zip(np.array_split(x_train, np.ceil(len(x_train)/self.LIMIT)), np.array_split(y_train, np.ceil(len(y_train)/self.LIMIT))))
                    x_train, y_train = remainder.pop()

            else:
                x_train, y_train = remainder.pop()
                
            yield x_train, y_train
                    
                    
    def train_generator_with_padding(self, sequence_length=10, LIMIT=default_limit):

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

        remainder = []

        while True:
            if len(remainder) == 0:
                # random file chosen and loaded
                filename = splitext(random.choice(self.files))[0]
                music_data = DataProcessor.load_processed_file(join(self.dir_path, filename + ".mu"))

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
                
                x_train = np.ones((batch_size, sequence_length, 3))

                y_train_notes = np.zeros((batch_size, len(self.note_vocab)))
                y_train_duration = np.zeros((batch_size, len(self.duration_vocab)))
                y_train_offset = np.zeros((batch_size, len(self.offset_vocab)))

                for i in range(batch_size):
                    # shift batches in x_train one step to the left
                    x_train = np.roll(x_train, -x_train.shape[1]*x_train.shape[2])
                    # add new note/duration/offset
                    x_train[-1] = np.roll(x_train[-2], -x_train.shape[2])

                    x_train[-1][-1][0] = float(self.note_to_num(music_data[i][0])) / float(len(self.note_vocab))
                    x_train[-1][-1][1] = float(self.duration_to_num(music_data[i][1])) / float(len(self.duration_vocab))
                    x_train[-1][-1][2] = float(self.offset_to_num(music_data[i][2])) / float(len(self.offset_vocab))

                    if i != batch_size-1:
                        y_train_notes[i][self.note_to_num(music_data[i+1][0])] = 1.0
                        y_train_duration[i][self.duration_to_num(music_data[i+1][1])] = 1.0
                        y_train_offset[i][self.offset_to_num(music_data[i+1][2])] = 1.0
                
                if batch_size > LIMIT:
                    remainder = list(zip(np.array_split(x_train, np.ceil(len(x_train)/LIMIT)), 
                                         np.array_split(y_train_notes, np.ceil(len(y_train_notes)/LIMIT)),
                                         np.array_split(y_train_duration, np.ceil(len(y_train_duration)/LIMIT)),
                                         np.array_split(y_train_offset, np.ceil(len(y_train_offset)/LIMIT))))
                    
                    x_train, y_train_notes, y_train_duration, y_train_offset = remainder.pop()

            else:
                x_train, y_train_notes, y_train_duration, y_train_offset = remainder.pop()
            
            
            self.next_batch_is_new_song = len(remainder) == 0
            
            yield [x_train], [y_train_notes, y_train_duration, y_train_offset]

    # STRUCTURE OF THE LOADED FILE
    # --> see create_processed_file
    @staticmethod
    def load_processed_file(f_path):
        if isfile(f_path)and splitext(f_path)[1] == ".mu":
            with open(f_path, "rb") as f:
                return pickle.load(f)
        else:
            print("File with path '" + f_path + "' could not be loaded...")
            return None    

    @staticmethod
    def retrieve_midi_from_loaded_data(data, target_dir="./"):

        stream = music21.stream.Stream()
        curr_offset = 0.0
        for tup in data:
            pitch, duration, offset = tup
            
            duration = DataProcessor.int_representation_to_quarterLength(float(duration))
            
            offset = DataProcessor.int_representation_to_quarterLength(float(offset))

            curr_offset += offset

            if "," in pitch:
                note = music21.chord.Chord(pitch.split(","))
            else:
                note = music21.note.Note(pitch)
            note.duration = music21.duration.Duration(duration)

            stream.insert(curr_offset, note, ignoreSort=True)
        stream.write('midi', fp=(target_dir + "song-" + get_datetime_str() +".mid"))

    ### NOTE
    def make_note_conversion_dictionaries(self):
        notes_to_num = dict((note,num) for num,note in enumerate(self.note_vocab))
        num_to_notes = dict((num,note) for num,note in enumerate(self.note_vocab))

        return notes_to_num, num_to_notes

    # shortcut to convert from note to number
    def note_to_num(self, note):
        return self.notes_to_num_dict[note]
    # shortcut to convert from number to note
    def num_to_note(self, num):
        return self.num_to_notes_dict[num]

    ### DURATION
    def make_duration_conversion_dictionaries(self):
        durations_to_num = dict((duration,num) for num,duration in enumerate(self.duration_vocab))
        num_to_durations = dict((num,duration) for num,duration in enumerate(self.duration_vocab))

        return durations_to_num, num_to_durations

    # shortcut to convert from duration to number
    def duration_to_num(self, duration):
        return self.durations_to_num_dict[duration]
    # shortcut to convert from number to duration
    def num_to_duration(self, num):
        return self.num_to_durations_dict[num]
    
    ### OFFSET
    def make_offset_conversion_dictionaries(self):
        offsets_to_num = dict((offset,num) for num,offset in enumerate(self.offset_vocab))
        num_to_offsets = dict((num,offset) for num,offset in enumerate(self.offset_vocab))

        return offsets_to_num, num_to_offsets
    
    # shortcut to convert from offset to number
    def offset_to_num(self, offset):
        return self.offsets_to_num_dict[offset]
    # shortcut to convert from number to offset
    def num_to_offset(self, num):
        return self.num_to_offsets_dict[num]
    
    def get_vocab(self):
        note_vocab = set()
        duration_vocab = set()
        offset_vocab = set()

        for file in glob.glob(self.dir_path + "*.mu"):
            with open(file, "rb") as f:
                data = pickle.load(f)
                for tup in data:
                    note_vocab.add(tup[0])
                    duration_vocab.add(tup[1])
                    offset_vocab.add(tup[2])
        
        return tuple(note_vocab), tuple(duration_vocab), tuple(offset_vocab)
    
    # convert int represented quarterLengths back
    @staticmethod
    def int_representation_to_quarterLength(value):
        return float(Decimal(value) * DataProcessor.twelfth)

    # convert quarterLengths to integer representation
    @staticmethod
    def quarterLength_to_int_representation(value):
        # this will always return integers values, but as floats
        if len(value.split("/")) > 1:
            return round(float(((Decimal(value.split("/")[0]) / Decimal(value.split("/")[1])) / DataProcessor.twelfth)))
        else:
            return round(float(Decimal(value) / DataProcessor.twelfth))

    