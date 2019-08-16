import music21
from os.path import isdir, isfile, join, splitext, basename, curdir
from os import listdir
from decimal import Decimal
import re
import numpy as np
import random # may not be random enough
import pickle
import glob

class DataProcessor():

    def __init__(self, dir_path):
        

        # declaration for ease of use later
        self.twelfth = Decimal("0.083333333333333333333333333333333333333") # 1 / 12
        self.lowest_piano_note = music21.note.Note("A0")
        self.highest_piano_note = music21.note.Note("C8")

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
            # if a there is no .txt file to a midi file, it will be created 
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
        if isfile(f_path) and (splitext(f_path)[1] == '.midi' or splitext(f_path)[1] == '.mid') and not splitext(f_path)[0].endswith("retrieved"):
            # stream of notes flattened and sorted
            m_stream = music21.converter.parse(f_path).flat.sorted
            
            notes = []

            for note in m_stream.notes:
                # converting both offset and duration
                ##### this float() could make a mess ##### 
                note_duration = note.duration.quarterLength
                
                # every pitch in the note (or chord) gets added
                # to the containers of each step of the duration
                # example:
                # C#: offset=>3.0,          duration=>0.5
                # --> 3.0/0.0833... = 36    0.5/0.0833... = 6
                # ---> notes["36"].append("C#"), notes["37"].append("C#"),
                # ---> ........ notes["41"].append("C#") - end

                
                    
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

                




                """
                for elem in note.pitches:
                    # if elem is a note with only ONE name (contrary to C# <-> D-)
                    # then the string for the note in the regex is only that:
                    # i.e. D4
                    if elem.accidental == None or elem.accidental.modifier == "":
                        notes_for_regex = elem.nameWithOctave
                    # if elem can have two names, the note in the regex is (note1 | note2)
                    else:
                        notes_for_regex = "(" + elem.nameWithOctave + "|" + elem.getEnharmonic().nameWithOctave + ")"

                    # if there is at least one occurance of the note elem, then it won't be added
                    if not re.compile("[\s,\,]?" + notes_for_regex + "[\s,\,]?").search(" ".join(notes[str(note_offset + i)])):
                        # name of the note converted to its number-representation (i.e. A0 -> 0, ..., C8-> 87)
                        notenum = self.note_to_num(elem.simplifyEnharmonic().nameWithOctave)
                        # if the note was already playing before -> 0
                        # if the note is now played for the first time -> 1
                        first_played_note = "1" if i==0 else "0"

                        notes[note_offset + i].append(notenum + "|" + first_played_note)
                """

            # if there is nothing in the notecontainer
            # then only "rest" is written
            with open(splitext(f_path)[0] + ".mu", "wb") as f:
                pickle.dump(notes, f)
            return True
        return False

    def get_vocab(self):
        vocab = set()
        for file in glob.glob(self.dir_path + "*.mu"):
            with open(file, "rb") as f:
                data = pickle.load(f)
                for note in data:
                    vocab.add(note)
        
        return tuple(vocab)

    def train_generator_test(self):
        seq_len = 100

        while True:
            x_train = np.ones((20, seq_len, 88))
            y_train = np.ones((20, 88))

            yield x_train, y_train

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
                filename = splitext(random.choice(self.files))[0]
                music_data = self.load_processed_file(join(self.dir_path, filename + ".mu"))
                #music_data = self.load_processed_file("megalovania")

                if len(music_data) < sequence_length:
                    sequence_length = len(music_data)

                # batch_size is number of shifts of the training data array + 1
                batch_size = len(music_data) - sequence_length + 1

                x_train = np.zeros((batch_size, sequence_length, 1)) # shape of training data is (batch_size, timesteps, features)
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
                filename = splitext(random.choice(self.files))[0]
                music_data = self.load_processed_file(join(self.dir_path, filename + ".mu"))
                #music_data = self.load_processed_file(join(self.dir_path, "megalovania" + ".mu"))

                # batch_size is number of shifts of the training data array
                batch_size = len(music_data)

                x_train = np.ones((batch_size, sequence_length, 1)) # shape of training data is (batch_size, timesteps, features)
                y_train = np.zeros((batch_size, len(self.vocab)))

                for i in range(batch_size):
                    # shift batches in x_train one step to the left
                    x_train = np.roll(x_train, -x_train.shape[1]*x_train.shape[2])
                    # add new note
                    x_train[-1] = np.roll(x_train[-2], -x_train.shape[2])
                    x_train[-1][-1][0] = float(self.note_to_num(music_data[i])) / float(len(self.vocab))

                    if i != batch_size-1:
                        y_train[i][self.note_to_num(music_data[i+1])] = 1.0

                    """
                    for j in range(sequence_length):
                        note = music_data[i+j]    
                        # numerical representation of note/chord, normalized
                        x_train[i][j][0] = float(self.note_to_num(note)) / float(len(self.vocab))

                    if i == batch_size-1:
                        y_train[i] = np.zeros(len(self.vocab))
                    else:
                        note = music_data[i+sequence_length]
                        y_train[i][self.note_to_num(note)] = 1.0
                    """
                
                if batch_size > LIMIT:
                    remainder = list(zip(np.array_split(x_train, np.ceil(len(x_train)/LIMIT)), np.array_split(y_train, np.ceil(len(y_train)/LIMIT))))
                    x_train, y_train = remainder.pop()

            else:
                x_train, y_train = remainder.pop()
                
            yield x_train, y_train

    # assumes the vec is really one hot encoded (i.e. in this case 0, .5, 1.0)
    def one_hot_vec_to_string(self, vec):
        returnstring = ""
        for i in range(len(vec)):
            if vec[i] != 0.0:
                if vec[i] == 0.5:
                    returnstring += str(i) + "|0"
                else:
                    returnstring += str(i) + "|1"
                returnstring += ","
        if len(returnstring) > 0:
            returnstring = returnstring[:-1]
        else:
            returnstring = "r"
        returnstring += " "
        return returnstring

    # STRUCTURE OF THE LOADED FILE
    # [  
    #   [ (note(string), curr_playing(int)), (note, curr_playing) ],         <- timestep 1
    #   [ (note, curr_playing) ],                                            <- timestep 2
    #   [ ...,                                                               <- timestep 3
    # ]
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

    
    """
    # this builds the conversion-dictionaries
    # enharmonics are also considered in the
    # notes_to_num dictionary:
    # {
    #   "A0" : "0",
    #   "A#0": "1",
    #   "B-0": "1",
    #   "B0" : "2",
    #   ...
    # }
    def make_conversion_dictionaries(self):
        # initializes dictionaries
        notes_to_num = {}
        num_to_notes = {}
        # saves the previous note of every iteration
        prev_note = None
        # range is hardcoded here because only all of the
        # notes on a piano are considered
        for i in range(0,88):
            # first entry in the dicts will always be the lowest
            # note of the piano (A0)
            if i == 0:
                notes_to_num[self.lowest_piano_note.pitch.nameWithOctave] = str(i)
                num_to_notes[str(i)] = self.lowest_piano_note.pitch.nameWithOctave

                prev_note = music21.note.Note(self.lowest_piano_note.pitch.nameWithOctave)
            else:
                # new_note is prev_note with '#' added and then simplified -> semitone above prev_note
                new_note = music21.note.Note(prev_note.pitch.name + "#" + str(prev_note.pitch.octave))
                new_note.pitch.simplifyEnharmonic(inPlace=True)
                
                notes_to_num[new_note.pitch.nameWithOctave] = str(i)
                num_to_notes[str(i)] = new_note.pitch.nameWithOctave

                # if the note has an enharmonic (same note but different name, e.g. C# <-> D-)
                # it is also added to the notes_to_num
                if not (new_note.pitch.accidental is None or new_note.pitch.accidental.modifier == ""):
                    notes_to_num[new_note.pitch.getEnharmonic().nameWithOctave] = str(i)

                prev_note = new_note

        return notes_to_num, num_to_notes
    """
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















        ########################## DEPRECATED #####################################
"""

    def create_processed_file(self, f_path):
        if isfile(f_path) and (splitext(f_path)[1] == '.midi' or splitext(f_path)[1] == '.mid'):
            # stream of notes flattened and sorted
            m_stream = music21.midi.translate.midiFilePathToStream(f_path).flat.sorted

            
            notes = {}

            # lower and upper bounds of stream
            # below / beyond the bounds there are just rests
            low_end = round(Decimal(str(m_stream.lowestOffset)) / self.twelfth)
            high_end = round(Decimal(str(m_stream.highestTime)) / self.twelfth) - low_end

            # initialization of the notecontainer for each step
            for i in range(0,high_end+1):
                notes[str(i)] = []

            for note in m_stream.notes:
                # already in converted format
                ##### this float() could make a mess #####
                note_offset = int(round((Decimal(float(note.offset)) / self.twelfth) - low_end)) 
                note_duration = int(round(Decimal(float(note.duration.quarterLength)) / self.twelfth))
                
                # every pitch in the note (or chord) gets added
                # to the containers of each step of the duration
                # example:
                # C#: offset=>3.0,          duration=>0.5
                # --> 3.0/0.0833... = 36    0.5/0.0833... = 6
                # ---> notes["36"].append("C#"), notes["37"].append("C#"),
                # ---> ........ notes["41"].append("C#") - end

                for i in range(0, note_duration):
                    for elem in note.pitches:
                        
                        # if elem is a note with only ONE name (contrary to C# <-> D-)
                        # then the string for the note in the regex is only that:
                        # i.e. D4
                        if elem.accidental == None or elem.accidental.modifier == "":
                            notes_for_regex = elem.nameWithOctave
                        # if elem can have two names, the note in the regex is (note1 | note2)
                        else:
                            notes_for_regex = "(" + elem.nameWithOctave + "|" + elem.getEnharmonic().nameWithOctave + ")"

                        # if there is at least one occurance of the note elem, then it won't be added
                        if not re.compile("[\s,\,]?" + notes_for_regex + "[\s,\,]?").search(" ".join(notes[str(note_offset + i)])):
                            notes[str(note_offset + i)].append(elem.simplifyEnharmonic().nameWithOctave)

            # if there is nothing in the notecontainer
            # then only "rest" is written
            with open(splitext(f_path)[0] + ".txt", "w") as f:
                for i in range(0, len(notes)):
                    if len(notes[str(i)]) == 0:
                        f.write("r,")
                    else:
                        f.write(" ".join(notes[str(i)]) + ",")
            
            return True
        return False
    
    def retrieve_midi_from_processed_file(self, f_path):
        if isfile(f_path) and splitext(f_path)[1] == '.txt':
            # used for storing notes that are playing at the moment
            # of the timestep / assigned to the note is the startoffset
            # of the note
            current_notes = {}

            m_stream = music21.stream.Stream()
            m_stream.autoSort = True

            with open(f_path, "r") as f:
                # list with all timesteps
                content = f.readlines()[0].split(',')
                # the last item will always be an empty String
                # so it can be removed
                del content[-1]
                # used to count the amount of
                # timesteps already passed
                current_timestep = 0
                
                # iterate over every timestep
                for timesteps_with_notes in content:
                    # list with all notes per timestep
                    notes = timesteps_with_notes.split(" ")

                    # iterate over all of the currently playing notes
                    for curr_note in current_notes.copy():
                        
                        # will be useful later
                        curr_note_enharmonic = music21.pitch.Pitch(curr_note).getEnharmonic().nameWithOctave
                        
                        # if the currently playing note (or its enharmonic) isn't in 
                        # 'notes' anymore, this means that the note is over
                        # -> note can be inserted into stream
                        if curr_note not in notes and curr_note_enharmonic not in notes:

                            n = music21.note.Note(curr_note)
                            
                            # offset is converted to quarterLength-form and set
                            n.offset = float(Decimal(current_notes[curr_note]) * self.twelfth)
                            
                            # duration is current_timestep - start_offset, then is converted to
                            # quarterLength-form, the set
                            n.duration = music21.duration.Duration(float(Decimal(current_timestep-current_notes[curr_note]) * self.twelfth))
                            m_stream.insert(n)
                            
                            # delete this note as it isn't playing any longer
                            del current_notes[curr_note]

                    # iterate over all the notes that should be currently playing
                    for curr_note in notes:
                        if curr_note != 'r':
                            # useful later
                            curr_note_enharmonic = music21.pitch.Pitch(curr_note).getEnharmonic().nameWithOctave
                            
                            # if curr_note (or its enharmonic) is not in current_notes, this means that
                            # curr_note should be playing but isn't yet, so we have to
                            # add it
                            if curr_note not in current_notes and curr_note_enharmonic not in current_notes:
                                current_notes[curr_note] = current_timestep

                    current_timestep += 1
                
                m_stream.write('midi', fp=(splitext(f_path)[0] + '-retrieved.midi'))
"""

"""
## META-INFO ABOUT THE FILEFORMAT
    ## files will look like this:
    ##                                                     dividor-space       dividor-space
    ## <----------------------timestep-1--------------------- >|<--timestep-2--->|<timestep-3>
    ## /---------------------note-1---------------------\ /n2\ V/n-1\ /n-2\ /n-3\V/n-1\ /n-2\
    ## [notenumber, 0-87]|[is it a new note? 0:no, 1:yes],..... .....,.....,..... .....,.....
    def create_processed_file_v2(self, f_path):
        if isfile(f_path) and (splitext(f_path)[1] == '.midi' or splitext(f_path)[1] == '.mid'):
            # stream of notes flattened and sorted
            m_stream = music21.converter.parse(f_path).flat.sorted
            
            notes = {}

            # lower and upper bounds of stream
            # below / beyond the bounds there are just rests
            low_end = round(Decimal(str(m_stream.lowestOffset)) / self.twelfth)
            high_end = round(Decimal(str(m_stream.highestTime)) / self.twelfth) - low_end

            # initialization of the notecontainer for each step
            for i in range(0,high_end+1):
                notes[str(i)] = []

            for note in m_stream.notes:
                # converting both offset and duration
                ##### this float() could make a mess #####
                note_offset = int(round((Decimal(float(note.offset)) / self.twelfth) - low_end)) 
                note_duration = int(round(Decimal(float(note.duration.quarterLength)) / self.twelfth))
                
                # every pitch in the note (or chord) gets added
                # to the containers of each step of the duration
                # example:
                # C#: offset=>3.0,          duration=>0.5
                # --> 3.0/0.0833... = 36    0.5/0.0833... = 6
                # ---> notes["36"].append("C#"), notes["37"].append("C#"),
                # ---> ........ notes["41"].append("C#") - end

                for i in range(0, note_duration):
                    for elem in note.pitches:
                        print(elem)
                        # if elem is a note with only ONE name (contrary to C# <-> D-)
                        # then the string for the note in the regex is only that:
                        # i.e. D4
                        if elem.accidental == None or elem.accidental.modifier == "":
                            notes_for_regex = elem.nameWithOctave
                        # if elem can have two names, the note in the regex is (note1 | note2)
                        else:
                            notes_for_regex = "(" + elem.nameWithOctave + "|" + elem.getEnharmonic().nameWithOctave + ")"

                        # if there is at least one occurance of the note elem, then it won't be added
                        if not re.compile("[\s,\,]?" + notes_for_regex + "[\s,\,]?").search(" ".join(notes[str(note_offset + i)])):
                            # name of the note converted to its number-representation (i.e. A0 -> 0, ..., C8-> 87)
                            notenum = self.note_to_num(elem.simplifyEnharmonic().nameWithOctave)
                            # if the note was already playing before -> 0
                            # if the note is now played for the first time -> 1
                            first_played_note = "1" if i==0 else "0"

                            notes[str(note_offset + i)].append(notenum + "|" + first_played_note)

            # if there is nothing in the notecontainer
            # then only "rest" is written
            with open(splitext(f_path)[0] + ".txt", "w") as f:
                for i in range(0, len(notes)):
                    if len(notes[str(i)]) == 0:
                        f.write("r ")
                    else:
                        f.write(",".join(notes[str(i)]) + " ")
            
            return True
        return False

    def retrieve_midi_from_processed_file_v2(self, f_path):
        if isfile(f_path) and splitext(f_path)[1] == '.txt':
            # used for storing notes that are playing at the moment
            # of the timestep / assigned to the note is the startoffset
            # of the note
            currently_playing_notes = {}

            m_stream = music21.stream.Stream()
            m_stream.autoSort = True
            m_stream.insert(0, music21.instrument.Piano())

            with open(f_path, "r") as f:
                # list with all timesteps
                timesteps = f.readlines()[0].split(' ')
                # the last item will always be an empty String
                # so it can be removed
                del timesteps[-1]
                # this is used to count the amount of
                # timesteps already passed
                current_timestep = 0
                
                # iterate over every timestep
                for timestep_content in timesteps:
                    # note_nums from the timestep_content loop are temporarily saved here
                    notes_from_timestep_content = []

                    # iterate over every note tuple / the notes that should
                    # be playing at the moment, if they exist
                    if timestep_content != 'r':
                        for note_tuple in timestep_content.split(','):
                            
                            note_num = note_tuple.split('|')[0]
                            notes_from_timestep_content.append(note_num)

                            is_startnote = note_tuple.split('|')[1]
                            
                            # the note has already been playing
                            if note_num in currently_playing_notes:
                                # it is a new note, which means that the note was
                                # hit a second time:
                                # --> the note will be added to the stream
                                # --> the timestep saved in the currently-playing-dictionary
                                #     has to be updated
                                if is_startnote == "1":
                                    # the new note
                                    note = music21.note.Note(self.num_to_note(note_num))
                                    # note offset from the beginning is converted to quarter-time-form and set
                                    note.offset = float(Decimal(currently_playing_notes[note_num]) * self.twelfth)
                                    # note duration is calculated, converted to quarter-time-form and set
                                    delta_t = current_timestep - currently_playing_notes[note_num]
                                    note.duration = music21.duration.Duration(float(Decimal(delta_t) * self.twelfth))
                                    # inserted into stream
                                    m_stream.insert(note)

                                    # timestep is updated
                                    currently_playing_notes[note_num] = current_timestep
                            # the note has not been playing yet
                            else:
                                # added to currently-playing-dictionary
                                currently_playing_notes[note_num] = current_timestep
                    
                    # loop over all currently playing notes
                    # to find out which shouln't be playing
                    # anymore
                    for curr_note in currently_playing_notes.copy():
                        # if note shouldn't be playing
                        # it will be saved in the stream and
                        # deleted from the dictionary
                        if curr_note not in notes_from_timestep_content:
                            # the new note
                            note = music21.note.Note(self.num_to_note(curr_note))
                            # note offset from the beginning is converted to quarter-time-form and set
                            note.offset = float(Decimal(currently_playing_notes[curr_note]) * self.twelfth)
                            # note duration is calculated, converted to quarter-time-form and set
                            delta_t = current_timestep - currently_playing_notes[curr_note]
                            note.duration = music21.duration.Duration(float(Decimal(delta_t) * self.twelfth))
                            # inserted into stream
                            m_stream.insert(note)
                            # entry in dictionary deleted
                            del currently_playing_notes[curr_note]

                    current_timestep += 1
                
                m_stream.write('midi', fp=(splitext(f_path)[0] + '-retrieved.midi'))
    
    # different approach!
    ################################      something fishy going on, megalovania wasn't encoded correctly!
    def create_processed_file_v3(self, f_path):
        if isfile(f_path) and (splitext(f_path)[1] == '.midi' or splitext(f_path)[1] == '.mid'):
            # stream of notes flattened and sorted
            m_stream = music21.converter.parse(f_path).flat.sorted
            
            notes = {}

            # lower and upper bounds of stream
            # below / beyond the bounds there are just rests
            low_end = round(Decimal(str(m_stream.lowestOffset)) / self.twelfth)
            high_end = round(Decimal(str(m_stream.highestTime)) / self.twelfth) - low_end

            # initialization of the notecontainer for each step
            for i in range(0,high_end+1):
                notes[i] = []

            for note in m_stream.notes:
                # converting both offset and duration
                ##### this float() could make a mess #####
                note_offset = int(round((Decimal(float(note.offset)) / self.twelfth) - low_end)) 
                note_duration = int(round(Decimal(float(note.duration.quarterLength)) / self.twelfth))
                
                # every pitch (converted to its number-representation) in the note (or chord) 
                # and its 'status' (i.e. start (1) or end (0) of note)
                # gets added to the containers of the beginning and end of the duration
                # example:
                # 44: offset=>3.0,          duration=>0.5
                # --> 3.0/0.0833... = 36    0.5/0.0833... = 6
                # ---> notes[36].append((44, 1) & notes[41].append((44, 2)) - end

                for elem in note.pitches:

                    note_num = self.note_to_num(elem.nameWithOctave)
                    
                    # set the start of the note
                    # META-INFO:
                    # [[note1, note2, note3], [status1, status2, status3]]
                    split_notes_and_status = list(zip(*notes[note_offset]))
                    
                    # safety measure [1]: when the list in notes at note_offset is empty,
                    # the above expression returns an empty list
                    # --> the expression below would produce an index out of bounds exception 
                    notes_at_offset = split_notes_and_status[0] if len(split_notes_and_status) > 0 else []
                    
                    if note_num in notes_at_offset:
                        # if the note is already in the notecontainer at the startoffset of the note,
                        # then set the 'status' to 1 to indicate the note starts at that offset (i.e.
                        # closing of the note is overwritten)
                        # 
                        # if the note wasn't closed before (i.e. the same note started but never ended),
                        # the end of the note is implied before the note starts again 
                        notes[note_offset][notes_at_offset.index(note_num)][1] = 1
                    else:
                        # otherwise it just gets added
                        notes[note_offset].append([note_num, 1])

                    # set the end of the note
                    
                    # safety measure: if there is a note with offset 0 and duration 0, this would cause
                    # wrong behavior if not checked
                    end_offset = note_offset+note_duration-1 if note_offset+note_duration-1 >= 0 else 0
                    
                    # for explanation see first 'split_notes_and_status'
                    split_notes_and_status = list(zip(*notes[end_offset]))
                    
                    # safety measure: see safety measure [1]
                    notes_at_offset = split_notes_and_status[0] if len(split_notes_and_status) > 0 else []
                    # here there is no need to check if, in case note_num already is in the container,
                    # the 'status' is 0, as there is no way it could have been set to 1
                    # this has to do with the fact the stream of notes is sorted 
                    if not (note_num in notes_at_offset):
                        notes[end_offset].append([note_num, 0])

            # if there is nothing in the notecontainer
            # then only "r" (= rest) is written
            with open(splitext(f_path)[0] + ".txt", "w") as f:
                for i in range(0, len(notes)):
                    if len(notes[i]) == 0:
                        f.write("r ")
                    else:
                        complete_string = ""
                        
                        for t in notes[i]:
                            complete_string += str(t[0]) + "|" + str(t[1]) + ","
                        complete_string = complete_string[:-1] + " "

                        f.write(complete_string)
            
            return True
        return False

    def retrieve_midi_from_processed_file_v3(self, f_path):
        # note_num in number-representation, offset & duration in int-representation
        def add_to_stream(stream, note_num, start_offset, end_offset):
            # the new note
            note = music21.note.Note(self.num_to_note(note_num))
            # note offset from the beginning is converted to quarter-time-form and set
            note.offset = float(Decimal(start_offset) * self.twelfth)
            # note duration is calculated, converted to quarter-time-form and set
            delta_t = end_offset - start_offset
            note.duration = music21.duration.Duration(float(Decimal(delta_t) * self.twelfth))
            # inserted into stream
            stream.insert(note)

        if isfile(f_path) and splitext(f_path)[1] == '.txt':
            # used for storing notes that are playing at the moment
            # of the timestep / assigned to the note is the startoffset
            # of the note
            currently_playing_notes = {}

            m_stream = music21.stream.Stream()
            m_stream.autoSort = True
            m_stream.insert(0, music21.instrument.Piano())

            with open(f_path, "r") as f:
                # list with all timesteps
                timesteps = f.readlines()[0].split(' ')
                # the last item will always be an empty String
                # so it can be removed
                del timesteps[-1]
                # this is used to count the amount of
                # timesteps already passed
                current_timestep = 0
                # iterate over every timestep
                for timestep_content in timesteps:

                    # iterate over every note tuple
                    if timestep_content != 'r':
                        for note_tuple in timestep_content.split(','):
                            note_num = int(note_tuple.split('|')[0])
                            status = int(note_tuple.split('|')[1])
                            
                            # if the note has been playing before and a new note_tuple with the same note 
                            # is in timestep_content, the note will be added to the stream regardless of the status 
                            if note_num in currently_playing_notes:
                                add_to_stream(m_stream, note_num, currently_playing_notes[note_num], current_timestep)

                            if status == 1:
                                # update timestep if the note should be played
                                currently_playing_notes[note_num] = current_timestep
                            else:
                                # or else delete entry in dictionary
                                if note_num in currently_playing_notes:
                                    del currently_playing_notes[note_num]
                    current_timestep += 1

                # if there are any notes that haven't been closed at the end of the song,
                # they will be closed here
                for note in currently_playing_notes:
                    add_to_stream(m_stream, note, currently_playing_notes[note], current_timestep)
                
                m_stream.write('midi', fp=(splitext(f_path)[0] + '-retrieved.midi'))
"""