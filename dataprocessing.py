import music21
from os.path import isdir, isfile, join, splitext, basename, curdir
from os import listdir
from decimal import Decimal
import numpy as np
import random # may not be random enough

class DataProcessor():

    def __init__(self, dir_path, version=1):
        # the two versions of file-processing are described in
        # their respective functions (i.e. create_processed_file_v1 and create_processed_file_v2)

        print("""
        WARNING: The processed midi-files do not save on which version of file-processing they were created.
        To make sure the files are processed with the version you need, it is best to delete them and create new ones.
        A feature to handle this internally will come soon.
        """)

        self.version = version

        # declaration for ease of use later
        self.twelfth = Decimal("0.083333333333333333333333333333333333333") # 1 / 12
        self.lowest_piano_note = music21.note.Note("A0")
        self.highest_piano_note = music21.note.Note("C8")

        # build conversiondict for notes to nums, needed later
        self.notes_to_num_dict, self.num_to_notes_dict = self.make_conversion_dictionaries()

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
            if not splitext(file)[0] + ".txt" in complete_filelist:
                # if not possible, the filename is removed from 'files'
                if self.create_processed_file(join(self.dir_path, file)):
                    print("File '" + splitext(file)[0] + ".txt' was created...")
                    created_files_counter += 1
                else:
                    files.remove(file)

        if len(files) == 0:
            raise FileNotFoundError("There are no fitting files in this directory...")
        else:
            self.files = files
            print(str(created_files_counter) + (" new file was created" if created_files_counter == 1 else " new files were created"))
            print("Finished!")

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
        # [features]        number of notes in a single timestep --> always 88 

        ############# FAILSAFE
        # --> GPU-memory capabilities aren't exceeded
        # if the computed batch_size is greater than LIMIT,
        # then the data is split into three sub-arrays of 
        # about the same size, but where each array-length is
        # less than LIMIT. These sub-arrays are yielded seperately,
        # only after the last sub-array has been yielded a new song
        # is loaded

        LIMIT = 3

        remainder = []

        while True:
            if len(remainder) == 0:
                filename = splitext(random.choice(self.files))[0]
                music_data = self.load_processed_file(filename)
                #music_data = self.load_processed_file("megalovania")

                if len(music_data) < sequence_length:
                    sequence_length = len(music_data)

                # batch_size is number of shifts of the training data array + 1
                batch_size = len(music_data) - sequence_length + 1

                x_train = np.zeros((batch_size, sequence_length, 88)) # shape of training data is (batch_size, timesteps, features)
                y_train = np.zeros((batch_size, 88))

                for i in range(batch_size): 
                    for j in range(sequence_length):
                        for note in music_data[i+j]:        
                            ## a note can be in three states:
                            #### version 1
                            # 0.0 : note not playing
                            # 0.5 : note playing, but has been playing before
                            # 1.0 : note playing and hasn't been playing before
                            #### version 2
                            # 0.0 : do nothing
                            # 0.5 : stop note
                            # 1.0 : start note
                            x_train[i][j][note[0]] = 0.5 if note[1] == 0 else 1.0
                    if i == batch_size-1:
                        y_train[i] = np.zeros(88)
                    else:
                        for note in music_data[i+sequence_length]:
                            y_train[i][note[0]] = 0.5 if note[1] == 0 else 1.0
                
                if batch_size > LIMIT:
                    remainder = list(zip(np.array_split(x_train, np.ceil(len(x_train)/LIMIT)), np.array_split(y_train, np.ceil(len(y_train)/LIMIT))))
                    x_train, y_train = remainder.pop()

            else:
                x_train, y_train = remainder.pop()
                
            yield x_train, y_train
                    
                    
    def train_generator_with_padding(self, sequence_length):
        pass

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

    # META-INFO ABOUT THE STRUCTURE OF THE LOADED FILE
    # [  
    #   [ [note, curr_playing], [note, curr_playing] ],                      <- timestep 1
    #   [ [note, curr_playing] ],                                            <- timestep 2
    #   [ [note, curr_playing], [note, curr_playing], [note, curr_playing] ],<- timestep 3
    # ]
    def load_processed_file(self, f_name):
        full_path = join(self.dir_path, f_name + ".txt")
        if isfile(full_path):
            notelist = []
            with open(full_path, "r") as f:
                # data is divided into timesteps
                data = f.readlines()[0].split(" ")
                if data[-1] == "":
                    del data[-1]
                for timestep in data:
                    notes = timestep.split(",")
                    processed_notes = []
                    # each note in a timestep is added
                    for note in notes:
                        if note != "r":
                            processed_notes.append([int(note.split("|")[0]), int(note.split("|")[1])])
                    notelist.append(processed_notes)
            return notelist
        else:
            print("File with path '" + full_path + "' could not be loaded...")
            return None

    def create_processed_file(self, f_path):
        if self.version == 1:
            return self.create_processed_file_v1(f_path)
        elif self.version == 2:
            return self.create_processed_file_v2(f_path)
        else:
            raise Exception("Version '" + self.version + "' does not exist...")

    def retrieve_midi_from_processed_file(self, f_path):
        if self.version == 1:
            self.retrieve_midi_from_processed_file_v1(f_path)
        elif self.version == 2:
            self.retrieve_midi_from_processed_file_v2(f_path)
        else:
            raise Exception("Version '" + self.version + "' does not exist...")

    

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

    # shortcut to convert from note to number
    def note_to_num(self, note):
        return self.notes_to_num_dict[note]
    # shortcut to convert from number to note
    def num_to_note(self, num):
        return self.num_to_notes_dict[str(num)]















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