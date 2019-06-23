import music21
from os.path import isdir, isfile, join, splitext, basename, curdir
from os import listdir
from decimal import Decimal
import re

class DataProcessor():

    def __init__(self): 
        # declaration for ease of use later
        self.twelfth = Decimal("0.083333333333333333333333333333333333333") # 1 / 12
        self.lowest_piano_note = music21.note.Note("A0")
        self.highest_piano_note = music21.note.Note("C8")

        # build conversiondict for notes to nums, needed later
        self.notes_to_num_dict, self.num_to_notes_dict = self.make_conversion_dictionaries()

    def process_files(self, dir_path):
        # check if path to directory is valid
        if isdir(dir_path):
            # files are saved here
            files = []

            complete_filelist = listdir(dir_path)

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
                    if not self.create_processed_file_v2(join(dir_path, file)):
                        files.remove(file)
                    else:
                        print("File '" + splitext(file)[0] + ".txt' was created...")
                        created_files_counter += 1

            if len(files) == 0:
                raise FileNotFoundError("There are no fitting files in this directory...")
            else:
                self.files = files
                print(str(created_files_counter) + (" new file was created" if created_files_counter == 1 else " new files were created"))
                print("Finished!")

    def generator(self):
        while True:
            pass # to be continued
    
    # there is going to be some loss of information:
    # notes played by different instruments at the same time
    # are going to be considered as only one note

    

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