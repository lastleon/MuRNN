from dataprocessing import DataProcessor
from decimal import Decimal
import music21

d = DataProcessor()
d.process_files("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")

d.retrieve_midi_from_processed_file_v2("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\megalovania.txt")

#d.retrieve_midi_from_processed_file("D:\\leont\\Documents\\Schule\\W-Seminar\\Serenata.txt")

#n = music21.note.Note("B#4")

#print(n.pitch.simplifyEnharmonic().nameWithOctave)

#music21.midi.translate.midiFilePathToStream("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\Bohemian_Rhapsody_for_Piano.mid").show('text')


#s = music21.midi.translate.midiFilePathToStream("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\bruh_besser.mid")
"""
n = music21.note.Note("D4")
next = music21.note.Note(n.pitch.name + "#" + str(n.pitch.octave))
next.pitch.simplifyEnharmonic(inPlace=True)
print(next.pitch.nameWithOctave)
"""

