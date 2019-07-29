from dataprocessing import DataProcessor
from decimal import Decimal
import music21

d = DataProcessor()
d.process_files("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")

d.retrieve_midi_from_processed_file_v2("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\megalovania.txt")

