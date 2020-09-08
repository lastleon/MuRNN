from dataprocessing import DataProcessor
from train import MuRNN
import argparse
from os.path import join

parser = argparse.ArgumentParser(prog="MuRNN")

parser.add_argument("model_path",
                    type=str,
                    help="The path to your trained model")
parser.add_argument("-song_length",
                    type=int,
                    default=200,
                    help="The song length (aka: number of predictions for notes the model does)")
parser.add_argument("-target_dir",
                    type=str,
                    default=None,
                    help="Specify the target directory for the midi-files,\ndefaults to *path to your model*/songs/")
parser.add_argument("-weights_filename",
                    type=str,
                    default="weights.hdf5",
                    help="Specify the weights-file to load for your model,\ndefaults to 'weights.hdf5'")
parser.add_argument("-amount",
                    type=int,
                    default=1,
                    help="Specify how many songs to produce")
parser.add_argument("-alpha",
                    type=float,
                    default=0.0,
                    help="Specify how creative the network can get")

args = parser.parse_args()

model = MuRNN()

model.load_model(args.model_path, weights_filename=args.weights_filename)


for _ in range(args.amount):
    DataProcessor.retrieve_midi_from_loaded_data(model.make_song(args.alpha, args.song_length), target_dir=args.target_dir if args.target_dir != None else join(args.model_path, "songs/"))
