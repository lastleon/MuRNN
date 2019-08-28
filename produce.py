from dataprocessing import DataProcessor
from train import MuRNN
import argparse

parser = argparse.ArgumentParser(prog="MuRNN")

parser.add_argument("model_name",
                    type=str,
                    help="The name of the trained model saved in the models directory")
parser.add_argument("dataset_dir",
                    type=str,
                    help="The path to the dataset")
parser.add_argument("-song_length",
                    type=int,
                    default=200,
                    help="The song length (aka: number of predictions for notes the model does)")

parser.add_argument("-amount",
                    type=int,
                    default=1,
                    help="How many songs should be produced")

args = parser.parse_args()

model = MuRNN()

model.load_model(args.model_name, args.dataset_dir)

for _ in range(args.amount):
    DataProcessor.retrieve_midi_from_loaded_data(model.make_song(args.song_length), target_dir=("./models/" + args.model_name + "/songs/"))
