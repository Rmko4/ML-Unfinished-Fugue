from midi_to_5d_vec import midi_to_5d_vec
import pandas as pd
import numpy as np

IGNORE_PAUSES = True
IMPORT_FILE = 'F.txt'

def generate_6d_train_vectors():
    midi: DataFrame = pd.read_csv(IMPORT_FILE, sep="\t", header=None)
    midi = midi.to_numpy()
    midi = midi.T

    # Only select first voice (for now) and discard last 16 midi notes
    midi = midi[0, :-16]

    train_vectors = []

    #@TODO: finish for loop
    duration = 0
    for i in range(midi.size - 1):
        if midi[i] == midi[i + 1]:
            duration += 1
        else:
            # create 6D vector 
            duration = 0



            

    

if __name__ == '__main__':
    generate_6d_train_vectors()