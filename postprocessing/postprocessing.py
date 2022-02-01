

from unittest.util import _MAX_LENGTH
from data_io.vector_encoders import OVE_OUT, OutputVectorEncoderMC
import os
import random
import sys
from typing import List

import copy

import numpy as np

# Quick fix to be able to import data_io module from the parent folder
sys.path.append(os.path.join(sys.path[0], '..'))


# raise probabilities to this power to increse high probabilities and reduce smaller ones
POWER_PROBABILITIES = 2
#Maximum length possible
MAX_LENGTH = 20

#probability adaptation for possible length, missing key means probability of 0
LENGTH_PROBABILITIES = {1:2,  2:0.17, 4:0.75, 6:8, 8:1.2, 10:3, 12: 1.4, 14:7, 16:0.3, 18:1, 20:1}

#probability adaptation for possible starting positions within the measure
#missing key means that position does not allow new note
MEASURE_POSITION_ADAPTATIONS = {0:0.55, 2:23, 3:14, 4:1.3, 6:5,7:6, 8:0.6, 10:12 ,11:15,12:1, 14:7 , 15:33}

#Differences voices probability adaptation
#non existion keys equal 1
DIFFERENCES_ADAPTATION = {0:0.4, 1:0.05, 2:0.7, 3:2.3, 4:2.4,5:1.3, 6:1,7:2.1,8:1.6, 9:2.28, 10: 0.8, 11:0.3, 12: 1, 13:0.2, 14:0.3,15:2.1, 16:1.8, 17:0.9, 18:0.9,19:1.5, 21:1.6,22:0.7, 23:0.2,24:1.5, 25:0.2, 26:0.7, 27:2, 28:1.4}

#Adapt difference in pitch compared to last note
DIFFERENCES_LAST_ADAPTATIONS = {1:1.4, 2:1.2,3:0.6, 4:0.8, 5:0.8,7:1.2 ,8:0.8, 10:0.9, 11:7}

#Adapt probabilities for each note (0 = C,1 = D_b,2 = D,3 = E_b,4 = E, 5 = f...,11 = B)
NOTE_ADAPTATIONS = {0:0.9,1:0.95, 3:0.95,4:1,5:0.7, 6:0.96, 11:1.1 }




class PostProcessorMC:
    def __init__(self, ove: OutputVectorEncoderMC, Y_prior: np.ndarray,
                 measure_length: int):
        '''
        ove: The output vector encoder of multi channel music
        Y_prior: The midi values that are used to identify what midi values
        have been generated upto this point.
        '''

        self.ove = ove

        # The length in symbols of a single measure.
        self.measure_length = measure_length
        # The current position within a measure in symbols (starts at 0).
        self.measure_idx = (Y_prior.shape[0] - 1) % self.measure_length

        # The duration for which the current note is playing (minimum 1).
        self.duration = np.empty((ove.n_channels))
        # The note prior to the current note.
        self.prev_note = np.empty((ove.n_channels))

        # A True or False mask indicating where the note changes
        # in the next time step for a each channel.
        change_indicator = np.diff(Y_prior, axis=0) != 0

        for channel in range(ove.n_channels):
            mask = change_indicator[:, channel]
            # The last time step at which the note changed
            last_change_idx = np.where(mask)[0][-1]

            # The previous note is still played at index last_change_idx
            self.prev_note[channel] = Y_prior[last_change_idx, channel]

            # The new note starts at the following index, therefore the
            # duration is minus one.
            self.duration[channel] = Y_prior.shape[0] - last_change_idx - 1

        # What I need as an additional input for the function "post_process_output" is:
        # -How long is the current note already playing?
        # -What was the pitch of the note before that?
        # -What is the current position in the measure?

    def __call__(self, y: OVE_OUT, *args) -> np.ndarray:
        '''
        This function is called when the object is called.
        Here should be the main body of the post processing functionality.
        Any additional keyword arguments that are passed on can be ignored.
        '''
        # Initially go to the next position in the measure or into a new measure
        self.forward_measure()

        midi_notes = []
        output_vectors = []

        for channel in range(self.ove.n_channels):
            if self.ove.flatten:
                output_vectors.append(
                    self.ove.output_vector_channel(y, channel)[0])
            else:
                output_vectors.append(y[channel][0])

        for i  in range(len(output_vectors)):
            output_vectors[i] = self.output_to_probability_vector(output_vectors[i])

            if self.prev_note[i] == 0:
                index_last_note = 0 
            else:
                index_last_note = int(1 + self.prev_note[i] - self.ove.note_min[i])
            #adapt length probabilities
            try:
                output_vectors[i][index_last_note] *= LENGTH_PROBABILITIES[int(self.duration[i])]
            except KeyError:
                #last note MUST be repeated
                output_vectors[i] = [ 1 if x == index_last_note else 0 for x in range(len(output_vectors[i]))]

            
            if self.duration[i] == MAX_LENGTH:
                #A different note must be played
                output_vectors[i][index_last_note] = 0 

            try:
                output_vectors[i][index_last_note] *= MEASURE_POSITION_ADAPTATIONS[self.measure_idx]
            except KeyError:
                #last note MUST be repeated
                output_vectors[i] = [ 1 if x == index_last_note else 0 for x in range(len(output_vectors[i]))]

            for difference in range(12):
                try:
                    adapt_prob = DIFFERENCES_LAST_ADAPTATIONS[difference]
                except KeyError:
                    continue
                if index_last_note - difference > 0:
                    output_vectors[i][index_last_note - difference ] *=adapt_prob
                if index_last_note + difference < len(output_vectors[i]):
                    output_vectors[i][index_last_note + difference ] *=adapt_prob                


            for j in range(1,len(output_vectors[i])):
                midi_mod_12 = (self.ove.note_min[i] + j +7 ) % 12
                try:
                    output_vectors[i][j] *= NOTE_ADAPTATIONS[midi_mod_12]
                except KeyError:
                    pass
            
                if sum(output_vectors[i]) == 0:
                    #edge case in which no note is good - a random one is taken
                    print("sum was 0")
                    output_vectors[i] = [1 for x in output_vectors[i] ]

            #make to probability vecor again
            output_vectors[i] = self.normalize_vector(output_vectors[i])


        combinations = []
        probability_combinations = []


        #assumes 4 voices, ugly code I know...
        #writes every possible combination 
        for i1  in range(len(output_vectors[0])):
            if output_vectors[0][i1] < 0.002:
                continue
            combination = [None,None,None,None]
            combination[0] = 0 if i1 == 0 else self.ove.note_min[0] + i1 -1
            for i2 in range(len(output_vectors[1])):
                if output_vectors[1][i2] < 0.002:
                    continue
                combination[1] = 0 if i2 == 0 else self.ove.note_min[1] + i2 -1
                for i3 in range(len(output_vectors[2])):
                    if output_vectors[2][i3] < 0.002:
                        continue
                    combination[2] = 0 if i3 == 0 else self.ove.note_min[2] + i3 -1
                    if output_vectors[0][i1] * output_vectors[1][i2] * output_vectors[2][i3] < 0.000005:
                        #combination is already too unlikely - skip last loop for computational efficiency
                        continue
                    for i4 in range(len(output_vectors[3])):
                        if output_vectors[3][i4] < 0.002:
                            continue
                        combination[3] = 0 if i4 == 0 else self.ove.note_min[3] + i4 -1
                        combinations.append(copy.deepcopy(combination))
                        probability_combinations.append(output_vectors[0][i1] * output_vectors[1][i2] * output_vectors[2][i3] * output_vectors[3][i4])


        for i in range(len(combinations)):
            prob = self.combination_probability(combinations[i])
            probability_combinations[i] *= prob

        
        #probability_combinations = self.normalize_vector(probability_combinations) # not necessary for random.choice

        midi_notes = random.choices(combinations, probability_combinations,k=1)[0]

        self.remember_state(midi_notes)
        return np.array(midi_notes)

        # I hope that helps - Remco

    def remember_state(self,midi_notes):
        for i in range(len(midi_notes)):
            if midi_notes[i] == self.prev_note[i]:
                self.duration[i] +=1
            else:
                self.duration[i] = 1
            self.prev_note[i] = midi_notes[i]
        
    def forward_measure(self):
        '''
        Go one position forward in the measure.
        0 is the first time step in a new measure.
        If self.measure_length is 16, then 15 is the last time step in that measure
        '''
        self.measure_idx = (self.measure_idx + 1) % self.measure_length

    # normalize vector such that it's sum is 1
    def normalize_vector(self,vector):
        sum = 0
        for x in vector:
            sum += x
        return [x / sum for x in vector]


    # create to probability vector
    # model output may contain sub-zero numbers
    # Takes each value to a power to increase differences between high and low probabilities
    def output_to_probability_vector(self,vector):
        # remove sub zero numbers
        vector = [x if x > 0 else 0 for x in vector]
        vector = self.normalize_vector(vector) 
        # take values to power to decrese entropy
        vector = [x ** POWER_PROBABILITIES for x in vector]
        return vector

    def combination_probability(self,combination):
        prob = 1
        for i in range(len(combination)):
            for j in range(i+1,len(combination)):
                if combination[i] == 0 or combination[j] == 0:
                    continue
                difference = abs(combination[i] - combination[j])
                try:
                    prob *= DIFFERENCES_ADAPTATION[difference]
                except KeyError:
                    #does not exist, so multiply by one
                    pass
        return prob

                











        # output_vectors is now a list of numpy arrays constructed from the
        # raw output (y: OVE_OUT).
        # Indexing the list will yield the probabilities (np.ndarray) of the
        # notes for channel 0 to ove.n_channels.
        # The probability at index 0 (ove.playing_idx) is the probability to
        # play no note.
        # ove.note_min[channel] will give the minimum note that can be played
        # by that channel. This is represented by index 1 (ove.notes_idx) for
        # that channel.
        # E.g. the probability that channel (or voice) 2 plays the lowest note
        # is output_vectors[2][1]. The midi value belonging to this probability
        # is ove.note_min[2].
        # The last value in the probability array will correspond to ove.note_max[channel]
        # The vocal range of notes is ove.note_range[channel].

        # Also note that in case of the MLP that the output_vectors for each channel
        # are an actual probability vector. They sum to one and are all positive
        # values, including whether to play or not. In case of linear and ridge
        # regression this is not the case.

        # The output should be a numpy array of length ove.n_channels,
        # containing the integers of the midi value.
