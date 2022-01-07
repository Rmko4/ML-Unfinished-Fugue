import numpy as np

#The relative importance of each of these components in determining the similarity
#of two different notes can be adjusted by changing the diameters of the chroma circle and
#the circle of fifths.

# CHROMA CIRCLE contains the position of each note in the chroma circle
# 1 indicates the first note in the chroma circle, in this case G
# G-G#-A-A#-B-C-C#-D-D#-E-F-F#
CHROMA_CIRCLE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
RADIUS_CHROMA = 1

# CIRCLE_FIVE contains the position of each note in the circle of fifths
CIRCLE_FIVE = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]
RADIUS_C5 = 1

# @TODO: to be determined
# The min and max MIDI values used in the training data
MIN_NOTE = 0
MAX_NOTE = 0

def get_chroma_coords(note):
    # find the angle in the chroma circle
    angle = (CHROMA_CIRCLE[note] - 1) * (360/12);
    angle_rad = np.deg2rad(angle)
    
    # compute the (x,y) coordinates for the chroma circle
    x = RADIUS_CHROMA * np.sin(angle_rad);
    y = RADIUS_CHROMA * np.cos(angle_rad);
    return x, y

def get_circle5_coords(note):
    # find the angle in the chroma circle
    angle = (CIRCLE_FIVE[note] - 1) * (360/12);
    angle_rad = np.deg2rad(angle)
    
    # compute the (x,y) coordinates for the chroma circle
    x = RADIUS_C5 * np.sin(angle_rad);
    y = RADIUS_C5 * np.cos(angle_rad);
    return x, y

def midi_to_5d_vec(midi_note):
    # Convert midi_note to an int [1,12]
    # 55 is MIDI value of note G3
    note  = (midi_note - 55) % 12 + 1

    chroma_x, chroma_y = get_chroma_coords(note)
    circle5_x, circle5_y = get_circle5_coords(note)

    # d is the distance (in semitones) of midi_note from A4 (69 in MIDI),
    # whose frequency is 440 Hz. fx is the frequency of the note
    d = midi_note - 69;
    fx = 2**(d/12) * 440;

    # the representation of pitch is scaled in such a way that a pitch
    # distance of 1 octave in the first dimension, is equal to the distance of
    # notes on the opposite sides on the chroma circle or the circle of fifths
    min_p = 2 * np.log2(2^((MIN_NOTE - 69)/12) * 440);
    max_p = 2 * np.log2(2^((MAX_NOTE - 69)/12) * 440);
    pitch = 2 * np.log2(fx) - max_p + (max_p - min_p)/2;

    return [pitch, chroma_x, chroma_y, circle5_x, circle5_y]

if __name__ == '__main__':
    midi_to_5d_vec()