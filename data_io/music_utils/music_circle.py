import numpy as np

# The relative importance of each of these components in determining the similarity
# of two different notes can be adjusted by changing the diameters of the chroma circle and
# the circle of fifths.

# CHROMA CIRCLE contains the position of each note in the chroma circle
# 1 indicates the first note in the chroma circle, in this case G
# G-G#-A-A#-B-C-C#-D-D#-E-F-F#
CHROMA_CIRCLE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
RADIUS_CHROMA = 1

# CIRCLE_FIVE contains the position of each note in the circle of fifths
CIRCLE_FIVE = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]
RADIUS_C5 = 1


def get_chroma_coords(note):
    # find the angle in the chroma circle
    angle = (CHROMA_CIRCLE[note] - 1) * (360/12)
    angle_rad = np.deg2rad(angle)

    # compute the (x,y) coordinates for the chroma circle
    x = RADIUS_CHROMA * np.sin(angle_rad)
    y = RADIUS_CHROMA * np.cos(angle_rad)
    return x, y


def get_circle5_coords(note):
    # find the angle in the chroma circle
    angle = (CIRCLE_FIVE[note] - 1) * (360/12)
    angle_rad = np.deg2rad(angle)

    # compute the (x,y) coordinates for the chroma circle
    x = RADIUS_C5 * np.sin(angle_rad)
    y = RADIUS_C5 * np.cos(angle_rad)
    return x, y