from pickle import FALSE
from postprocessing import *

LEN_TIMESERIES = 10000 #in notes

first = True

#One Voice
if False:
    timeseries = []
    measure_pos = 0
    for x in range(LEN_TIMESERIES):

        note, length = mode_outpout_to_note([1 for x in range(12)],[1 for x in range(15)],measure_pos)
        measure_pos += length /16
        if measure_pos > 1:
            measure_pos -=1

        while length > 0:
            length -= 1/16
            timeseries.append(note)

    with open('pure_postprocessing_one_voice.txt', 'w') as f:
        for note in timeseries:
            
            if first:
                f.write((str)(note) + "\t0\t0\t0" )
                first =False
            else:
                f.write("\n" + (str)(note) + "\t0\t0\t0" )

#Two voices
if True:
    pos_voice1 = 0
    pos_voice2 = 0
    notes_v1 = []
    notes_v2 = []
    while len(notes_v1) < LEN_TIMESERIES or len(notes_v2) < LEN_TIMESERIES:
        other = None
        if pos_voice1 <= pos_voice2:
            if pos_voice2 > pos_voice1:
                other = notes_v2[-1]
            notes_other_voices = [ other,None,None]
            note, length = mode_outpout_to_note([1 for x in range(24)],[1 for x in range(15)],(pos_voice1 % 16) /16,notes_other_voices)
            pos_voice1 += length
            while length > 0:
                length -= 1/16
                notes_v1.append(note)
        else:
            if pos_voice1 > pos_voice1:
                other = notes_v1[-1]
            notes_other_voices = [ other,None,None]
            note, length = mode_outpout_to_note([1 for x in range(24)],[1 for x in range(15)],(pos_voice2 % 16) /16,notes_other_voices)
            pos_voice2 += length
            while length > 0:
                length -= 1/16
                notes_v2.append(note)

    while len(notes_v1) > LEN_TIMESERIES:
        notes_v1.pop()
    while len(notes_v2) > LEN_TIMESERIES:
        notes_v2.pop()



    with open('pure_postprocessing_two_voices.txt', 'w') as f:
        for i in range (LEN_TIMESERIES):
            
            if first:
                f.write((str)(notes_v1[i]) + "\t" + (str)(notes_v2[i]) + "\t0\t0")
                first =False
            else:
                f.write("\n" +(str)(notes_v1[i]) + "\t" + (str)(notes_v2[i]) + "\t0\t0")



