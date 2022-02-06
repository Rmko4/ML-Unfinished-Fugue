import matplotlib.pyplot as plt
import pandas as pd
import sys

notes = [[],[],[],[]]
mod12_notes = [[],[],[],[]]

f = open( sys.argv[1], "r")

lines = f.read()
lines = lines.split("\n")
for i,line in enumerate(lines):
    lines[i] = line.split("\t")

for i in range(len(lines)):
    for j in range(4):
        if len(lines[i]) == 1:
            continue
        
        try:
            if (int)(lines[i][j]) != 0:
                notes[j].append((int)(lines[i][j])+8)
                mod12_notes[j].append( (((int)(lines[i][j]) +8 ) % 12) + 1)
            else:
                mod12_notes[j].append(0)
                notes[j].append(0)
        except ValueError:
            print(lines[i])

f.close()


labels = [ 'C',"D_b","D","E_b","E","F", "G_b","G","A_b","A","B_b","B"]

allInOne =  [*mod12_notes[0], *mod12_notes[1],*mod12_notes[2],*mod12_notes[3]]

#length of notes distrubution
length_frequencies = [0 for x in range(10000)]
i= 0 
while i != len(allInOne):
    if allInOne[i] == 0:
        i+=1
        continue
    length = 1
    while i+1 != len(allInOne) and allInOne[i] == allInOne[i+1]:
        length +=1
        i+=1
    i+=1
    length_frequencies[length] +=1


plt.bar([ x for x in range(1,21)], length_frequencies[1:21] , width= 0.95) 
plt.title("Frequency of note lengths \n Ridge Regression without post-processing")
plt.xlabel("Length in number of time-steps")
plt.ylabel("n Occurrences")
plt.xticks([ x for x in range(1,21)],[ str(x) for x in range(1,21)])
plt.show()

#note start position
start_positions = [0 for x in range(16)]
i= 0 
for channel in notes:
    for i in range(1,len(channel)):
        if channel[i-1] != channel[i]:
            start_positions[i % 16] +=1


plt.bar([ x / 16 for x in range(16)], start_positions , width= 0.05) 
plt.title("Frequency of note starting positions within measure \n Ridge Regression without post-processing")
plt.xlabel("Measure position")
plt.ylabel("n Occurrences")
measure_positions = ["0","1/8", "1/4", "3/8", "1/2", "5/8", "3/4", "7/8" ]
plt.xticks([ x / 16 for x in range(0,16,2)], measure_positions)
plt.show()


#frequencies of specific notes
frequency_counts = [0 for j in range(12)]
uniqueNotes = [allInOne[i] for i in range(len(allInOne)-1) if allInOne[i] != 0 and allInOne[i] != allInOne[i+1]]
for note in uniqueNotes:
    frequency_counts[note-1] +=1

plt.bar(labels,frequency_counts)
plt.title("Frequency for each note in all voices \n Ridge Regression without post-processing")
plt.xlabel("Pitch")
plt.ylabel("n Occurrences")
plt.show()

#average length of each note
total_length = [0 for j in range(12)]
i= 0 
while i != len(allInOne):
    if allInOne[i] == 0:
        i+=1
        continue
    length = 1
    while i+1 != len(allInOne) and allInOne[i] == allInOne[i+1]:
        length +=1
        i+=1
    total_length[allInOne[i]-1] += length
    i+=1



if False:
    #divided by 16 since the sample frequency is 16 per bar
    average_length = [0 if frequency_counts[i] == 0  else total_length[i] / frequency_counts[i] for i in range(12)]

    plt.bar(labels,average_length)
    plt.title("Average length for each note")
    plt.show()

    #liklyhood a note starts a bar
    onlyBeginning = [ x for i, x in enumerate(allInOne) if i % 16 == 0]
    bar_starting_counts = [0 for j in range(12)]
    for note in onlyBeginning:
        if note != 0:
            bar_starting_counts[note-1] +=1

    normalized_bar_starting_counts = [bar_starting_counts[i] / frequency_counts[i]  for i in range(12)]

    plt.title("Normalized frequency a measure starts with a note")
    plt.bar(labels,normalized_bar_starting_counts)
    plt.show()

#Differences compared to last note
differences_counts = [0 for j in range(50)]
for i in range(len(allInOne) -1):
    if allInOne[i] != 0 and allInOne[i+1] != 0 and allInOne[i] != allInOne[i+1]:
        differences_counts[allInOne[i] - allInOne[i+1]] +=1

plt.bar([str(x) for x in range(1,14)],differences_counts[1:14])
plt.title("Difference in pitch compared to last note \n Ridge Regression without post-processing")
plt.xlabel("Pitch difference")
plt.ylabel("n Occurrences")
plt.show()

#Difference notes of different voices:
differences_counts = [0 for x in range(100)]
for timepoint in range(len(mod12_notes[0])):
    breaks = [notes[0][timepoint] == 0, notes[1][timepoint] == 0, notes[2][timepoint] == 0, notes[3][timepoint] == 0]
    if not breaks[0] and not breaks[1]:
        differences_counts[abs(notes[0][timepoint] - notes[1][timepoint])] +=1
    if not breaks[0] and not breaks[2]:
        differences_counts[abs(notes[0][timepoint] - notes[2][timepoint])] +=1
    if not breaks[0] and not breaks[3]:
        differences_counts[abs(notes[0][timepoint] - notes[3][timepoint])] +=1
    if not breaks[1] and not breaks[2]:
        differences_counts[abs(notes[1][timepoint] - notes[2][timepoint])] +=1
    if not breaks[1] and not breaks[3]:
        differences_counts[abs(notes[1][timepoint] - notes[3][timepoint])] +=1
    if not breaks[2] and not breaks[3]:
        differences_counts[abs(notes[2][timepoint] - notes[3][timepoint])] +=1

plt.title("Differences between voices at same time  \n Bach's 14th Fugue")
plt.bar([str(x) for x in range(38)],differences_counts[:38])
plt.xlabel("Pitch difference")
plt.ylabel("n Occurrences")
plt.show()

if False:
    #Lengths of difference between voices:
    length_differences_counts = [0 for x in range(100)]
    differences_0_1 = [ abs(v1-v2) for v1,v2 in zip(notes[0],notes[1]) if v1 != 0 and v2 != 0 ]
    differences_0_2 = [ abs(v1-v2) for v1,v2 in zip(notes[0],notes[2]) if v1 != 0 and v2 != 0 ]
    differences_0_3 = [ abs(v1-v2) for v1,v2 in zip(notes[0],notes[3]) if v1 != 0 and v2 != 0 ]
    differences_1_2 = [ abs(v1-v2) for v1,v2 in zip(notes[1],notes[2]) if v1 != 0 and v2 != 0 ]
    differences_1_3 = [ abs(v1-v2) for v1,v2 in zip(notes[1],notes[3]) if v1 != 0 and v2 != 0 ]
    differences_2_3 = [ abs(v1-v2) for v1,v2 in zip(notes[2],notes[3]) if v1 != 0 and v2 != 0 ]

    all_differences_voices = [*differences_0_1, *differences_0_2,*differences_0_3, *differences_1_2,*differences_1_3, *differences_2_3 ]
    i = 0
    while i+1 < len(all_differences_voices) :
        length = 1
        while  i +1 != len(all_differences_voices) and  all_differences_voices[i] == all_differences_voices[i+1]:
            length +=1
            i+=1
        length_differences_counts[all_differences_voices[i]] += length
        i+=1


    plt.title("Average of length of differences between voices")
    plt.bar([str(x) for x in range(38)],[ x / (y*16)  if y != 0 else 0 for x,y in zip(length_differences_counts,differences_counts) ][:38]) # since a measure is 16 
    plt.show()



        


