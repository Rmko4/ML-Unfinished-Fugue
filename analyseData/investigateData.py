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
        if (int)(lines[i][j]) != 0:
            notes[j].append((int)(lines[i][j])+8)
            mod12_notes[j].append( (((int)(lines[i][j]) +8 ) % 12) + 1)
        else:
            mod12_notes[j].append(0)
            notes[j].append(0)

f.close()


labels = [ 'C',"D_b","D","E_b","E","F", "G_b","G","A_b","A","B_b","B"]

allInOne =  [*mod12_notes[0], *mod12_notes[1],*mod12_notes[2],*mod12_notes[3]]

#frequencies of specific notes
frequency_counts = [0 for j in range(12)]
uniqueNotes = [allInOne[i] for i in range(len(allInOne)-1) if allInOne[i] != 0 and allInOne[i] != allInOne[i+1]]
for note in uniqueNotes:
    frequency_counts[note-1] +=1

plt.bar(labels,frequency_counts)
plt.title("Frequency for each note in all voices")
plt.show()

#average length of each note
total_length = [0 for j in range(12)]
for i in range(len(allInOne)):
    if allInOne[i] == 0:
        continue
    length = 0
    while i+1 != len(allInOne) and allInOne[i] == allInOne[i+1]:
        length +=1
        i+=1
    total_length[allInOne[i]-1] += length
#divided by 16 since the sample frequency is 16 per bar
average_length = [(total_length[i] / frequency_counts[i] ) / 16 for i in range(12)]

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
plt.title("Difference in pitch compared to last note")
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

plt.title("Counts of differences between voices at same time")
plt.bar([str(x) for x in range(38)],differences_counts[:38])
plt.show()

#Lengths of difference between voices:
length_differences_counts = [0 for x in range(100)]
differences_0_1 = [ abs(v1-v2) for v1,v2 in zip(notes[0],notes[1]) if v1 != 0 and v2 != 0 ]
differences_0_2 = [ abs(v1-v2) for v1,v2 in zip(notes[0],notes[2]) if v1 != 0 and v2 != 0 ]
differences_0_3 = [ abs(v1-v2) for v1,v2 in zip(notes[0],notes[3]) if v1 != 0 and v2 != 0 ]
differences_1_2 = [ abs(v1-v2) for v1,v2 in zip(notes[1],notes[2]) if v1 != 0 and v2 != 0 ]
differences_1_3 = [ abs(v1-v2) for v1,v2 in zip(notes[1],notes[3]) if v1 != 0 and v2 != 0 ]
differences_2_3 = [ abs(v1-v2) for v1,v2 in zip(notes[2],notes[3]) if v1 != 0 and v2 != 0 ]

all_differences_voices = [*differences_0_1, *differences_0_2,*differences_0_3, *differences_1_2,*differences_1_3, *differences_2_3 ]
for i in range(len(all_differences_voices)-1):
    length = 0
    while  i +1 != len(all_differences_voices) and  all_differences_voices[i] == all_differences_voices[i+1]:
        length +=1
        i+=1
    length_differences_counts[all_differences_voices[i]] += length


plt.title("Average of length of differences between voices")
plt.bar([str(x) for x in range(38)],[ x / (y*16)  if y != 0 else 0 for x,y in zip(length_differences_counts,differences_counts) ][:38]) # since a measure is 16 
plt.show()



        


