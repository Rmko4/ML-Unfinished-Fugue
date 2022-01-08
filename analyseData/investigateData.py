import matplotlib.pyplot as plt
import pandas as pd

notes = [[],[],[],[]]
mod12_notes = [[],[],[],[]]

f = open("data.txt", "r")
lines = f.read()
lines = lines.split("\n")
for i,line in enumerate(lines):
    lines[i] = line.split("\t")

for i in range(len(lines)):
    for j in range(4):
        notes[j].append((int)(lines[i][j]))
        if (int)(lines[i][j]) != 0:
            mod12_notes[j].append( (((int)(lines[i][j]) +8) % 12) + 1)
        else:
            mod12_notes[j].append(0)

f.close()

labels = ["C","D_b","D","E_b","E","F", "G_b","G","A_b","A","B_b","B"]

allInOne = [x for x in  [*mod12_notes[0], *mod12_notes[1],*mod12_notes[2],*mod12_notes[3]]]

#frequencies of specific notes
frequency_counts = [0 for j in range(12)]
uniqueNotes = [allInOne[i] for i in range(len(allInOne)-1) if allInOne[i] != 0 and allInOne[i] != allInOne[i+1]]
for note in uniqueNotes:
    frequency_counts[note-1] +=1

plt.bar(labels,frequency_counts)
plt.title("Frequency for each note in all voices")
plt.show()

#average length of each note
total_counts = [0 for j in range(12)]
average_length = [0 for j in range(12)]
length = 0
for i in range(len(allInOne)):
    if allInOne[i] == 0:
        continue
    length = 0
    while i+1 != len(allInOne) and allInOne[i] == allInOne[i+1]:
        length +=1
        i+=1
    average_length[allInOne[i]-1] += length
#divided by 16 since the sample frequency is 16 per bar
average_length = [(average_length[i] / frequency_counts[i] ) / 16 for i in range(12)]

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

#liklyhood a not-diminor is floowed by d-minor or not
D_minor = [1,3,5,6,8,10,11]
uniqueNotes = [allInOne[i] for i in range(len(allInOne)-1) if allInOne[i] != 0 and allInOne[i] != allInOne[i+1]]
Dminor_to_Dminor = 0
Dminor_to_NotDminor = 0
NotDminor_to_Dminor = 0
NotDminor_to_NotDminor = 0
for i in range(len(uniqueNotes)-1):
    if uniqueNotes[i] in D_minor and uniqueNotes[i+1] in D_minor:
        Dminor_to_Dminor += 1
    elif uniqueNotes[i] in D_minor and uniqueNotes[i+1] not in D_minor:
        Dminor_to_NotDminor += 1
    elif uniqueNotes[i] not in D_minor and uniqueNotes[i+1] in D_minor:
        NotDminor_to_Dminor += 1
    elif uniqueNotes[i] not in D_minor and uniqueNotes[i+1] not in D_minor:
        NotDminor_to_NotDminor += 1

totalFromDminorChanges = Dminor_to_Dminor + Dminor_to_NotDminor
totalFromNotDminorChanges = NotDminor_to_Dminor + NotDminor_to_NotDminor 

Dminor_to_Dminor /= totalFromDminorChanges
Dminor_to_NotDminor /= totalFromDminorChanges
NotDminor_to_Dminor /= totalFromNotDminorChanges
NotDminor_to_NotDminor /= totalFromNotDminorChanges

plt.title("Dminor / not Dminor notes transitions")
plt.bar(["From Dminor to Dminor","From Dminor to not Dminor","From not Dminor to Dminor","From not Dminor to not Dminor"],[Dminor_to_Dminor,Dminor_to_NotDminor,NotDminor_to_Dminor,NotDminor_to_NotDminor])
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
plt.bar([str(x) for x in range(26)],differences_counts[:26])
plt.show()


        

        


