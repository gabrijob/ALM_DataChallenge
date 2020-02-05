import csv

entries = []

for k in range(3):
    with open('data/Xtr' + str(k) +'.csv', newline='') as xtr_file:
        xtr_rows = csv.reader(xtr_file, delimiter=' ')
        entries_k = []

        with open('data/Ytr' + str(k) + '.csv', newline='') as ytr_file:
            ytr_rows = csv.reader(ytr_file, delimiter=' ')
            i = 0
            for x_row, y_row in zip(xtr_rows, ytr_rows):
                if i > 0:
                    entries_k.append((x_row[0].split(',')[1], y_row[0].split(',')[1]))
                i = i + 1

            entries.append(entries_k)

    #for row in entries[k]:
    #    print (row)

print ("Number of classes:" + str(len(entries)))
for k in range(len(entries)):
    print ("Number of sequences for class " + str(k) + ": " + str(len(entries[k])))
