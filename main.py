import csv
import kernel_ridge as krr
import numpy as np


def read_regular_training_files():
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

    return entries


def read_matrix_training_files():
    x_entries = []
    y_entries = []

    for k in range(3):
        with open('data/Xtr' + str(k) +'_mat100.csv', newline='') as xtr_file:
            xtr_rows = csv.reader(xtr_file, delimiter=' ')
            x_entries_k = []
            for x_row in xtr_rows:
                x_entries_k.append(x_row)
            
            x_entries.append(x_entries_k)            


        with open('data/Ytr' + str(k) + '.csv', newline='') as ytr_file:
            ytr_rows = csv.reader(ytr_file, delimiter=' ')
            y_entries_k = []
            i = 0
            for y_row in ytr_rows:
                if i > 0:
                    y_entries_k.append(y_row[0].split(',')[1])
                i = i +1

            y_entries.append(y_entries_k)
        
    return x_entries,y_entries

def read_matrix_test_files():
    test_entries = []

    for k in range(3):
        with open('data/Xte' + str(k) +'_mat100.csv', newline='') as xte_file:
            xte_rows = csv.reader(xte_file, delimiter=' ')
            test_entries_k = []
            for test_row in xte_rows:
                test_entries_k.append(test_row)

            test_entries.append(test_entries_k)

    return test_entries

def verify_training(x,y,weights):
    f_train = []
    for k in range (3):
        f_train.append(krr.test(x[k], weights[k]))

    train_res = []
    for k in range (3):
        k_res = []
        for sequence_res in f_train[k]:
            if (sequence_res > 0.5):
                k_res.append(1)
            else:
                k_res.append(0)

        train_res.append(k_res)
    
    y_array = np.array(y, np.int)
    
    #print (train_res[0])
    #print (y_array[0])

    precision = []
    for k in range(3):
        good = 0
        total = 0
        for train_res_u, y_res_u in zip(train_res[k], y_array[k]): 
            if (train_res_u == y_res_u):
                good = good + 1
            total = total +1
        precision.append(good/total)

    print(precision)


def write_results_file(results):
    with open('results.csv', 'w') as res_file:
        writer = csv.writer(res_file)
        row = ['Id', 'Bound']
        writer.writerow(row)

        id_res = 0
        file_rows = []
        for k in range(3):
            for result in results[k]:
                row = [id_res, result]
                #file_rows.append(row)
                writer.writerow(row)
                id_res = id_res + 1


def main():
    # Read the samples from the csv's
    x_entries, y_entries = read_matrix_training_files()
    
    #x = np.array(x_entries, dtype=np.float)
    #y = np.array(y_entries, dtype=np.int)
    
    #print(x)
    #print(y)

    print ("Number of classes:" + str(len(x_entries)))
    for k in range(len(x_entries)):
        print ("Number of sequences for class " + str(k) + ": " + str(len(x_entries[k])))

    # Train the kernels
    weights = []
    for k in range(3):
        weights.append(krr.regression(x_entries[k], y_entries[k]))

    #print (weights)
    

    # Verify training
    verify_training(x_entries, y_entries, weights)

    # Test
    test_entries = read_matrix_test_files()
    f = []
    for k in range (3):
        f.append(krr.test(test_entries[k], weights[k]))

    #print (f)

    # Getting the Bound/Unbound values
    test_res = []
    for k in range (3):
        k_res = []
        for sequence_res in f[k]:
            if (sequence_res > 0.5):
                k_res.append(1)
            else:
                k_res.append(0)

        test_res.append(k_res)

    #print(test_res)
    
    
    write_results_file(test_res)
    

if __name__== "__main__":
    main()

