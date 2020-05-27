import csv
import numpy as np



def CSV_data(csv_file):

    b = np.zeros(shape=(100, 8, 8, 3), dtype=np.float32)
    for s in range(100):
        csv_filename = csv_file + str(s)
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
  
            i = 0
            j = 0

            for row in reader:
                a = np.array(row)

                b[s][i][j] = a
                j += 1

                if j == 8:
                    i += 1
                    j = 0
                    continue
    print('successful')
    return b



def CSV_data_test(csv_file):

    b = np.zeros(shape=(7, 8, 8, 3), dtype=np.float32)
    for s in range(70, 77):
        csv_filename = csv_file + str(s)
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)

            i = 0
            j = 0

            for row in reader:
                a = np.array(row)

                b[s-70][i][j] = a
                j += 1

                if j == 8:
                    i += 1
                    j = 0
                    continue
    print('successful')
    return b


csv_file = 
a = CSV_data(csv_file)
b = CSV_data_test(csv_file)
print(a.shape)
print(b.shape)
print(type(a))
