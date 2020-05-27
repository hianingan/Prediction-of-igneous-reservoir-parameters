import csv
import numpy as np

csv_name = 



def CSV_label(csv_name):

    with open(csv_name, 'r') as f:
        reader = csv.reader(f)

        n = 0
        b = np.zeros(shape=(100, 1), dtype=np.float32)

        for row in reader:
            if n <= 99:
                b[n] = row
            n += 1
        print('successful')
        return b


def CSV_label_test(csv_name):

    with open(csv_name, 'r') as f:
        reader = csv.reader(f)

        n = 0
        b = np.zeros(shape=(39, 1), dtype=np.float32)

        for row in reader:
            if n >= 500:
                b[n - 500] = row
            n += 1
        print('successful')
        return b


a = CSV_label(csv_name)
b = CSV_label_test(csv_name)
print(b.shape)
print(type(a))
print(a)
