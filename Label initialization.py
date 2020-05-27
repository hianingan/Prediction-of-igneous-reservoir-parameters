import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv


xls_name = 
root = 
file_name = 


def make_tensor(xls_name, sheet_name):

    df = pd.read_excel(xls_name, sheet_name=sheet_name)
    a = []
    for i in range(539):
        data = df.iloc[i, 2]
        data = round(data, 3)
        a.append(data)
    b = np.array(a, dtype=float)
    return b


def load_csv(root, filename, tensor):

    if not os.path.exists(os.path.join(root, filename)):
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            csv_write = csv.writer(f)
            for i in range(539):
                print(i)
                csv_write.writerow([tensor[i]])
            print(filename)


def main():
    a = make_tensor(xls_name, '一')
    print(a)
    load_csv(root, file_name, a)


if __name__ == '__main__':
    main()
