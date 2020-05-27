import numpy as np
import pandas as pd
import random
import os
import csv



xls_name = 
root = 



def make_tensor(xls_name, sheet_name, random=0):

    df = pd.read_excel(xls_name, sheet_name=sheet_name)
    a = []
    b = []
    c = []
    for i in range(64):
        data = df.iloc[i, 0]
        data = round(data, 3)
        a.append(data)
    for j in range(64):
        data = df.iloc[j, 1] + rand(6, 0.001, 0.009)
        data = round(data, 3)
        b.append(data)
    for k in range(64):
        data = df.iloc[k, 2] + rand(6, 0.001, 0.009)
        data = round(data, 3)
        c.append(data)
    d = np.array([a, b, c], dtype=float)
    return d



def load_csv(root, filename, tensor):

    if not os.path.exists(os.path.join(root, filename)):
        with open(os.path.join(root, filename), mode='w', newline='') as f:
            csv_write = csv.writer(f)
            for i in range(64):
                csv_write.writerow([tensor[0][i], tensor[1][i], tensor[2][i]])
            print(filename)


def rand(seed, min, max):

    random.seed = seed
    a = random.uniform(min, max)
    return a


def main():
    for i in range(77):
        sheet_name = '序号' + str(i + 1)
        file_name = str(i + 462)
        a = make_tensor(xls_name, sheet_name)
        load_csv(root, file_name, a)


if __name__ == '__main__':
    main()
