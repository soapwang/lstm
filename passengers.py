import codecs
import numpy as np


def load_data(filename):
    with codecs.open(filename, "r", "utf-8") as f:
        lines = f.readlines()
        data = []
        for line in lines:
            if ',' in line:
                _split = line.strip().split(",")
                data.append(float(_split[1]))
        return data

        
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataset = scaler(dataset)
    dataX, dataY = [], []
    if look_back == 1:
        dataX = dataset[:-1]
        dataY = dataset[1:]
    elif look_back > 1:
        dataY = dataset[look_back:]
        for i in range(len(dataset) - look_back):
            d = []
            for j in range(look_back):
                d.append(dataset[i + j])
            dataX.append(d)
    return np.array(dataX, dtype=np.float32), dataY


# scale the element in a to (0,1)
def scaler(a):
    max_value = max(a)
    min_value = min(a)
    a = [(e - min_value) / (max_value - min_value) for e in a]
    return a