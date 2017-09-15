import csv
import cv2 
import numpy as np


lines = []

with open('./data/driving_log.csv') as cf:
    reader = csv.reader(cf)
    for line in reader:
        lines.append(line)

images = []
measurments = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path, )
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)

X_train = np.array(images)
y_train = np.array(measurments)


import matplotlib.pyplot as plt
degs = len(set(y_train))
print("degs: ", degs)

print(y_train)
hist, bin_edges = np.histogram(y_train, bins=degs)
center = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.bar(center, hist, align='center')
plt.show()


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D

def run_nn():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split = 0.2, shuffle=True, nb_epoch = 2)
    model.save('model.h5')
