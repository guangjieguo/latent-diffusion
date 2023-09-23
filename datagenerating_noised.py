import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip
from load_smallnorb import load_smallnorb
import random

# download smallnorb data
(train_images, train_labels),(test_images,test_labels) = load_smallnorb()

# Specify the names of the save files
filename = "dataset_noised_planes1500.data"

#extract plane images form train_images and test_labels
print("Extracting plane images form train_images and test_labels...")
N = 0
for i in range(len(train_labels)):
    if train_labels[i, 2] == 2:
        N = N + 1
for i in range(len(test_labels)):
    if test_labels[i, 2] == 2:
        N = N + 1
plane_images = np.zeros((N,96,96,2)).astype('uint8')
M = 0
for i in range(len(train_labels)):
    if train_labels[i, 2] == 2:
        plane_images[M,:,:,:] = train_images[i,:,:,:]
        M = M + 1
for i in range(len(test_labels)):
    if test_labels[i, 2] == 2:
        plane_images[M, :, :, :] = test_images[i, :, :, :]
        M = M + 1

#define a function adding noise
def addnoise(image, num):
    result = image
    int_list = list(range(9216))
    random_ints = random.sample(int_list, num)
    for i in random_ints:
        quotient, remainder = divmod(i, 96)
        random_int = random.randrange(256)
        result[quotient, remainder] = random_int
    return result

#size of noise each time
num = 1500

#create noised images
print("Creating noised images...")
plane_images_noised = np.zeros((N,96,96,4)).astype('uint8')
for i in range(N):
    image = np.zeros((96, 96)).astype('uint8')
    for k in range(96):
        for l in range(96):
            image[k,l] = plane_images[i, k, l, 0]
    plane_images_noised[i, :, :, 0] = addnoise(image, num)
for j in range(3):
    for i in range(N):
        image = np.zeros((96, 96)).astype('uint8')
        for k in range(96):
            for l in range(96):
                image[k, l] = plane_images_noised[i, k, l, j]
        plane_images_noised[i, :, :, j+1] = addnoise(image, num)

#merge noised images as an array for input and label
#input_data
print("Building training_set and testing_set...")
input_data = np.zeros((N*4,96,96,1)).astype('uint8')
M = 0
for j in range(4):
    for i in range(N):
        input_data[M, :, :, 0] = plane_images_noised[i, :, :, j]
        M = M + 1

#label_data
label_data = np.zeros((N*4,96,96,1)).astype('uint8')
M = 0
for i in range(N):
    label_data[i, :, :, 0] = plane_images[i, :, :, 0]
for j in range(3):
    for i in range(N):
        label_data[N+M, :, :, 0] = plane_images_noised[i, :, :, j]
        M = M + 1

# save noised planes_data
print("Saving data to file...")
with gzip.open(filename, 'w') as f:
    pickle.dump((input_data, label_data), f)

# show some noised images
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.1)
for i, ax in enumerate(axes.flat):
    image = input_data[i*N, :, :, 0]
    ax.imshow(image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
