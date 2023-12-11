from keras.datasets import cifar10, cifar100
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
import cv2
import requests
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

def show_samples(data, labels): 
    plt.subplots(figsize=(10, 10)) 
    for i in range(12): 
        plt.subplot(3, 4, i+1) 
        k = np.random.randint(0, data.shape[0]) 
        plt.title(labels[k]) 
        plt.imshow(data[k]) 
    plt.tight_layout() 
    plt.show()


def grayscale(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img

# load CIFAR-10 data
(x_train_1, y_train_1), (x_test_1, y_test_1) = cifar10.load_data()
x_train_1, x_test_1 = x_train_1/255, x_test_1/255

class_names_cifar10 = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Create a dictionary mapping numerical labels to class names
label_to_class_name = {i: class_names_cifar10[i] for i in range(len(class_names_cifar10))}

# Apply the mapping to the training and testing labels
y_train_1 = [label_to_class_name[label] for label in y_train_1.flatten()]
y_test_1 = [label_to_class_name[label] for label in y_test_1.flatten()]



class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

classes_needed = ["cattle", "fox", "baby", "boy", "girl", "man", "woman", "rabbit", "squirrel", "maple", "oak", "palm", "pine", "willow", "bicycle", "bus", "motorcycle", "pickup truck", "train", "lawn-mower", "tractor"]



(x_train_2, y_train_2), (x_test_2, y_test_2) = cifar100.load_data(label_mode='fine')
x_train_2, x_test_2 = x_train_2/255, x_test_2/255

# match the y_train and y_test to the classes_needed
y_train_2 = [class_names[label] for label in y_train_2.flatten()]
y_test_2 = [class_names[label] for label in y_test_2.flatten()]

# only keep the images that are in the classes_needed
x_train_2 = x_train_2[np.isin(y_train_2, classes_needed)]
y_train_2 = np.array(y_train_2)[np.isin(y_train_2, classes_needed)]
x_test_2 = x_test_2[np.isin(y_test_2, classes_needed)]
y_test_2 = np.array(y_test_2)[np.isin(y_test_2, classes_needed)] 

# show_samples(x_train_1, y_train_1)  
# show_samples(x_train_2, y_train_2)

print("Training data shape:", x_train_2.shape)
print("Testing data shape:", x_test_2.shape)
print("Training labels shape:", y_train_2.shape)
print("Testing labels shape:", y_test_2.shape)

# join the 2 x_train_1 and x_train_2 and y_train_1 and y_train_2, and do the same for the test data
x_train = np.concatenate([x_train_1, x_train_2])
y_train = np.concatenate([y_train_1, y_train_2])
x_test = np.concatenate([x_test_1, x_test_2])
y_test = np.concatenate([y_test_1, y_test_2])

print("data combined")
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)
print("Training labels shape:", y_train.shape)
print("Testing labels shape:", y_test.shape)

# show_samples(x_train, y_train)

num_of_samples = []
cols = 5
unique_classes = np.unique(y_train)

fig, axs = plt.subplots(nrows=len(unique_classes), ncols=cols, figsize=(5, 50))
fig.tight_layout()

for i in range(cols):
    for j in unique_classes:
        x_selected = x_train[y_train == j]
        axs[np.where(unique_classes == j)[0][0]][i].imshow(
            x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray")
        )
        axs[np.where(unique_classes == j)[0][0]][i].axis("off")
        if i == 2:
            num_of_samples.append(len(x_selected))
            axs[np.where(unique_classes == j)[0][0]][i].set_title(str(j))
plt.show()


print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(unique_classes, num_of_samples)
plt.title("Distribution of the training set")
plt.ylabel("Number of images")
plt.show()

# Display the 1000th image
plt.imshow(x_train[1000])  
plt.title(f"Class: {y_train[1000]}")
plt.axis("off")
print(x_train[1000].shape)
print(y_train[1000])
plt.show()

# grey scale the 1000th image
img = grayscale(x_train[1000])
plt.imshow(img)
plt.axis("off")
print(img.shape)
















