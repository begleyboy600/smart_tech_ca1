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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

def show_samples(data, labels): 
    plt.subplots(figsize=(10, 10)) 
    for i in range(12): 
        plt.subplot(3, 4, i+1) 
        k = np.random.randint(0, data.shape[0]) 
        plt.title(labels[k]) 
        plt.imshow(data[k]) 
    plt.tight_layout() 
    plt.show()
    
def load_data():
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

    classes_needed = ['cattle', 'fox', 'baby', 'boy', 'girl', 'man', 'woman', 'rabbit', 'squirrel', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree', 'bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'tractor']


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

    return (x_train_1, y_train_1), (x_test_1, y_test_1), (x_train_2, y_train_2), (x_test_2, y_test_2)

    # show_samples(x_train_1, y_train_1)  
    # show_samples(x_train_2, y_train_2)

(x_train_1, y_train_1), (x_test_1, y_test_1), (x_train_2, y_train_2), (x_test_2, y_test_2) = load_data()


def concatenate_and_get_validation_data(x_train_1, y_train_1, x_test_1, y_test_1, x_train_2, y_train_2, x_test_2, y_test_2):
    x_train = np.concatenate([x_train_1, x_train_2])
    y_train = np.concatenate([y_train_1, y_train_2])
    x_test = np.concatenate([x_test_1, x_test_2])
    y_test = np.concatenate([y_test_1, y_test_2])

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    return x_train, x_val, y_train, y_val, x_test, y_test

x_train, x_val, y_train, y_val, x_test, y_test = concatenate_and_get_validation_data(x_train_1, y_train_1, x_test_1, y_test_1, x_train_2, y_train_2, x_test_2, y_test_2)

def get_train_val_test_data_shape(x_train, x_val, y_train, y_val, x_test, y_test):
    print("Training data shape:", x_train.shape)
    print("Testing data shape:", x_test.shape)
    print("validation data shape:", x_val.shape)
    print("validation labels shape:", y_val.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing labels shape:", y_test.shape)


get_train_val_test_data_shape(x_train, x_val, y_train, y_val, x_test, y_test)

# show_samples(x_train, y_train)

def get_images_from_each_class(x_train, y_train):
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
    return num_of_samples, unique_classes, x_selected

num_of_samples, unique_classes, x_selected = get_images_from_each_class(x_train, y_train)

def get_distribution_of_classes(x_train, y_train, unique_classes, x_selected):
    num_of_samples = []
    for i in unique_classes:
        x_selected = x_train[y_train == i]
        num_of_samples.append(len(x_selected))
    print(num_of_samples)
    plt.figure(figsize=(12, 4))
    plt.bar(unique_classes, num_of_samples)
    plt.title("Distribution of the training set")
    plt.ylabel("Number of images")
    plt.show()

def grayscale(img):
    if img.dtype == np.uint8:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.dtype == np.float64:
        img_gray = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(f"Unsupported image type: {img.dtype}")

    return img_gray

def show_grayscale(img):
    print("grayscale image")
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.show()


def equalize_img(img):
    img_eq = cv2.equalizeHist(img)
    return img_eq

def show_equalized_image(img):
    print("equalized image")
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.show()


def preprocessing(img):
  img = grayscale(img)
  img = equalize_img(img)
  return img

def get_class_and_count(y_train):
    class_counts = {label: np.sum(y_train == label) for label in np.unique(y_train)}

# Display the 1000th image
plt.imshow(x_train[1000])  
plt.title(f"Class: {y_train[1000]}")
plt.axis("off")
print(x_train[1000].shape)
print(y_train[1000])
plt.show()

x_train = np.array(list(map(preprocessing, x_train)))
x_val = np.array(list(map(preprocessing, x_val)))
x_test = np.array(list(map(preprocessing, x_test)))


def reshape_data(x_train, x_val, x_test):
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 1)
    x_val = x_val.reshape(x_val.shape[0], 32, 32, 1)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 1)
    return x_train, x_val, x_test

x_train, x_val, x_test = reshape_data(x_train, x_val, x_test)

def get_xtrain_xval_xtest_data_shape(x_train, x_val, y_train, y_val, x_test, y_test):
    print("x train shape:", x_train.shape)
    print("x val shape:", x_val.shape)
    print("x test shape:", x_test.shape)

get_xtrain_xval_xtest_data_shape(x_train, x_val, y_train, y_val, x_test, y_test)

img = x_train[1000]
# display the image
plt.imshow(img.squeeze(), cmap='gray')
plt.axis("off")
plt.show()

def data_augmentation(x_train, y_train):
    classes_for_data_augmentation = [
        'baby', 'bicycle', 'boy', 'bus', 'cattle', 'fox', 'girl', 'lawn_mower',
        'man', 'maple_tree', 'motorcycle', 'oak_tree', 'palm_tree', 'pickup_truck', 'pine_tree',
        'rabbit', 'squirrel', 'tractor', 'train', 'willow_tree', 'woman'
    ]

    # Filter x_train and y_train for the desired classes
    filtered_indices = np.isin(y_train, classes_for_data_augmentation)
    x_train_desired_classes = x_train[filtered_indices]
    y_train_desired_classes = y_train[filtered_indices]

    # Shuffle the filtered data to ensure randomness
    x_train_desired_classes, y_train_desired_classes = shuffle(x_train_desired_classes, y_train_desired_classes, random_state=42)

    # Data augmentation for the desired classes
    datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10)
    datagen.fit(x_train_desired_classes)
    batches = datagen.flow(x_train_desired_classes, y_train_desired_classes, batch_size=5)
    x_batch, y_batch = next(batches)
    return x_batch, y_batch

x_batch, y_batch = data_augmentation(x_train, y_train)
def concatenate_augmented_data(x_train, y_train, x_batch, y_batch):
    x_train = np.concatenate([x_train, x_batch])
    y_train = np.concatenate([y_train, y_batch])
    return x_train, y_train

x_train, y_train = concatenate_augmented_data(x_train, y_train, x_batch, y_batch)

class_counts = {label: np.sum(y_train == label) for label in np.unique(y_train)}


# Plot the distribution
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values())
plt.title("Distribution of Classes in Updated Training Data")
plt.xlabel("Class Label")
plt.ylabel("Number of Samples")
plt.show()

print("=========================================")
for label, count in class_counts.items():
    print(f"Class {label}: {count} samples")
print("=========================================")


def label_encoding(y_train, y_val, y_test):
    label_encoder = LabelEncoder()
    y_train_numeric = label_encoder.fit_transform(y_train)
    y_val_numeric = label_encoder.transform(y_val)
    y_test_numeric = label_encoder.transform(y_test)

    nunique_classes = label_encoder.classes_
    num_classes = len(unique_classes)
    return y_train_numeric, y_val_numeric, y_test_numeric, num_classes

y_train_numeric, y_val_numeric, y_test_numeric, num_classes = label_encoding(y_train, y_val, y_test)

def one_hot_encoding(y_train_numeric, y_val_numeric, y_test_numeric, num_classes):
    y_train = to_categorical(y_train_numeric, num_classes)
    y_val = to_categorical(y_val_numeric, num_classes)
    y_test = to_categorical(y_test_numeric, num_classes)

    return y_train, y_val, y_test

y_train, y_val, y_test = one_hot_encoding(y_train_numeric, y_val_numeric, y_test_numeric, num_classes)









