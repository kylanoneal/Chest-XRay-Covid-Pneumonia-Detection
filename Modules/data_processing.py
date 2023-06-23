import os
import time
import torch
import numpy as np
from numpy import random
import cv2

def load_dataset(img_paths, label_dict, num_images, img_size, shuffle=True, is_color=False, zero_centered=True,
                 batch_size=None, horizontal_flip=False):
    start_time = time.time()
    data = []
    labels = []

    channels = 3 if is_color else 1

    # Shuffle data
    if shuffle:
        random.shuffle(img_paths)

    print("Paths shuffled. Time: ", time.time() - start_time)
    images = []
    # Read images and corresponding labels
    for i in range(num_images):

        filename = img_paths[i]
        labels.append(label_dict[filename])
        if is_color:
            img = cv2.imread(filename)
        else:
            img = cv2.imread(filename, 0)

        # Resize
        img = cv2.resize(img, img_size, cv2.INTER_LINEAR)

        # Normalize
        img = img.astype(float) / 255 * 2 - 1

        if horizontal_flip:
            flipped_img = cv2.flip(img, 1)
            flipped_img = np.reshape(flipped_img, (channels, img_size[0], img_size[1]))
            labels.append(label_dict[filename])
            data.append(flipped_img)

        img = np.reshape(img, (channels, img_size[0], img_size[1]))
        data.append(img)

    print("Images read, resized, normalized, and reshaped. Time:", time.time() - start_time)

    # Perform zero-centered regularization
    if zero_centered:
        for i in range(len(data)):
            data[i] -= np.mean(data[i])

        print("Images zero centered. Time: ", time.time() - start_time)

    if horizontal_flip:
        bundle = list(zip(data, labels))
        random.shuffle(bundle)
        data, labels = zip(*bundle)

    # Convert to torch tensors
    data = torch.FloatTensor(data)
    labels = torch.LongTensor(labels)

    print("Converted to torch tensors. Time: ", time.time() - start_time)

    # Create minibatches
    if batch_size is not None:
        data_batches = []
        label_batches = []

        total_batches = int(len(data) / batch_size)

        data = [data[i * batch_size: (i + 1) * batch_size] for i in range(total_batches)]
        labels = [labels[i * batch_size: (i + 1) * batch_size] for i in range(total_batches)]

    print("Minibatches created. Time: ", time.time() - start_time)
    return list(zip(data, labels))


# Create label dictionary: Key = filename, value = label
# Function is necessary because labels are stored in .txt file

def get_label_dict(label_path, images_path):
    label_file = open(label_path, 'r')
    lines = label_file.readlines()
    label_dict = {}

    print("Number of labels: ", len(lines))
    for line in lines:
        split = line.split(' ')

        if split[2] == 'normal':
            lbl = 0
        elif split[2] == 'pneumonia':
            lbl = 1
        else:
            lbl = 2

        label_dict[images_path + split[1]] = lbl

    return label_dict

def remove_non_labeled(label_dict, img_paths):

    return [img_path for img_path in img_paths if img_path in label_dict]

def get_dataloader(images_path, label_path, img_size, batch_size):
    label_dict = get_label_dict(label_path, images_path)

    img_paths = [images_path + file for file in os.listdir(images_path)]

    # Remove non-labeled images (there are a few hundred without labels)
    img_paths = [img_path for img_path in img_paths if img_path in label_dict]

    print("Total number of labeled images:", len(img_paths))

    return load_dataset(img_paths, label_dict, num_images=len(img_paths), img_size=img_size,
                                batch_size=batch_size)

