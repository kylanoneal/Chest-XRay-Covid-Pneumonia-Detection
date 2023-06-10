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
        img = img.astype(np.float) / 255 * 2 - 1

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
    return zip(data, labels)


# Create label dictionary: Key = filename, value = label
# Function is necessary because labels are stored in .txt file

def get_label_dict(path, img_path):
    label_file = open(path, 'r')
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

        label_dict[img_path + split[1]] = lbl

    return label_dict


# Some images do not have a label
def remove_non_labeled(paths, label_dict):
    labeled_img_paths = []
    for path in paths:
        if path in label_dict.keys():
            labeled_img_paths.append(path)

    return labeled_img_paths