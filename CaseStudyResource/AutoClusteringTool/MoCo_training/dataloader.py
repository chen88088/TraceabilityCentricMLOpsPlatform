import os
import random
import numpy as np
import tensorflow as tf

from augment import Augment


AUTO = tf.data.experimental.AUTOTUNE


def set_dataset(task, data_path='./data'):
    # Initialize empty lists to hold the file paths and labels
    file_paths = []
    labels = []

    # Iterate through each subfolder in the data directory
    for subdir, _, files in os.walk(data_path):
        npy_files = sorted([f for f in files if f.endswith('.npy')])

        # Append the file paths and generate random labels
        for npy_file in npy_files:
            file_paths.append(os.path.join(subdir, npy_file))
            labels.append(random.randint(0, 1))  # Random label

    # Define a function to load and preprocess the images
    def load_and_preprocess_image(path, label):
        # Load .npy file and ensure it has the shape (320, 320, 4)
        x = np.load(path)
        if x.shape[-1] != 4:
            raise ValueError(f"Expected 4 channels in image, but got {x.shape[-1]}.")
        x = x[..., :3]  # Select only the first 3 channels
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return x, label

    # Convert lists to TensorFlow Dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    dataset = dataset.map(lambda path, label: tf.numpy_function(load_and_preprocess_image, [path, label], (tf.float32, tf.int32)))

    # Print the dataset length for verification
    print(len(list(dataset)))

    if task == 'lincls':
        # For linear classification, return the dataset split into training and validation
        trainset = dataset.take(int(0.8 * len(file_paths)))  # 80% for training
        valset = dataset.skip(int(0.8 * len(file_paths)))  # 20% for validation
        return trainset, valset

    return dataset


class DataLoader:
    def __init__(self, args, mode, datalist, batch_size, num_workers=1, shuffle=True):
        self.args = args
        self.mode = mode
        self.datalist = datalist
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.dataloader = self._dataloader()

    def __len__(self):
        return len(self.datalist)


    def augmentation(self, img, shape):
        augset = Augment(self.args, self.mode)
        if self.args.task in ['v1', 'v2']:
            img_list = []
            for _ in range(2): # query, key
                aug_img = tf.identity(img)
                if self.args.task == 'v1':
                    aug_img = augset._augmentv1(aug_img, shape) # moco v1
                else:
                    radius = np.random.choice([3, 5])
                    aug_img = augset._augmentv2(aug_img, shape, (radius, radius)) # moco v2
                img_list.append(aug_img)
            return img_list
        else:
            return augset._augment_lincls(img, shape)

    def dataset_parser(self, img, label=None):
        shape = tf.shape(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        if self.args.task in ['v1', 'v2']:
            # MoCo (v1 or v2)
            query, key = self.augmentation(img, shape)
            inputs = {'query': query, 'key': key}
            labels = tf.zeros([])  # Labels are not needed for MoCo pretraining
        elif self.args.task == 'lincls':
            # Linear classification (lincls)
            inputs = self.augmentation(img, shape)
            labels = tf.one_hot(label, self.args.classes)  # One-hot encode labels for classification
        else:
            raise ValueError(f"Unknown task: {self.args.task}")

        return (inputs, labels)

    def shuffle_BN(self, value, labels):
        if self.num_workers > 1:
            pre_shuffle = [(i, value['key'][i]) for i in range(self.batch_size)]
            random.shuffle(pre_shuffle)
            shuffle_idx = []
            value_temp = []
            for vv in pre_shuffle:
                shuffle_idx.append(vv[0])
                value_temp.append(tf.expand_dims(vv[1], axis=0))
            value['key'] = tf.concat(value_temp, axis=0)
            unshuffle_idx = np.array(shuffle_idx).argsort().tolist()
            value.update({'unshuffle': unshuffle_idx})
        return (value, labels)
        
    def _dataloader(self):
        dataset = self.datalist

        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(1024)  # Shuffle buffer size of 1024

        dataset = dataset.map(self.dataset_parser, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)
        if self.args.shuffle_bn and self.args.task in ['v1', 'v2']:
            # only moco
            dataset = dataset.map(self.shuffle_BN, num_parallel_calls=AUTO)
        return dataset