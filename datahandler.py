"""Training data management and operations."""

import os
import time
import random
import numpy as np
from sklearn.model_selection import train_test_split

from utilityfun import import_image

class TrainingData():
    """Reads and holds all of the training images with their labels.
    """
    def __init__(self, training_data_path):
        self.img_extensions = ['png', 'jpg', 'jpeg']
        self.training_set = self.import_training_data(training_data_path)
        self.training_set_features = None

    def import_training_data(self, data_path):
        """Import the training data from a given input path. Assuming that
        the input path only contains folders where the folder name designates
        the image label. All sub-directories are searched but their names are
        ignored.
        """
        folders = os.listdir(data_path)
        training_set = {}

        for folder in folders:
            folder_path = os.path.join(data_path, folder)
            if os.path.isdir(folder_path):
                new_images = self.search_folder_for_images(folder_path)
                folder_name = folder.split('/')[-1]
                training_set[folder_name] = new_images

        print('Training data set imported.')
        return training_set

    def search_folder_for_images(self, folder):
        """Recursively search through and import all images that are found
        in a folder.
        """
        files = os.listdir(folder)

        images = []
        for file in files:
            file_path = os.path.join(folder, file)
            if self.get_extension(file) in self.img_extensions:
                images.append(import_image(file_path))
            elif os.path.isdir(file_path):
                images.extend(self.search_folder_for_images(file_path))

        return images

    def get_random_training_images(self, label=None, number=1):
        """Return a random sample of the training images currently loaded."""
        label = list(self.training_set.keys())[0] if label is None else label
        training_images = self.training_set[label]
        return [random.choice(training_images) for i in range(0, number)]

    def extract_features(self, image_analyser):
        """Extracts image features from all elements in the data set."""
        self.training_set_features = {}
        for label, images in self.training_set.items():
            start = time.time()
            self.training_set_features[label] = []
            for image in images:
                image_analyser(image)
                features = image_analyser.get_image_features()
                self.training_set_features[label].append(features)
            end = time.time()
            print('Extracted feature vectors from images labelled as ' \
                + label + ' in', end - start, 'sec.')

    def get_data(self, test_fraction=0.25):
        """Returns the shuffled training data as lists.

        Parameters
        ----------
        test_fraction: float, The fraction of the data set that should be
        allocated to the test set. It must be a value between 0.0 and 1.0.

        Returns
        ----------
        split_set: list, length=4, The training features, training labels,
        test features, test labels.
        """
        features = []
        labels = []

        for label, feature_vectors in self.training_set_features.items():
            features.extend(feature_vectors)
            labels.extend([label for feature_vector in feature_vectors])

        rand_state = np.random.randint(0, 100)
        test_train_set = train_test_split(features, labels,
                                          test_size=test_fraction,
                                          random_state=rand_state)
        return (test_train_set[0], test_train_set[1],
                test_train_set[2], test_train_set[3])

    @staticmethod
    def get_extension(filename):
        """Isolates the filetype from file name."""
        return filename.split('.')[-1]
