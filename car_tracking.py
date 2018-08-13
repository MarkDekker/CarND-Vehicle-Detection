"""Car tracker for input video feed.
"""

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC



class ImageFrame():
    """Holds all of the information related to the current image frame under
    consideration and the respective methods to analyse and modify the image.
    """
    def __init__(self, image, frame_name='Current Frame'):
        self.image = image
        self.frame_name = frame_name
        self.hog_params = {'orientations': 9,
                           'pix_per_cell': 8,
                           'cell_per_block': 2}
        self.hog_features = None
        self.search_area = ((0, 0), (1, 1))

    def __call__(self, image, frame_name='Current Frame'):
        # Clear all variables dependent on the image
        self.image = image
        self.frame_name = frame_name
        self.hog_features = self.extract_hog_features()

    def display_frame(self, title=''):
        """Plot the currently loaded image frame in the car tracker."""
        if self.image is not None:
            plot_image(self.image, title=title)

    def save_frame(self, output_folder='./output_images/'):
        """Save the currently loaded image frame to a file."""
        if self.frame_name is None:
            name = 'No_frame_name__'
        else:
            name = self.frame_name

        save_path = os.path.join(output_folder, name) + '.jpg'
        if self.image is not None:
            cv2.imwrite(save_path, cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        else:
            print('No frame available for export!')

    def extract_hog_features(self):
        """Extract the "Histogram of Oriented Gradients" for the region of
        interest of the current image frame.
        """
        img = self.get_area_of_interest()

        orient = self.hog_params['orientations']
        pix_per_cell = self.hog_params['pix_per_cell']
        cell_per_block = self.hog_params['cell_per_block']
        return hog(img,
                   orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   visualise=False, feature_vector=False)

    def get_hog_features(self):
        """Returns the Histogram of Oriented Gradients """
        if self.hog_features is None:
            raise ValueError('There are no HOG features available.')
        else:
            return self.hog_features

    def get_area_of_interest(self):
        """Reduces the image to the area of interest."""
        img_h = self.image.shape[0]
        img_w = self.image.shape[1]
        top = int(self.search_area[0][0] * img_h)
        bottom = int(self.search_area[0][0] * img_h)
        left = int(self.search_area[0][0] * img_w)
        right = int(self.search_area[0][0] * img_w)

        return self.image[left:right, top:bottom, :]


class TrainingData():
    """Reads and holds all of the training images with their labels.
    """
    def __init__(self, training_data_path):
        self.img_extensions = ['png', 'jpg', 'jpeg']
        self.training_set = self.import_training_data(training_data_path)

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
        label = self.training_set.keys()[0] if label is None else label
        training_images = self.training_set[label]
        return [random.choice(training_images) for i in range(0, number)]

    @staticmethod
    def get_extension(filename):
        """Isolates the filetype from file name."""
        return filename.split('.')[-1]

    def get_data(self, test_fraction=0.25):
        """Returns the shuffled training data as lists.

        Parameters
        ----------
        test_fraction: float, The fraction of the data set that should be allocated
        to the test set. It must be a value between 0.0 and 1.0.

        Returns
        ----------
        split_set: list, length=4, The training features, training labels,
        test features, test labels.
        """
        features = []
        labels = []

        for label, images in self.training_set.items():
            features.extend(images)
            labels.extend([label for image in images])

        labels, features = shuffle((labels, features))

        return train_test_split(features, labels, test_size=test_fraction)

class GridSearch():
    def __init__(self, search_windows, search_areas):
        self.search_windows = search_windows
        self.search_areas = search_areas

    def crawl_img(self, img, window_size, step_x, step_y, callback_fun):
        """Calls a function for each step while crawling over the input image.

        The input image is sampled based on the supplied search window parameters
        and the supplied callback function is applied for each step.

        Returns
        --------
        dict: Containing two entries - 'position' with the search window
        position, 'output' with the result from what the callback function returns
        for each step.
        """
        print("Searching")


class Classifier():
    """Base class for the ML models to identify cars."""
    def __call__(self):
        self.clf = None
        raise NotImplementedError

    #@Classmethod
    # Train
    def train(self, features, labels):
        """Trains the classifier with the supplied training data."""
        self.clf.fit(features, labels)

    # Predict
    def predict(self, features):
        """Predicts the labels for a given list of input features."""
        features = normalise(features)
        self.clf.predict(features)

    # Normalise
    def normalise(self, features):
        """Normalises the input feature set. """
        X = np.vstack(feature_list).astype(np.float64)
        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)



class ClassifierSVM(Classifier):
    """A support Vector Machine (SVM) based car classifier."""
    def __init__(self, params=None):
        if params is None:
            params = {'C': 1.0,
                      'kernel': 'rbf',
                      'max_iter': -1}
        self.clf = SVC(**params)

        



#-----------------------------------------------------------------------------#
#                           Utility functions
#-----------------------------------------------------------------------------#



def import_image(image_path):
    """Import an image from the supplied path.
    Returns the image name and the image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_name = image_path.split('/')[-1].split('.')[0]
    return img

def plot_image(img, title=''):
    """Plot the supplied image and the corresponding title."""
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.axis('off')
    plt.title(title, fontsize=20)


def compare_images(img_org, img_undist):
    """Display an image comparison in a subplot."""
    plt.subplots(1, 2, figsize=(20, 10), dpi=150)
    plt.subplot(1, 2, 1)
    plot_image(img_org, 'Image Before')
    plt.subplot(1, 2, 2)
    plot_image(img_undist, 'Image After')


def overlay_image(img, overlay_img, opacity=1.0):
    """Reliably combine two images based on the opacity. Black pixels are treated as transparent.
    The method also takes in a grayscale base image.
    """
    if len(img.shape) < 3:
        img = np.stack((img, img, img), axis=-1)

    img_out = np.zeros(img.shape)
    overlay_img = (overlay_img * opacity).astype(int)
    faded_img = (img * (1 - opacity)).astype(int)

    for i in range(0, img.shape[2]):
        channel_out = np.where(overlay_img[:, :, i] == 0,
                               img[:, :, i],
                               faded_img[:, :, i] + overlay_img[:, :, i])

        img_out[:, :, i] = np.where(channel_out > 255, 255, channel_out)

    return img_out.astype('uint8')
