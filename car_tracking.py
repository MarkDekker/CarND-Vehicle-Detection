"""Car tracker for input video feed.
"""

import os
import random
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

class ImageFrame():
    """Holds all of the information related to the current image frame under
    consideration and the respective methods to analyse and modify the image.
    """
    def __init__(self, area_of_interest, hog_parameters):
        self.colorspace = 'HLS'
        self.image = None
        self.hog_params = hog_parameters
        self.hog_features = None
        self.hog_image = None
        self.search_area = area_of_interest
        self.color_bins = None
        self.color_bin_image = None

    def __call__(self, image):
        # Clear all variables dependent on the image
        converter = getattr(cv2, "COLOR_RGB2" + self.colorspace)
        self.image = cv2.cvtColor(image, converter)

        cell_size = self.hog_params['pix_per_cell']
        bin_size = int(cell_size/2)
        self.color_bins, self.color_bin_image = \
                                        self.extract_colour_bins(bin_size)

        try:
            self.hog_features, self.hog_image = \
                                                self.extract_hog_features()
        except:
            try:
                self.hog_features = self.extract_hog_features()
            except ValueError:
                print('HOG feature extraction returned too many values.')

    def display_frame(self, title=''):
        """Plot the currently loaded image frame in the car tracker."""
        if self.image is not None:
            plot_image(self.image, title=title)

    def save_frame(self, output_folder='./output_images/',
                   frame_name='Current_Frame'):
        """Save the currently loaded image frame to a file."""
        name = frame_name
        save_path = os.path.join(output_folder, name) + '.jpg'
        if self.image is not None:
            converter = getattr(cv2, "COLOR_" + self.colorspace + "2RGB")
            cv2.imwrite(save_path, cv2.cvtColor(self.image, converter))
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
        visualise = self.hog_params['visualise']

        return hog(img[:, :, 1],
                   orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   visualise=visualise, block_norm='L2-Hys')

    def extract_colour_bins(self, bin_size):
        """Extract the colour bins from the image based on the cell size."""
        img = self.image
        new_size = (int(img.shape[0]/bin_size), int(img.shape[1]/bin_size))
        result = cv2.resize(self.image, new_size)
        return result.ravel(), result

    def get_hog_features(self):
        """Returns the Histogram of Oriented Gradients """
        if self.hog_features is None:
            raise ValueError('There are no HOG features available.')
        else:
            return self.hog_features
    
    def get_color_bin_features(self):
        """Returns the Histogram of Oriented Gradients """
        if self.color_bins is None:
            raise ValueError('There are no HOG features available.')
        else:
            return self.color_bins

    def get_hog_visualisation(self):
        """Returns the Histogram of Oriented Gradients """
        if self.hog_image is None:
            raise ValueError('There is no HOG visualisation available.')
        else:
            return self.hog_image

    def get_colour_bin_visualisation(self):
        """Extract the colour bins from the image based on the cell size."""
        if self.color_bins is None:
            raise ValueError('There are no colour bin features available.')
        else:
            converter = getattr(cv2, "COLOR_" + self.colorspace + "2RGB")
            return cv2.cvtColor(self.color_bin_image, converter)

    def get_area_of_interest(self):
        """Reduces the image to the area of interest."""
        img_h = self.image.shape[0]
        img_w = self.image.shape[1]
        top = int(self.search_area[0][1] * img_h)
        bottom = int(self.search_area[1][1] * img_h)
        left = int(self.search_area[0][0] * img_w)
        right = int(self.search_area[1][0] * img_w)

        return self.image[top:bottom, left:right, :]


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

    def extract_features(self, image_frame):
        """Extracts image features from all elements in the data set."""
        self.training_set_features = {}
        for label, images in self.training_set.items():
            start = time.time()
            self.training_set_features[label] = []
            for image in images:
                image_frame(image)
                hog_features = image_frame.get_hog_features()
                color_features = image_frame.get_color_bin_features()
                self.training_set_features[label].append(
                                np.concatenate((hog_features, color_features)))
            end = time.time()
            print('Extracted feature vectors from images labelled as ' \
                + label + ' in', end - start, 'sec.')

    @staticmethod
    def get_extension(filename):
        """Isolates the filetype from file name."""
        return filename.split('.')[-1]

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

        for label, images in self.training_set.items():
            features.extend(images)
            labels.extend([label for image in images])

        labels, features = shuffle((labels, features))
        return train_test_split(features, labels, test_size=test_fraction)

class GridSearch():
    def __init__(self, search_windows, search_areas, block_size=10):
        self.search_windows = search_windows
        self.search_areas = search_areas
        self.block_size = block_size

    def crawl_img(self, img, window_size, step_x, step_y, callback_fun):
        """Calls a function for each step while crawling over the input image.

        The input image is sampled based on the supplied search window
        parameters and the supplied callback function is applied for each step.

        Returns
        --------
        dict: Containing two entries - 'position' with the search window
        position, 'output' with the result from what the callback function
        returns for each step.
        """
        print("Searching")


class Classifier():
    """Base class for the ML models to identify cars."""
    def __init__(self, classifier):
        self.clf = classifier
        self.feature_scaler = None

    def __call__(self):
        raise NotImplementedError

    def train(self, features, labels):
        """Trains the classifier with the supplied training data."""
        features = self.normalise(features)
        self.clf.fit(features, labels)

    def predict(self, features):
        """Predicts the labels for a given list of input features."""
        features = self.normalise(features)
        self.clf.predict(features)

    def normalise(self, features):
        """Normalises the input feature set. """
        features = np.vstack(features).astype('float64')

        if self.feature_scaler is None:
            raise ValueError('The feature scaler has not yet been fit!')
        else:
            return self.feature_scaler.transform(features)

    def fit_feature_scaler(self, training_features):
        """Fit a feature scaler to the training data to easily normalise inputs
        later.
        """
        training_features = np.vstack(training_features).astype('float64')
        self.feature_scaler = StandardScaler().fit(training_features)

    @classmethod
    def svm(cls, params=None):
        """Set up a support vector machine classifier."""
        if params is None:
            params = {'C': 1.0,
                      'kernel': 'rbf',
                      'max_iter': -1}
        return cls(SVC(**params))



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


def compare_images(img_org, img_undist, titles=None):
    """Display an image comparison in a subplot."""
    if titles is None:
        titles = ('Image Before', 'Image After')
    
    plt.subplots(1, 2, figsize=(10, 5), dpi=80)
    plt.subplot(1, 2, 1)
    plot_image(img_org, titles[0])
    plt.subplot(1, 2, 2)
    plot_image(img_undist, titles[1])


def overlay_image(img, overlay_img, opacity=1.0):
    """Reliably combine two images based on the opacity. Black pixels are
    treated as transparent. The method also takes in a grayscale base image.
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

def quick_rectangle(img, corners, color='green', opacity=0.9, 
                    thickness=4, filled=False):
    """Draws a rectangle on the input image."""
    colors = {'green': (30, 255, 120),
              'blue': (20, 104, 229),
              'red': (224, 52, 0),
              'orange': (252, 163, 9),
              'yellow': (252, 228, 10)}

    if color.lower() not in colors:
        color = 'green'
        print('Warning unknown color, using green instead.  Please choose from:\
              green, blue, red, orange and yellow.')
    else:
        color = color.lower()
    
    thickness = int(thickness)

    outline = np.zeros(img.shape)

    width = img.shape[1]
    height = img.shape[0]
    corners = (corners * np.array([width, height])).astype(int)
    corners = tuple(map(tuple, corners))

    outline = cv2.rectangle(outline, corners[0], corners[1], colors[color],
                            thickness=thickness)

    if filled:
        fill = np.zeros(img.shape)
        fill = cv2.rectangle(fill, corners[0], corners[1], colors[color],
                             thickness=-1)
        img = overlay_image(img, fill, opacity=opacity*0.3)

    return overlay_image(img, outline, opacity=opacity)
