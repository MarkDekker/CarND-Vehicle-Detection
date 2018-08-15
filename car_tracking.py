"""Car tracker for input video feed.
"""

import os
import random
import time
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

class ImageFrame():
    """Holds all of the information related to the current image frame under
    consideration and the respective methods to analyse and modify the image.
    """
    def __init__(self, hog_parameters):
        self.colorspace = 'HLS'
        self.image = None
        self.hog_params = hog_parameters
        self.hog_features = None
        self.hog_image = None
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
        img = self.image
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

    def get_image(self):
        """Returns the Histogram of Oriented Gradients """
        if self.image is None:
            raise ValueError('There is no image available.')
        else:
            return self.image

    def get_colour_bin_visualisation(self):
        """Extract the colour bins from the image based on the cell size."""
        if self.color_bins is None:
            raise ValueError('There are no colour bin features available.')
        else:
            converter = getattr(cv2, "COLOR_" + self.colorspace + "2RGB")
            return cv2.cvtColor(self.color_bin_image, converter)


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

        for label, feature_vectors in self.training_set_features.items():
            features.extend(feature_vectors)
            labels.extend([label for feature_vector in feature_vectors])

        labels, features = shuffle((labels, features))
        return train_test_split(features, labels, test_size=test_fraction)


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
        return self.clf.predict(features)

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
    
    @classmethod
    def svm_linear(cls, params=None):
        """Set up a linear support vector machine classifier."""
        if params is None:
            params = {'C': 1.0,
                      'dual': 'false',
                      'max_iter': -1}
        return cls(LinearSVC(**params))


class GridSearch():
    """Tool to search and image with a sliding window approach."""
    def __init__(self, search_windows, search_areas, hog_params,
                 training_resolution=64):
        self.search_areas = search_areas
        self.search_windows = self.process_windows(search_windows)
        self.frame = ImageFrame(hog_params)
        self.training_res = training_resolution
        self.crawl_result = None

    def process_windows(self, search_windows):
        """Resizes the steps to ensure consistency with the search area."""

        total_steps = 0

        for name, window in search_windows.items():
            area = self.search_areas[window['search_area']]
            area_width = (area[1][0] - area[0][0])
            area_height = (area[1][1] - area[0][1])

            steps_x = int(math.ceil((area_width - window['size'][0])
                          / window['step_x']))
            window['step_x'] = int(math.ceil((area_width - window['size'][0])
                                   / steps_x))
            steps_y = int(math.ceil((area_height - window['size'][1])
                          / window['step_y']))
            window['step_y'] = int(math.ceil((area_height - window['size'][1])
                                   / steps_y))

            search_windows[name] = window
            total_steps += (steps_x) * (steps_y)

        print('With the current search window and area setup, each frame ' + \
              'will be sampled', total_steps, 'times.')
        return search_windows

    def get_window_position(self, step_number, window):
        """Get the absolute position of the window based on the step number as
        a fraction of the image dimensions
        """
        top = self.search_areas[window['search-area']][0][1]
        left = self.search_areas[window['search-area']][0][0]
        area = self.search_areas[window['search_area']]
        area_width = (area[0][0] - area[1][0])

        steps_x = int(area_width / window['step_x'])

        top_offset = int(step_number / steps_x) * window['step_y']
        left_offset = int(step_number % steps_x) * window['step_x']

        return (left + left_offset), (top + top_offset)

    def set_frame_image(self, img):
        """Update the image frames with the supplied image."""
        if self.frame is None:
            raise ValueError('No image frame objects exist.')
        else:
            std_size = (self.training_res, self.training_res)
            img = cv2.resize(img, std_size)
            return self.frame(img)

    def crawl_image(self, img, classifier, highlight=True):
        """Calls a function for each step while crawling over the input image.

        The input image is sampled based on the supplied search window
        parameters and the supplied callback function is applied for each step.

        Returns
        --------
        dict: Containing two entries - 'position' with the search window
        position, 'output' with the result from what the callback function
        returns for each step.
        """
        search_result = []
        
        for name, window in self.search_windows.items():
            lowest = self.search_areas[window['search_area']][1][1]
            rightmost = self.search_areas[window['search_area']][1][0]
            top = self.search_areas[window['search_area']][0][1]
            bottom = top + window['step_y']
            right = 0

            while bottom < lowest - 1:
                left = self.search_areas[window['search_area']][0][0]
                while right < rightmost - 1:
                    left += min(window['step_x'], rightmost - right)
                    right = left + window['size'][0]

                    window_area = ((left, top), (right, bottom))
                    window_img = get_area_of_interest(img, window_area)
                    self.set_frame_image(window_img)
                    frame = self.frame
                    features = [np.concatenate((frame.get_hog_features(),
                                                frame.get_color_bin_features()))]

                    search = {}
                    search['position'] = (left, top)
                    search['window'] = name
                    search['label'] = classifier.predict(features)[0]

                    search_result.append(search)

                right = 0
                top += min(window['step_y'], lowest - bottom)
                bottom = top + window['size'][1]
            print(name, bottom, lowest)


        if highlight:
            highlighted_cars = self.highlight_labels(img, search_result,
                                                     'vehicles')
            return search_result, highlighted_cars
        else:
            return search_result

    def highlight_labels(self, img, search_result, target_label,
                         color='yellow'):
        """Draws a rectangle around windows which detected a given label."""
        annotated_img = np.copy(img)
        for step in search_result:
            if step['label'] == target_label:
                window_area = self.get_window_area(step['position'],
                                                   step['window'])
                annotated_img = quick_rectangle(annotated_img, window_area,
                                                color=color, opacity=0.4,
                                                filled=True, thickness=2)

        return annotated_img

    def get_window_area(self, top_left_crnr, window_name):
        """Returns the  coordinates of the search window in pixels."""
        height = self.search_windows[window_name]['size'][1]
        width = self.search_windows[window_name]['size'][0]
        left = top_left_crnr[0]
        top = top_left_crnr[1]
        right = left + width
        bottom = top + height

        return (top_left_crnr, (right, bottom))


#-----------------------------------------------------------------------------#
#                           Utility functions
#-----------------------------------------------------------------------------#

def get_area_of_interest(image, search_area):
    """Reduces the image to the area of interest."""
    top = search_area[0][1]
    bottom = search_area[1][1]
    left = search_area[0][0]
    right = search_area[1][0]

    return image[top:bottom, left:right, :]

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
    corners = tuple(map(tuple, corners))
    outline = cv2.rectangle(outline, corners[0], corners[1], colors[color],
                            thickness=thickness)

    if filled:
        fill = np.zeros(img.shape)
        fill = cv2.rectangle(fill, corners[0], corners[1], colors[color],
                             thickness=-1)
        img = overlay_image(img, fill, opacity=opacity*0.3)

    return overlay_image(img, outline, opacity=opacity)
