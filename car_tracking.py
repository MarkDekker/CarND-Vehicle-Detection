"""Car tracker for input video feed.
"""

import os
import random
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC

class ImageAnalyser():
    """Holds all of the information related to the current image under
    consideration and the respective methods to analyse and modify the image.
    """
    def __init__(self, hog_parameters, spatial_size=(32, 32),
                 histogram_bins=32, colorspace='RGB'):
        self.colorspace = colorspace
        self.image = None
        self.hog_params = hog_parameters
        self.hog_features = None
        self.hog_image = None
        self.spatial_size = spatial_size
        self.histogram_bins = histogram_bins

    def __call__(self, image):
        self.set_image(image)
        if self.hog_params['visualise']:
            self.hog_features, self.hog_image = \
                                                self.extract_hog_features()
        else:
            self.hog_features = self.extract_hog_features()

    def set_image(self, image):
        """Updates image in object and applies the preset colourspace."""
        if self.colorspace != 'RGB':
            converter = getattr(cv2, "COLOR_RGB2" + self.colorspace)
            image = cv2.cvtColor(image, converter)
        self.image = image

    def change_colorspace(self, new_colorspace):
        """Change the colorspace from RGB."""
        image = self.image

        old_colorspace = self.colorspace
        possible_spaces = ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
        if (new_colorspace in possible_spaces and
                new_colorspace != old_colorspace):
            converter = getattr(cv2, "COLOR_" + old_colorspace + "2"
                                + new_colorspace)
            self.colorspace = new_colorspace
            self.image = cv2.cvtColor(image, converter)

    def extract_hog_features(self):
        """Extract the "Histogram of Oriented Gradients" for the region of
        interest of the current image.
        """
        img = self.image
        if self.colorspace != 'RGB':
            converter = getattr(cv2, "COLOR_" + self.colorspace + "2RGB")
            img = cv2.cvtColor(img, converter)
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        orient = self.hog_params['orientations']
        pix_per_cell = self.hog_params['pix_per_cell']
        cell_per_block = self.hog_params['cell_per_block']
        visualise = self.hog_params['visualise']

        return hog(img,
                   orientations=orient,
                   pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   visualise=visualise, block_norm='L2-Hys',
                   feature_vector=False)

    def get_hog_features(self, window=None):
        """Returns the Histogram of Oriented Gradients """
        if self.hog_features is None:
            raise ValueError('There are no HOG features available.')
        else:
            if window is not None:
                window_size = window[1][0] - window[0][0]
                cells_block = self.hog_params['cell_per_block']
                blocks_per_window = window_size - cells_block + 1
                hog_sample = self.hog_features[window[0][1]: window[0][1]
                                               + blocks_per_window,
                                               window[1][1]: window[1][1]
                                               + blocks_per_window]
            else:
                hog_sample = self.hog_features
            
            return hog_sample
    def get_spatial_features(self, window=None):
        """Returns spatially sorted bins of colour values. """
        image_window = self.get_image_window(window)
        return cv2.resize(image_window, self.spatial_size)

    def get_histogram_features(self, window=None):
        """Returns the colour value histogram for a window in an image."""
        image_window = self.get_image_window(window)
        bin_ranges = {'others': ((0, 256), (0, 255), (0, 255)),
                      'HLS': ((0, 181), (0, 255), (0, 255)),
                      'HSV': ((0, 181), (0, 255), (0, 255))}
        bin_range = (bin_ranges[self.colorspace]
                     if self.colorspace in bin_ranges
                     else bin_ranges['others'])
        histogram = []

        for channel in range(image_window.shape[2]):
            channel_hist = np.histogram(image_window[:, :, channel],
                                        bins=self.histogram_bins,
                                        range=bin_range[channel])[0].tolist()
            histogram.append(channel_hist)

        return np.array(histogram)

    def get_image_window(self, window):
        """Get the portion of an image defined by the window."""
        if window is not None:
            pix_per_cell = self.hog_params['pix_per_cell']
            top = window[0][1] * pix_per_cell
            #print(window)
            bottom = window[1][1] * pix_per_cell
            left = window[0][0] * pix_per_cell
            right = window[1][0] * pix_per_cell
            return self.image[top:bottom, left:right, :]
        else:
            return self.image

    def get_image_features(self, hog_features=True, spatial=True,
                           histograms=True, window=None):
        """Combines image features and returns these as a feature vector."""
        features = np.array([])
        if hog_features:
            hog_features = self.get_hog_features(window).ravel()
            #print(hog_features.shape)
            features = np.hstack((features, hog_features))
        if spatial:
            spatial_features = self.get_spatial_features(window).ravel()
            features = np.hstack((features, spatial_features))
        if histograms:
            histogram_features = self.get_histogram_features(window).ravel()
            features = np.hstack((features, histogram_features))

        return features

    def get_image(self):
        """Returns the Histogram of Oriented Gradients """
        if self.image is None:
            raise ValueError('There is no image available.')
        else:
            if self.colorspace != 'RGB':
                converter = getattr(cv2, "COLOR_" + self.colorspace + "2RGB")
                image = cv2.cvtColor(self.image, converter)
            else:
                image = self.image
            return image

    def get_spatial_visualisation(self, window=None):
        """Extract the colour bins from the image based on the cell size."""
        image = self.get_spatial_features(window)
        if self.colorspace != 'RGB':
            converter = getattr(cv2, "COLOR_" + self.colorspace + "2RGB")
            image = cv2.cvtColor(image, converter)
        return image

    def get_hog_visualisation(self):
        """Returns the Histogram of Oriented Gradients """
        if self.hog_image is None:
            raise ValueError('There is no HOG visualisation available.')
        else:
            return self.hog_image

    def get_histogram_visualisation(self, window=None):
        """Plot the histogram for the current window."""
        hist = self.get_histogram_features(window).tolist()
        if len(self.colorspace) == 5:
            series_labels = [self.colorspace[0], self.colorspace[1:3], 
                             self.colorspace[3:]]
        else:
            series_labels = [label for label in self.colorspace]
        plot_histogram(hist, 'Colour Histogram', series_labels)




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


class Classifier():
    """Base class for the ML models to identify cars."""
    def __init__(self, classifier):
        self.clf = classifier
        self.feature_scaler = None

    def __call__(self):
        raise NotImplementedError

    def train(self, features_train, labels_train, features_test,
              labels_test):
        """Trains the classifier with the supplied training data."""
        features_train = self.normalise(features_train)
        features_test = self.normalise(features_test)
        self.clf.fit(features_train, labels_train)

        start = time.time()
        test_set_accuracy = self.clf.score(features_test, labels_test)
        self.predict(features_test)
        end = time.time()

        print('Classifier test accuracy = ',
              round(test_set_accuracy, 4),
              '(took', round(end - start, 2), 'seconds on', len(features_test),'entries)')

    def predict(self, features):
        """Predicts the labels for a given list of input features."""
        features = self.normalise(features)
        return self.clf.predict(features)

    def normalise(self, features):
        """Normalises the input feature set. """
        if self.feature_scaler is None:
            raise ValueError('The feature scaler has not yet been fit!')
        else:
            return self.feature_scaler.transform(features)

    def fit_feature_scaler(self, training_features):
        """Fit a feature scaler to the training data to easily normalise inputs
        later.
        """
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
                      'max_iter': -1}
        return cls(LinearSVC())


class GridSearch():
    """Tool to search and image with a sliding window approach."""
    def __init__(self, search_windows, search_areas, image_analyser,
                 training_resolution=64):
        self.search_areas = search_areas
        self.search_windows = search_windows
        self.image_analyser = image_analyser
        self.training_res = training_resolution
        self.pix_in_cell = image_analyser.hog_params['pix_per_cell']
        self.crawl_result = None

    def  get_nsteps(self, search_window):
        """Get the total number of steps for a search window."""
        steps_x = self.get_steps(search_window, 'x')
        steps_y = self.get_steps(search_window, 'y')

        if steps_x - int(steps_x) != 0:
            print('Please ensure that your steps in x are set up such that the'
                  + ' search fits perfectly into the search area "'
                  + search_window['search_area'] + '".')
            print(steps_x)
        if steps_y - int(steps_y) != 0:
            print('Please ensure that your steps in y are set up such that the'
                  + ' search fits perfectly into the search area "'
                  + search_window['search_area'] + '".')
            print(steps_y)

        return int(steps_x) * int(steps_y)

    def get_steps(self, search_window, direction):
        """Get the number of steps for a search window along a direction."""
        i = 0 if direction == 'x' else 1
        area = self.search_areas[search_window['search_area']]
        window_length = search_window['size'][i]
        area_length = area[1][i] - area[0][i]
        return ((area_length - window_length) 
                / search_window['step_' + direction])

    def get_window_position(self, step, window):
        """Get the absolute position of the window based on the step number as
        in image cell dimensions.
        """
        (left, top), _ = self.get_relative_window_coordinates(window, step)
        scale_factor = ((self.pix_in_cell * window['size'][0])
                        / self.training_res)
        left, top = (int(left * scale_factor),
                     int(top * scale_factor))

        search_area = self.search_areas[window['search_area']]
        top_offset = search_area[0][1]
        left_offset = search_area[0][0]

        return ((left + left_offset), 
                (top + top_offset))

    def update_image_analyser(self, img, window):
        """Update the image frames with the supplied image."""
        if self.image_analyser is None:
            raise ValueError('No image analyser object exists.')
        else:
            area_of_interest = self.search_areas[window['search_area']]
            area_of_interest = self.convert_to_px(area_of_interest)
            img_new = get_area_of_interest(img, area_of_interest)
            window_size = window['size'][0] * self.pix_in_cell
            scale_factor = self.training_res / window_size
            new_size = (int(img_new.shape[1] * scale_factor), 
                        int(img_new.shape[0] * scale_factor))
            img = cv2.resize(img_new, new_size)

            return self.image_analyser(img)
    
    def convert_to_px(self, coordinates):
        """Converts coordinates expressed in cells to pixels."""
        converted = []
        for entry in coordinates:
            if type(entry) is list or type(entry) is tuple:
                converted_entry = self.convert_to_px(entry)
            else:
                converted_entry = entry * self.pix_in_cell
            converted.append(converted_entry)
        return converted

    def sliding_search(self, img, classifier):
        """Returns a list of windows where a vehicle was found."""
        windows_with_vehicles = []
        analyser = self.image_analyser
        
        for name, window in self.search_windows.items():
            n_steps = self.get_nsteps(window)
            self.update_image_analyser(img, window)

            for step in range(n_steps):
                position = self.get_window_position(step, window)
                position = self.convert_to_px(position)
                
                window_relative = self.get_relative_window_coordinates(window,
                                                                       step)
                features = analyser.get_image_features(window=window_relative)
                label = classifier.predict(features.reshape(1, -1))[0]

                if label == 'vehicles':
                    windows_with_vehicles.append({'position': position,
                                                  'label': label,
                                                  'window': name})

        return windows_with_vehicles

    def get_relative_window_coordinates(self, window, step):
        """Returns window coordinates in scaled cells (based on training 
        resolution) relative to scaled area of interest.
        """
        steps_x = self.get_steps(window, 'x')
        step_x = int(step % steps_x)
        step_y = int(step / steps_x)
        training_res_cells = int(self.training_res / self.pix_in_cell)

        scaled_x_step = int(window['step_x'] / window['size'][0]
                            * training_res_cells)
        scaled_y_step = int(window['step_y'] / window['size'][0]
                            * training_res_cells)

        left = scaled_x_step * step_x
        top = scaled_y_step * step_y

        return ((left, top), 
                (left + training_res_cells, top + training_res_cells))

    def highlight_windows(self, img, search_results, color='yellow'):
        """Draws a rectangle around windows in the search result."""
        annotated_img = np.copy(img)
        for result in search_results:
            window = self.search_windows[result['window']]
            window_dimensions = self.convert_to_px(window['size'])
            width, height = (window_dimensions[0], window_dimensions[1])
            left, top = result['position']
            window_area = [(left, top), (left + width, 
                                         top + height)]
            
            annotated_img = quick_rectangle(annotated_img, window_area,
                                            color=color, opacity=0.4,
                                            filled=True, thickness=2)

        return annotated_img


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

def save_image(image, output_folder='./output_images/',
               name='Current_Image', colorspace='RGB'):
    """Save image to a file."""
    save_path = os.path.join(output_folder, name) + '.jpg'

    if colorspace != 'RGB':
        converter = getattr(cv2, "COLOR_" + colorspace + "2RGB")
        image = cv2.cvtColor(image, converter)

    cv2.imwrite(save_path, image)

def plot_histogram(values, chart_title, series_labels):
    """Plots the supplied colour histogram as a barchart."""
    x = np.arange(0, 260, 260/len(values[0]))
    n_series = len(series_labels)
    colors = [[0.88, 0.75, 0.35, 0.7],
              [0.75, 0.63, 0.25, 0.7],
              [0.60, 0.47, 0.10, 0.7]]

    plt.subplots(1, n_series, figsize=(10, 3), dpi=120)
    plt.title(chart_title)

    if n_series > 1:
        assert n_series == len(values), "Supplied data series must have corresponding labels."
        width =  260/len(values[0])
        for i, series_label in enumerate(series_labels):
            plt.subplot(1, n_series, i+1)
            plt.bar(x, values[i][:], width, color=colors[i], label=series_label)
            plt.legend(loc='best')
            plt.xlim(0, 256)
    else:
        plt.bar(x, values, align='center', alpha=0.5)
    
    plt.show()
