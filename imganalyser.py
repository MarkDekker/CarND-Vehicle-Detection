"""Image and subimage analysis routines."""

import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt

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



#-----------------------------------------------------------------------------#
#                           Utility functions
#-----------------------------------------------------------------------------#

def plot_histogram(values, chart_title, series_labels):
    """Plots the supplied colour histogram as a barchart."""
    x_val = np.arange(0, 260, 260/len(values[0]))
    n_series = len(series_labels)
    colors = [[0.88, 0.75, 0.35, 0.7],
              [0.75, 0.63, 0.25, 0.7],
              [0.60, 0.47, 0.10, 0.7]]

    plt.subplots(1, n_series, figsize=(10, 3), dpi=120)
    plt.title(chart_title)

    if n_series > 1:
        assert n_series == len(values), "Supplied data series must have corresponding labels."
        width = 260/len(values[0])
        for i, series_label in enumerate(series_labels):
            plt.subplot(1, n_series, i+1)
            plt.bar(x_val, values[i][:], width, color=colors[i], label=series_label)
            plt.legend(loc='best')
            plt.xlim(0, 256)
    else:
        plt.bar(x_val, values, align='center', alpha=0.5)

    plt.show()
