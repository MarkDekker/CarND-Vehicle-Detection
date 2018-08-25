"""Module to filter searches for unwanted results."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
from utilityfun import plot_image, quick_rectangle

class SearchFilter():
    """Provides filtering functionality to handle duplicates and false
    positives.
    """

    def __init__(self, image, search_windows):
        self.image = image
        self.search_windows = search_windows
        self.hitmap_shape = self.get_hitmap_shape(image, len(search_windows))
        self.hitmap = None
        self.reset()
        self.detected_cars = None # array with box coords, centroid, 'velocity'

    def update(self, image, search, threshold=2):
        """Update the image and latest search results."""
        self.image = image
        if self.detected_cars is not None:
            for car in self.detected_cars:
                car['status'] = 'previous'
        self.log_search(search)
        self.filter(threshold=threshold)
        self.remove_old_cars()
        self.hitmap = self.hitmap * 0.4

    def reset(self):
        """Resets the filter for a new search."""
        self.hitmap = np.zeros(self.hitmap_shape)

    def get_hitmap_shape(self, image, n_windows):
        """Defines the shape of the hitmap array.

        The "hitmap" (classifier positive hits) array has the width and
        height of the image in pixels. The depth is determined by how many
        different types of search windows there are plus one with a total sum.
        """

        height = image.shape[0]
        width = image.shape[1]
        channels = n_windows + 1

        return (height, width, channels)

    def log_search(self, search):
        """Logs the pixels for which the search returned a hit."""

        windows = self.search_windows

        for result in search:
            left = result['position'][0]
            top = result['position'][1]
            width = windows[result['window']]['size'][0] * 8
            height = windows[result['window']]['size'][1] * 8
            i = list(windows.keys()).index(result['window'])

            self.hitmap[top:(top + height), left:(left + width), i] += 1
            self.hitmap[top:(top + height), left:(left + width), -1] += 1

    def visualise_hitmap(self, hitmap_channel=-1):
        """Plot hitmap as heatmap on top of image."""
        hitmap_img = self.hitmap[:, :, hitmap_channel]
        overlay_cmap = transparent_cmap(plt.cm.get_cmap('plasma'))
        plt.figure(figsize=(12, 7))
        plot_image(self.image)
        plt.imshow(hitmap_img, cmap=overlay_cmap)
        plt.colorbar()
        plt.show()
        

    def filter(self, threshold=2):
        """Filter the search hits based on a threshold value."""
        thresholded = np.where(self.hitmap > threshold, self.hitmap, 0)
        labels = label(thresholded)
        boxes = self.get_bounding_boxes(labels)

        for box in boxes:
            car_centroid = self.get_centroid(box)
            if not self.update_existing_car(car_centroid, box):
                self.create_new_car(car_centroid, box)

        self.remove_old_cars()

    def get_centroid(self, box):
        """Gets the centroid from an given box."""
        x_pos = (box[1][0] - box[0][0]) / 2 + box[0][0]
        y_pos = (box[1][1] - box[0][1]) / 2 + box[0][1]
        return (x_pos, y_pos)

    def get_bounding_boxes(self, filter_result):
        """Determines the bounding boxes around filtered cars."""
        cars = []
        for car in range(1, filter_result[1]+1):
            carpixels = (filter_result[0] == car).nonzero()
            carpixels_y = np.array(carpixels[0])
            carpixels_x = np.array(carpixels[1])
            cars.append(((np.min(carpixels_x), np.min(carpixels_y)),
                         (np.max(carpixels_x), np.max(carpixels_y))))

        return cars

    def update_existing_car(self, centroid, box, radius=50):
        """Update an existing car if the new centroid is within range."""
        updated = False
        if self.detected_cars is not None:
            for car in self.detected_cars:
                if self.get_distance(centroid, car['centroid']) < radius:
                    car['box'] = box
                    car['velocity'] = (centroid[0] - car['centroid'][0],
                                    centroid[1] - car['centroid'][1])
                    car['centroid'] = centroid
                    car['status'] = 'current'
                    updated = True
        return updated

    @staticmethod
    def get_distance(point_1, point_2):
        """Gets the distance between two points"""
        d_x = point_2[0] - point_1[0]
        d_y = point_2[1] - point_1[1]

        return (d_x ** 2 + d_y ** 2) ** 0.5

    def create_new_car(self, centroid, box):
        """Add a car to the list of detected cars."""
        car = {'box': box, 'centroid': centroid, 'velocity': [0, 0],
               'status': 'current'}
        if self.detected_cars is None:
            self.detected_cars = [car]
        else:
            self.detected_cars.append(car)

    def remove_old_cars(self):
        """Check whether cars have not been updated and then removes these."""
        if self.detected_cars is not None:
            for car in self.detected_cars:
                if car['status'] == 'previous':
                    self.detected_cars.remove(car)

    def plot_detected_cars(self, color='yellow'):
        """Draws rectangles around the cars that were detected."""
        annotated_img = np.copy(self.image)
        for car in self.detected_cars:
            window_area = car['box']
            annotated_img = quick_rectangle(annotated_img, window_area,
                                            color=color, opacity=0.6,
                                            filled=True, thickness=2)
        plot_image(annotated_img)
        plt.show()


# --------------------------------------------------------------------------- #

def transparent_cmap(cmap):
    "Create colourmap with increasing alpha values."

    alpha_cmap = cmap
    alpha_cmap._init()
    alpha_cmap._lut[:, -1] = np.linspace(0, 0.8, 259)
    return alpha_cmap
