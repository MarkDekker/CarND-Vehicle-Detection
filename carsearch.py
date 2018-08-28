"""Search routines within supplied images."""

import numpy as np
import cv2

from utilityfun import * #quick_rectangle #project utility

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
            img_new = self.get_area_of_interest(img, area_of_interest)
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
            if isinstance(entry, list) or isinstance(entry, tuple):
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

                # current_window = (position, (position[0] + 8 * window['size'][0],
                #                              position[1] + 8 * window['size'][1]))
                # window_name = "temp_" + name + "_" + str(step)
                # #save_image(self.get_area_of_interest(img, current_window), name=window_name)

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

        scaled_x_step = (window['step_x'] / window['size'][0]
                         * training_res_cells)
        scaled_y_step = (window['step_y'] / window['size'][0]
                         * training_res_cells)

        left = int(scaled_x_step * step_x)
        top = int(scaled_y_step * step_y)

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

    @staticmethod
    def get_area_of_interest(image, search_area):
        """Reduces the image to the area of interest."""
        top = search_area[0][1]
        bottom = search_area[1][1]
        left = search_area[0][0]
        right = search_area[1][0]

        return image[top:bottom, left:right, :]
