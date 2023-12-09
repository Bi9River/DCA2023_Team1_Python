import torch
import cv2
import numpy as np


class ObjectDetector(object):
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)
        self.sift = cv2.SIFT_create(nfeatures=500, contrastThreshold=0.04, edgeThreshold=10)
        self.flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 100})
        self.previous_descriptors = []
        self.previous_labels = []

    def extract_features(self, frame, results):
        """
        Extract features around the detected objects
        :param frame: video frame or image sequence frame
        :param results: object centers and labels
        :return: descriptors of the detected objects
        """
        descriptors_list = []
        for result in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, result[:4])
            crop_img = frame[y1:y2, x1:x2]
            _, descriptors = self.sift.detectAndCompute(crop_img, None)
            descriptors_list.append(descriptors)
        return descriptors_list

    def match_objects(self, current_descriptors):
        """
        Match objects in the current frame with objects in the previous frame
        :param current_descriptors: descriptors of objects in the current frame
        :return: matches
        """
        matches = []

        # Check if there are previous descriptors
        if not self.previous_descriptors or not current_descriptors:
            return matches

        for current_desc in current_descriptors:
            if current_desc is not None and current_desc.size > 0:
                best_match_idx = -1
                min_distance = float('inf')

                for idx, previous_desc in enumerate(self.previous_descriptors):
                    if previous_desc is not None and previous_desc.size > 0:
                        # Calculate the distance between the current descriptor and each previous descriptor
                        for prev_desc in previous_desc:
                            distance = np.linalg.norm(current_desc - prev_desc)
                            if distance < min_distance:
                                min_distance = distance
                                best_match_idx = idx

                matches.append(best_match_idx)
            else:
                matches.append(-1)

        return matches

    def detect(self, frame):
        """
        Detect objects in the frame
        :param frame: image frame
        :return: result centers and labels
        """
        imgs = [frame]
        results = self.model(imgs)
        result_positions = []
        current_descriptors = self.extract_features(frame, results)

        # If there are no previous descriptors, then we are in the first frame
        if not self.previous_descriptors:
            for i, result in enumerate(results.xyxy[0].numpy()):
                x_pos = int((result[0] + result[2]) / 2)
                y_pos = int((result[1] + result[3]) / 2)
                label = f"{results.names[result[5]]}_{i}"
                result_positions.append([x_pos, y_pos, label])
        else:
            # Associate the objects in the current frame with the objects in the previous frame
            matches = self.match_objects(current_descriptors)

            for i, result in enumerate(results.xyxy[0].numpy()):
                x_pos = int((result[0] + result[2]) / 2)
                y_pos = int((result[1] + result[3]) / 2)
                label = f"{results.names[result[5]]}_{matches[i]}"
                result_positions.append([x_pos, y_pos, label])

        # Update previous descriptors as current descriptors
        self.previous_descriptors = current_descriptors

        return result_positions