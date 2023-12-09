import cv2
import numpy as np
from tqdm import tqdm
import time

from KalmanFilter import KalmanFilter
from ObjectDetector import ObjectDetector
from Voter import Voter
from EstimationSelector import EstimationSelector
import BasicDetector

# Video path (also can be image sequence folder path)
video_path = "../resources/Single_Ball_Test_Video.mp4"

def get_frames(is_video, path):
    """
    Get frames from video or image sequence
    :param is_video: set to true if we are using video
    :param path: the path to the video or image sequence folder
    :return: a list of frames
    """
    frames = []
    if is_video:
        VideoCap = cv2.VideoCapture(path)
        print('VideoCap.isOpened():', VideoCap.isOpened())
        print('Start reading video')
        frame_count = int(VideoCap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in tqdm(range(frame_count)):
            ret, frame = VideoCap.read()
            if ret:
                frames.append(frame)
            else:
                break
        print('Finish reading video')
    else:
        for i in tqdm(range(0, 100)):
            # TODO - Change the format of the image sequence if needed
            frames.append(path + '\\%08d.jpg' % i) # GOT-10k dataset format
    return frames

def main():
    frames = []
    is_video = True # Set to false if we are using sequences of images
    frames_cap1 = get_frames(is_video, video_path)
    frames_cap2 = get_frames(is_video, video_path)
    frames_cap3 = get_frames(is_video, video_path)
    frames = frames_cap1

    OD_1 = ObjectDetector()
    OD_2 = ObjectDetector()
    OD_3 = ObjectDetector()
    OD = ObjectDetector()

    # Save kalman filter, estimation selector and last results of Kalman filter for each object, indexed by label
    kalman_filters = {}
    estimation_selectors = {}
    kf_last_results = {}

    counter = 0
    for i in range(len(frames_cap1)):
        print('frame:', counter)
        counter += 1

        # DEPENDABILITY #1 - Three Input Sources
        frame_cap1 = frames_cap1[i]
        frame_cap2 = frames_cap2[i]
        frame_cap3 = frames_cap3[i]

        detection_results_1 = OD_1.detect(frame_cap1)
        detection_results_2 = OD_2.detect(frame_cap2)
        detection_results_3 = OD_3.detect(frame_cap3)
        detection_results = OD.detect(frame_cap1)

        # DEPENDABILITY #2 - Voting
        detection_results_dependable = []
        for j in range(len(detection_results_1)):
            x_pos = Voter().vote([[detection_results_1[j][0], detection_results_2[j][0], detection_results_3[j][0]]])[0].item()
            y_pos = Voter().vote([[detection_results_1[j][1], detection_results_2[j][1], detection_results_3[j][1]]])[1].item()
            label = Voter().vote([[detection_results_1[j][2], detection_results_2[j][2], detection_results_3[j][2]]], type='label')
            detection_results_dependable.append([x_pos, y_pos, label])

        for x_pos, y_pos, label in detection_results_dependable:
            if label not in kalman_filters:
                # Create new Kalman Filter, Estimation Selector and last results of Kalman filter for this new object
                kalman_filters[label] = KalmanFilter(0.1, 0, 0, 0.2, 0.01, 0.01)
                estimation_selectors[label] = EstimationSelector()
                kf_last_results[label] = [x_pos, y_pos]

            # Corresponding Kalman Filter, Estimation Selector and last results of Kalman filter for this object
            KF = kalman_filters[label]
            ES = estimation_selectors[label]
            kf_last_result = kf_last_results[label]

            # DEPENDABILITY #3 - Estimation Selection
            x_pos_estimated, y_pos_estimated = ES.select([x_pos, y_pos], kf_last_result)
            # x_pos_estimated, y_pos_estimated = int(x_pos), int(y_pos) # Without Estimation Selection

            # Update & Predict using Kalman Filter
            KF.update(np.array([[x_pos_estimated], [y_pos_estimated]]))
            (_x, _y) = KF.predict()
            kf_last_results[label] = [_x[0, 0], _y[0, 0]]
            x = int(_x[0, 0])
            y = int(_y[0, 0])

            # Draw the detected object and predicted position on the frame
            cv2.circle(frame_cap1, (x_pos, y_pos), 10, (0, 191, 255), 3)
            cv2.rectangle(frame_cap1, (int(x - 10), int(y - 10)), (int(x + 10), int(y + 10)), (255, 0, 0), 2)
            cv2.putText(frame_cap1, label, (x_pos + 15, y_pos - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 191, 255), 2)

        # Resize window
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame", 1280, 720)
        cv2.imshow('Frame', frame_cap1)
        cv2.waitKey(1)
        # time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()