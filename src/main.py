import cv2
import numpy as np
from tqdm import tqdm
import time
from matplotlib import pyplot as plt

from KalmanFilter import KalmanFilter
from ObjectDetector import ObjectDetector
from Voter import Voter
from EstimationSelector import EstimationSelector
import BasicDetector

# Video path (also can be image sequence folder path)
# video_path = "../resources/Const_Velocity_Const_Motion.mp4"
# # VIDEO_PATH = "../resources/moving_circle_1.mp4"
# TEST_CATEGORY = "Const_Velocity_Const_Motion"
# VIDEO_PATH = "../resources/"+TEST_CATEGORY+".mp4"
#
# print('VIDEO_PATH:', VIDEO_PATH)
# print('TEST_CATEGORY:', TEST_CATEGORY)

# Object Detector or Basic Detector
IS_USING_OBJECT_DETECTOR = True

ground_truth_positions = []
predicted_positions = []


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

def main(test_category, video_path):
    frames = []
    ground_truth_positions.clear()
    predicted_positions.clear()
    is_video = True # Set to false if we are using sequences of images
    frames_cap1 = get_frames(is_video, video_path)
    frames_cap2 = get_frames(is_video, video_path)
    frames_cap3 = get_frames(is_video, video_path)
    frames_cap_ground_truth = get_frames(is_video, video_path)

    OD_1 = ObjectDetector()
    OD_2 = ObjectDetector()
    OD_3 = ObjectDetector()
    OD = ObjectDetector()

    # Save kalman filter, estimation selector and last results of Kalman filter for each object, indexed by label
    kalman_filters = {}
    estimation_selectors = {}
    kf_last_results = {}

    counter = 0
    print('Total frames:', len(frames_cap1))

    # print frame width and height
    print('frame width:', frames_cap1[0].shape[1])
    print('frame height:', frames_cap1[0].shape[0])
    for i in range(len(frames_cap1)):
    # for i in range(100):
        print('frame:', counter)
        counter += 1

        # DEPENDABILITY #1 - Three Input Sources
        frame_cap1 = frames_cap1[i]
        frame_cap2 = frames_cap2[i]
        frame_cap3 = frames_cap3[i]
        frame_cap_ground_truth = frames_cap_ground_truth[i]

        detection_results_1 = None
        detection_results_2 = None
        detection_results_3 = None
        if IS_USING_OBJECT_DETECTOR:
            filtered_detection_results_1 = OD_1.detect(frame_cap1)
            filtered_detection_results_2 = OD_2.detect(frame_cap2)
            filtered_detection_results_3 = OD_3.detect(frame_cap3)
            detection_results_1 = []
            detection_results_2 = []
            detection_results_3 = []
        else:
            detection_results_1 = BasicDetector.detect(frame_cap1)
            detection_results_2 = BasicDetector.detect(frame_cap2)
            detection_results_3 = BasicDetector.detect(frame_cap3)



        # Record Ground Truth
        detection_results_ground_truth = OD.detect(frame_cap_ground_truth)
        print('detected:', detection_results_ground_truth)
        if len(detection_results_ground_truth) != 0:
            # for each frame, load the sports ball position
            for gt in detection_results_ground_truth:
                # check if the label starts with sports
                if gt[2].startswith('sports'):
                    x_ground_truth = detection_results_ground_truth[0][0]
                    y_ground_truth = detection_results_ground_truth[0][1]
                    ground_truth_positions.append([x_ground_truth, y_ground_truth])
        else:
            ground_truth_positions.append([-1, -1])
        print('detection_results_ground_truth:', ground_truth_positions)

        if len(filtered_detection_results_1) != 0:
            for dt in filtered_detection_results_1:
                # check if the label starts with sports
                if dt[2].startswith('sports'):
                    x_pos = dt[0]
                    y_pos = dt[1]
                    label = dt[2]
                    detection_results_1.append([x_pos, y_pos, label])
        else:
            detection_results_1.append([-1, -1, ''])

        if len(filtered_detection_results_2) != 0:
            for dt in filtered_detection_results_2:
                # check if the label starts with sports
                if dt[2].startswith('sports'):
                    x_pos = dt[0]
                    y_pos = dt[1]
                    label = dt[2]
                    detection_results_2.append([x_pos, y_pos, label])
        else:
            detection_results_2.append([-1, -1, ''])

        if len(filtered_detection_results_3) != 0:
            for dt in filtered_detection_results_3:
                # check if the label starts with sports
                if dt[2].startswith('sports'):
                    x_pos = dt[0]
                    y_pos = dt[1]
                    label = dt[2]
                    detection_results_3.append([x_pos, y_pos, label])
        else:
            detection_results_3.append([-1, -1, ''])

        # DEPENDABILITY #2 - Voting
        detection_results_dependable = []
        loop_cnt = min(len(detection_results_1), len(detection_results_2), len(detection_results_3))
        for j in range(loop_cnt):
            candidate1_exist = False
            candidate2_exist = False
            candidate3_exist = False
            if detection_results_1[j][2].startswith('sports'):
                candidate1_exist = True
                x_pos_candidate1 = detection_results_1[j][0]
                y_pos_candidate1 = detection_results_1[j][1]
                label_candidate1 = detection_results_1[j][2]
            if detection_results_2[j][2].startswith('sports'):
                candidate2_exist = True
                x_pos_candidate2 = detection_results_2[j][0]
                y_pos_candidate2 = detection_results_2[j][1]
                label_candidate2 = detection_results_2[j][2]
            if detection_results_3[j][2].startswith('sports'):
                candidate3_exist = True
                x_pos_candidate3 = detection_results_3[j][0]
                y_pos_candidate3 = detection_results_3[j][1]
                label_candidate3 = detection_results_3[j][2]

            x_candidates = []
            y_candidates = []
            label_candidates = []
            if candidate1_exist:
                x_candidates.append(x_pos_candidate1)
                y_candidates.append(y_pos_candidate1)
                label_candidates.append(label_candidate1)
            if candidate2_exist:
                x_candidates.append(x_pos_candidate2)
                y_candidates.append(y_pos_candidate2)
                label_candidates.append(label_candidate2)
            if candidate3_exist:
                x_candidates.append(x_pos_candidate3)
                y_candidates.append(y_pos_candidate3)
                label_candidates.append(label_candidate3)
            print('1 candidate exist:', candidate1_exist, '2 candidate exist:', candidate2_exist, '3 candidate exist:', candidate3_exist)
            print('x_candidates:', x_candidates)
            print('y_candidates:', y_candidates)

            if not candidate1_exist and not candidate2_exist and not candidate3_exist:
                x_candidates.append(-1), x_candidates.append(-1), x_candidates.append(-1)
                y_candidates.append(-1), y_candidates.append(-1), y_candidates.append(-1)
                label_candidates.append('sports ball_0')

            x_pos = Voter().vote(x_candidates).item()
            # x_pos = Voter().vote(x_candidates)[0].item()
            y_pos = Voter().vote(y_candidates).item()
            label = Voter().vote([label_candidate1, label_candidate2, label_candidate3], type='label')
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
            predicted_positions.append([x_pos_estimated, y_pos_estimated])

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
        # cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Frame", 1280, 720)
        # cv2.imshow('Frame', frame_cap1)
        # cv2.waitKey(1)
        # time.sleep(0.01)

    cv2.destroyAllWindows()

    filtered_ground_truth = []
    filtered_predicted = []
    total_cnt = 0
    correct_cnt = 0
    for gt, pred in zip(ground_truth_positions, predicted_positions):
        if not np.array_equal(gt, [-1, -1]) and not np.array_equal(pred, [-1, -1]):
            total_cnt += 1
            filtered_ground_truth.append(gt)
            filtered_predicted.append(pred)
            # check the distance between gt and pred
            if np.linalg.norm(np.array(gt) - np.array(pred)) < 2:
                correct_cnt += 1
            else:
                print('gt:', gt)
                print('pred:', pred)

    print('correct_cnt:', correct_cnt)
    print('total_cnt:', total_cnt)
    # print percentage of correct prediction
    print('percentage of correct prediction:', correct_cnt / total_cnt * 100, '%')

    # save the accuracy, counts to a  single file
    with open('accuracy.txt', 'a') as f:
        f.write(test_category + ': ' + str(correct_cnt / total_cnt * 100) + '%\n')
        f.write('correct_cnt: ' + str(correct_cnt) + '\n')
        f.write('total_cnt: ' + str(total_cnt) + '\n')
        f.write('------------------------\n')

    filtered_ground_truth = np.array(filtered_ground_truth)
    filtered_predicted = np.array(filtered_predicted)

    fig, ax = plt.subplots()
    ax.plot(np.array(filtered_ground_truth)[:, 0], np.array(filtered_ground_truth)[:, 1], 'r--', label='ground truth')
    ax.plot(np.array(filtered_predicted)[:, 0], np.array(filtered_predicted)[:, 1], 'b--', label='predicted')
    ax.legend(loc='upper right')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ymin, ymax = 0, 1080
    xmin, xmax = 0, 1920
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    plt.show()
    # save the figure
    fig.savefig(test_category + '.png', dpi=500)
    fig.clf()

def dispatch():
    # TEST_CATEGORIES = ['acc_lin']
    TEST_CATEGORIES = ['uni_lin', 'uni_cir', 'acc_lin', 'acc_cir', 'Const_Velocity_Const_Motion']
    # TEST_CATEGORIES = ['Const_velocity_Const_Motion', '1', '2', '3', '4']
    for CURRENT_TEST_CATEGORY in TEST_CATEGORIES:
        TEST_CATEGORY = CURRENT_TEST_CATEGORY
        VIDEO_PATH = "../resources/"+TEST_CATEGORY+".mp4"
        print('VIDEO_PATH:', VIDEO_PATH)
        print('TEST_CATEGORY:', TEST_CATEGORY)
        main(TEST_CATEGORY, VIDEO_PATH)

if __name__ == "__main__":
    dispatch()