class EstimationSelector(object):
    def __init__(self):
        self.count = 0  # a counter for the number of frames that the distance is larger than the threshold
        self.isCorrecting = False  # is correcting the result or not

    def distance(self, point1, point2):
        # calculate the distance between two points
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

    def select(self, candidate, kalman_result, threshold=5, max_count=3):
        # calculate the distance between the candidate and the kalman result
        distance = self.distance(candidate, kalman_result)
        # print('candidate:', candidate)
        # print('kalman_result:', kalman_result)

        # check if the distance is larger than the threshold
        if distance <= threshold:
            self.isCorrecting = False  # quit the correcting state
            self.count = 0
            print('result:', candidate)
            return candidate
        else:
            if not self.isCorrecting:
                self.count += 1
                if self.count >= max_count:
                    self.isCorrecting = True  # enter the correcting state
                    self.count = 0

            if self.isCorrecting:
                print('result:', candidate)
                return candidate
            else:
                print('result:', kalman_result)
                return kalman_result
