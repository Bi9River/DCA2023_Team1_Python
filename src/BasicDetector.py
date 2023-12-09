import numpy as np
import cv2

def detect(frame, debugMode=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if debugMode:
        cv2.imshow('gray', gray)

    img_edges = cv2.Canny(gray, 50, 190, 3)
    if debugMode:
        cv2.imshow('img_edges', img_edges)

    ret, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)
    if debugMode:
        cv2.imshow('img_thresh', img_thresh)

    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_radius_thresh = 3
    max_radius_thresh = 50

    result_positions = []
    for idx, c in enumerate(contours):
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        if min_radius_thresh < radius < max_radius_thresh:
            label = f"basic_object_{idx}"
            x_pos = int(x)
            y_pos = int(y)
            result_positions.append([x_pos, y_pos, label])

    if debugMode:
        cv2.imshow('contours', img_thresh)

    return result_positions
