import os

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression


class TextDetection:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", "frozen_east_text_detection.pb")
        self.net = cv2.dnn.readNet(model_path)

    def detect_text_areas(self, orig_image):
        image = orig_image.copy()
        (origH, origW) = orig_image.shape[:2]
        (newW, newH) = (640, 640)
        rW = origW / float(newW)
        rH = origH / float(newH)

        image = cv2.resize(image, (newW, newH))
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        self.net.setInput(blob)

        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        (scores, geometry) = self.net.forward(layerNames)
        (rects, confidences) = self.decode_predictions(scores, geometry)
        expanded_bounding_boxes = []
        expansion_factor = 2  # Adjust this value to get the desired expansion
        for startX, startY, endX, endY in rects:
            width = endX - startX
            height = endY - startY

            startX = int(max(0, startX - (width * (expansion_factor - 1)) / 2))
            startY = int(max(0, startY - (height * (expansion_factor - 1)) / 2))
            endX = int(min(origW, endX + (width * (expansion_factor - 1)) / 2))
            endY = int(min(origH, endY + (height * (expansion_factor - 1)) / 2))

            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            expanded_bounding_boxes.append((startX, startY, endX, endY))

        bounding_boxes = non_max_suppression(
            np.array(expanded_bounding_boxes), probs=confidences
        )

        return bounding_boxes

    def decode_predictions(self, scores, geometry):
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            for x in range(0, numCols):
                if scoresData[x] < 0.5:
                    continue

                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        return (rects, confidences)
