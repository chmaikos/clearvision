import os
import cv2
import numpy as np


class TextDetection:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'models', 'frozen_east_text_detection.pb')
        self.net = cv2.dnn.readNet(model_path)

    def detect_text_areas(self, image_path):
        image = cv2.imread(image_path)
        orig = image.copy()
        (H, W) = image.shape[:2]
        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)
        image = cv2.resize(image, (newW, newH))
        blob = cv2.dnn.blobFromImage(image, 1.0, (newW, newH),
                                     (123.68, 116.78, 103.94),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)

        # Specify the output layer names
        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]

        # Get the output layers
        (scores, geometry) = self.net.forward(layerNames)

        (rects, confidences) = self.decode_predictions(scores, geometry)
        bounding_boxes = self.get_bounding_boxes(rects, confidences, rW, rH, orig)
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

    def get_bounding_boxes(self, rects, confidences, rW, rH, orig):
        boxes = []
        for _, box in enumerate(rects):
            startX, startY, endX, endY = box
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            boxes.append((startX, startY, endX, endY))
        return boxes
