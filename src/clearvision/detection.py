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
        image = cv2.copy(orig_image)
        (newW, newH) = (640, 640)
        image = cv2.resize(image, (newW, newH))
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False
        )
        self.net.setInput(blob)

        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

        (scores, geometry) = self.net.forward(layerNames)
        (rects, confidences) = self.decode_predictions(scores, geometry)
        bounding_boxes = non_max_suppression(np.array(rects), probs=confidences)

        # Draw rectangles on the original image
        result_image = self.draw_bounding_boxes(image, bounding_boxes)

        # Display the image with rectangles
        cv2.imshow("Image with Rectangles", result_image)
        cv2.waitKey(0)  # Wait for a key press to close the window

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

    def draw_bounding_boxes(self, image, bounding_boxes):
        result_image = image.copy()
        for startX, startY, endX, endY in bounding_boxes:
            cv2.rectangle(result_image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        return result_image
