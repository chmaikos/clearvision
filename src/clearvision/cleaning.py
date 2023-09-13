import logging

import cv2


class TextCleaning:
    def clean_text_areas(self, image, bounding_boxes):
        cleaned_images = []
        for startX, startY, endX, endY in bounding_boxes:
            # Extract the region of interest
            roi = image[startY:endY, startX:endX]
            cv2.imshow("Debug Window", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Check if the ROI is empty
            if roi is None or roi.size == 0:
                logging.warning(
                    f"Warning: \
                        Empty ROI detected at coordinates \
                        (({startX}, {startY}), \
                        ({endX}, {endY}))"
                )
                continue
            # Convert the ROI to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # binary = cv2.adaptiveThreshold(gray, 255,
            #                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                cv2.THRESH_BINARY, 11, 2)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            cleaned_images.append(gray)
        return cleaned_images
