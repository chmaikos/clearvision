import logging

import cv2

from clearvision.imageproc.toolkit import (
    adjust_contrast_brightness,
    denoise_image,
    deskew,
    thresholding,
)


class TextCleaning:
    def clean_text_areas(self, image, bounding_boxes):
        cleaned_images = []
        for startX, startY, endX, endY in bounding_boxes:
            # Extract the region of interest
            roi = image[startY:endY, startX:endX]

            # Check if the ROI is empty
            if roi is None or roi.size == 0:
                logging.warning(
                    f"Warning: \
                        Empty ROI detected at coordinates \
                        (({startX}, {startY}), \
                        ({endX}, {endY}))"
                )
                continue

            roi = denoise_image(roi)

            roi = deskew(roi, method="moments")
            roi = adjust_contrast_brightness(roi, method="clahe")

            roi = thresholding(roi, method="otsu")

            # Debug: Display the processed ROI
            cv2.imshow("Debug Window", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cleaned_images.append(roi)
        return cleaned_images
