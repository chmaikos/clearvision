from typing import List, Tuple

import cv2


class TextCleaning:
    def __init__(self):
        pass

    def clean_text_areas(
        self, image_path: str,
        bounding_boxes: List[Tuple[int, int, int, int]]
            ) -> List[cv2.UMat]:
        """
        Clean and preprocess the detected text areas to facilitate OCR.

        Args:
        - image_path (str): The path to the image file.
        - bounding_boxes (list of tuple): A list of detected text 
        areas with their coordinates.

        Returns:
        - list of cv2.UMat: A list of cleaned text areas ready for OCR.
        """
        # Load the original image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        cleaned_images = []
        for box in bounding_boxes:
            x, y, w, h = box

            # Step 1: Extract the region of interest (ROI) from the original image
            roi = image[y:y+h, x:x+w]

            # Step 2: Convert the ROI to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Step 3: Apply adaptive thresholding to binarize the image
            binary_roi = cv2.adaptiveThreshold(gray_roi, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 11, 2)

            # Step 4: Apply morphological operations to clean the image (optional)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel)

            # Step 5: Denoise the image using Non-Local Means Denoising (optional)
            cleaned_roi = cv2.fastNlMeansDenoising(cleaned_roi, None, 30, 7, 21)

            cleaned_images.append(cv2.UMat(cleaned_roi))

        return cleaned_images
