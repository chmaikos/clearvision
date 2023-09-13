import cv2
from PIL import Image
from pytesseract import pytesseract

from clearvision.cleaning import TextCleaning
from clearvision.detection import TextDetection


class OCR:
    def __init__(self):
        self.text_detection = TextDetection()
        self.text_cleaning = TextCleaning()

    def perform_ocr(self, image_path):
        image = cv2.imread(image_path)
        bounding_boxes = self.text_detection.detect_text_areas(image)
        cleaned_images = self.text_cleaning.clean_text_areas(image, bounding_boxes)

        results = []
        for i, cleaned_image in enumerate(cleaned_images):
            # Display the cleaned image in a debug window

            pil_image = Image.fromarray(cleaned_image.astype("uint8"))
            text = pytesseract.image_to_string(pil_image)
            results.append(
                {"text": text, "coordinates": bounding_boxes[i], "confidence": None}
            )
        return results
