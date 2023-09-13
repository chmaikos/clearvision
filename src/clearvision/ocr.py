from pytesseract import pytesseract
from PIL import Image
from clearvision.detection import TextDetection
from clearvision.cleaning import TextCleaning
import cv2


class OCR:
    def __init__(self):
        self.text_cleaning = TextCleaning()

    def perform_ocr(self, image_path):
        text_detection = TextDetection()
        image = cv2.imread(image_path)
        bounding_boxes = text_detection.detect_text_areas(image_path)
        cleaned_images = self.text_cleaning.clean_text_areas(image, bounding_boxes)

        results = []
        for i, cleaned_image in enumerate(cleaned_images):
            pil_image = Image.fromarray(cleaned_image.astype('uint8'))
            text = pytesseract.image_to_string(pil_image)
            results.append({"text": text,
                            "coordinates": bounding_boxes[i],
                            "confidence": None})
        return results
