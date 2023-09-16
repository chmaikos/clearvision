import cv2
from PIL import Image
from pytesseract import pytesseract

from clearvision.cleaning import TextCleaning
from clearvision.detection import TextDetection


class OCR:
    def __init__(self):
        pass

    def perform_ocr(self, image_path):
        text_detection = TextDetection()
        text_cleaning = TextCleaning()

        image = cv2.imread(image_path)
        bounding_boxes = text_detection.detect_text_areas(image)
        cleaned_images = text_cleaning.clean_text_areas(image, bounding_boxes)

        results = []
        for i, cleaned_image in enumerate(cleaned_images):
            pil_image = Image.fromarray(cleaned_image.astype("uint8"))

            # Get detailed OCR results including confidence scores
            ocr_result = pytesseract.image_to_data(
                pil_image, output_type=pytesseract.Output.DICT
            )

            # Calculate mean confidence
            confidences = [conf for conf in ocr_result["conf"] if conf != -1]
            mean_confidence = (
                round(sum(confidences) / len(confidences), 1) if confidences else 0.0
            )

            # Get the full OCR text
            text = " ".join([text for text in ocr_result["text"] if text])

            # Get the most suspicious characters (with the lowest confidence)
            suspicious_chars = []
            for j, word in enumerate(ocr_result["text"]):
                if ocr_result["conf"][j] != -1 and ocr_result["conf"][j] < 80:
                    for k, char in enumerate(word):
                        char_conf = pytesseract.image_to_data(
                            pil_image.crop(
                                (
                                    ocr_result["left"][j] + k,
                                    ocr_result["top"][j],
                                    ocr_result["left"][j] + k + 1,
                                    ocr_result["top"][j] + ocr_result["height"][j],
                                )
                            ),
                            output_type=pytesseract.Output.DICT,
                        )["conf"][0]
                        if char_conf < 80:
                            suspicious_chars.append(
                                {"char": char, "position_in_word": k + 1}
                            )

            results.append(
                {
                    "text": text,
                    "coordinates": bounding_boxes[i].tolist(),
                    "mean_confidence": mean_confidence,
                    "suspicious_chars": suspicious_chars,
                }
            )
        return results
