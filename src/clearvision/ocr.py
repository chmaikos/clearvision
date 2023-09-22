import logging
from typing import Dict, Union

import keras_ocr


class OCR:
    def __init__(self, scale_factor: float = 2.0):
        self.pipeline = keras_ocr.pipeline.Pipeline(scale=scale_factor)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def perform_ocr(self, image_path: str) -> Union[Dict, None]:
        try:
            img = keras_ocr.tools.read(image_path)
        except Exception as e:
            self.logger.error(f"Error reading image: {e}")
            return None

        try:
            ocr_results = self.pipeline.recognize([img])[0]
        except Exception as e:
            self.logger.error(f"Error performing OCR: {e}")
            return None

        results = []
        for text, box in ocr_results:
            full_text = " ".join(text.split())
            results.append(
                {
                    "text": full_text,
                    "coordinates": box.astype(int).tolist(),
                }
            )

        if not results:
            self.logger.warning("No text found.")
            return None

        return {
            "ocr_result": results,
        }
