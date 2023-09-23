import logging
from typing import Any, Dict, Union

import keras_ocr
import numpy as np


class OCR:
    def __init__(self, scale_factor: float = 2.0):
        self.pipeline = keras_ocr.pipeline.Pipeline(scale=scale_factor)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def perform_ocr(self, image_input: Union[str, np.ndarray]) -> Union[Dict, None]:
        img = self._read_image(image_input)
        if img is None:
            return None

        try:
            ocr_results = self.pipeline.recognize([img])[0]
        except Exception as e:
            self.logger.error(f"Error performing OCR: {e}")
            return None

        return self._process_results(ocr_results)

    def _read_image(
        self, image_input: Union[str, np.ndarray]
    ) -> Union[np.ndarray, None]:
        try:
            return (
                keras_ocr.tools.read(image_input)
                if isinstance(image_input, str)
                else image_input
            )
        except Exception as e:
            self.logger.error(f"Error reading image: {e}")
            return None

    def _process_results(self, ocr_results: Any) -> Union[Dict, None]:
        results = [
            {"text": " ".join(text.split()), "coordinates": box.astype(int).tolist()}
            for text, box in ocr_results
        ]
        if not results:
            self.logger.warning("No text found.")
            return None
        return {"ocr_result": results}
