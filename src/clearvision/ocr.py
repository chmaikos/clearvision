from typing import Any, Dict, Union

import image_ocr as keras_ocr
import numpy as np

from clearvision import logger


class OCR:
    def __init__(
        self,
        scale_factor: float = 2.0,
        recognizer_weights: str = None,
        detector_weights: str = None,
    ):
        self.detector = (
            keras_ocr.detection.Detector(weights_path=detector_weights)
            if detector_weights
            else keras_ocr.detection.Detector()
        )
        self.recognizer = (
            keras_ocr.recognition.Recognizer(weights_path=recognizer_weights)
            if recognizer_weights
            else keras_ocr.recognition.Recognizer()
        )
        self.pipeline = keras_ocr.pipeline.Pipeline(
            detector=self.detector, recognizer=self.recognizer, scale=scale_factor
        )
        logger.info("Pipeline setup complete")

    def perform_ocr(self, image_input: Union[str, np.ndarray]) -> Union[Dict, None]:
        img = self._read_image(image_input)
        if img is None:
            return None

        try:
            ocr_results = self.pipeline.recognize([img])[0]
        except Exception as e:
            logger.error(f"Error performing OCR: {e}")
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
            logger.error(f"Error reading image: {e}")
            return None

    def _process_results(self, ocr_results: Any) -> Union[Dict, None]:
        results = [
            {"text": " ".join(text.split()), "coordinates": box.astype(int).tolist()}
            for text, box in ocr_results
        ]
        if not results:
            logger.warning("No text found.")
            return None
        return {"ocr_result": results}
