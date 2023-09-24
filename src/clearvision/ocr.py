from typing import Any, Dict, Union

import image_ocr as keras_ocr
import numpy as np

from clearvision import logger


class OCR:
    def __init__(
        self,
        scale_factor: float = 2.0,
        recognizer_weights: str = None,
        alphabet: str = keras_ocr.recognition.DEFAULT_ALPHABET,
    ):
        self.detector = keras_ocr.detection.Detector()

        if recognizer_weights:
            self.recognizer = keras_ocr.recognition.Recognizer(
                weights=None, alphabet=alphabet
            )
            self._load_custom_weights(self.recognizer.model, recognizer_weights)
        else:
            self.recognizer = keras_ocr.recognition.Recognizer()

        self.pipeline = keras_ocr.pipeline.Pipeline(
            detector=self.detector, recognizer=self.recognizer, scale=scale_factor
        )
        logger.info("Pipeline setup complete")

    def _load_custom_weights(self, model, weights_path):
        try:
            model.load_weights(weights_path)
        except ValueError as e:
            logger.warning(f"Failed to load custom weights: {e}")

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
