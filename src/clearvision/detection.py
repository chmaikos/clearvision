from typing import List, Tuple

from dbnet import DBNet


class TextDetection:
    def __init__(self):
        self.model = DBNet()

    def detect_text_areas(self, image_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Detect areas with text in the given image.

        Args:
        - image_path (str): The path to the image file.

        Returns:
        - list of tuple: A list of detected text areas with
        their coordinates represented as (x, y, width, height).
        """
        # Load the image and perform text detection
        boxes = self.model.detect(image_path)

        # Convert the box coordinates to the required format
        bounding_boxes = [(int(box[0]), int(box[1]),
                           int(box[2] - box[0]),
                           int(box[3] - box[1])) for box in boxes]

        # Filter out small bounding boxes
        min_area = 100  # Minimum area to keep a bounding box
        bounding_boxes = [box for box in bounding_boxes if box[2] * box[3] > min_area]

        return bounding_boxes
