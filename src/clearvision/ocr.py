import cv2
import keras_ocr


class OCR:
    def __init__(self):
        self.pipeline = keras_ocr.pipeline.Pipeline(scale=2)

    def perform_ocr(self, image_path, confidence_threshold=0.8):
        image = cv2.imread(image_path)
        img = keras_ocr.tools.read(image_path)

        # Get detailed OCR results including bounding boxes and confidence scores
        ocr_results = self.pipeline.recognize([img])[0]

        results = []
        for text, box in ocr_results:
            # Get the full OCR text
            full_text = " ".join(text.split())

            # Draw the bounding box and text on the image for debugging
            box = box.astype(int)
            cv2.polylines(image, [box], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(
                image,
                full_text,
                (box[0][0], box[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

            results.append(
                {
                    "text": full_text,
                    "coordinates": box.tolist(),
                }
            )

        # Show the debug window
        cv2.imshow("Debug Window", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return results
