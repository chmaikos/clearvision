import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from clearvision.ocr import OCR

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description='Perform OCR on a single image or multiple images in a directory.')
    parser.add_argument('input', type=str, help='Path to the image file or directory.')
    args = parser.parse_args()

    ocr = OCR()

    if os.path.isdir(args.input):
        image_paths = [os.path.join(args.input, f) for f in os.listdir(args.input)
                       if os.path.isfile(os.path.join(args.input, f))]
        output_data = {}

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(ocr.perform_ocr, image_paths))

        for i, res in enumerate(results):
            if res is not None:
                output_data[image_paths[i]] = res
            else:
                logging.error(f"Failed to process image: {image_paths[i]}")

        with open('output.json', 'w') as json_file:
            json.dump(output_data, json_file, indent=4)
    else:
        results = ocr.perform_ocr(args.input)
        if results is not None:
            with open('output.json', 'w') as json_file:
                json.dump({args.input: results}, json_file, indent=4)
        else:
            logging.error(f"Failed to process image: {args.input}")


if __name__ == '__main__':
    main()