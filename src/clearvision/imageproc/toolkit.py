import logging

import cv2
import numpy as np
from skimage.filters import threshold_local, threshold_niblack, threshold_sauvola


def thresholding(image, method="adaptive", block_size=13, C=1):
    """Applies various thresholding techniques on the input grayscale image.

    Args:
    gray_image (ndarray): Input grayscale image.
    method (str): Thresholding method to apply ('adaptive_mean', 'adaptive_gaussian',
    'otsu', 'binary', 'sauvola', 'niblack', 'local').
    block_size (int): Size of a pixel neighborhood that is used to calculate a
    threshold value for the pixel.
    C (int): Constant subtracted from the mean or weighted mean.

    Returns:
    ndarray: Thresholded binary image.
    """
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if method == "adaptive_mean":
            thresh = cv2.adaptiveThreshold(
                gray_image,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                block_size,
                C,
            )
        elif method == "adaptive_gaussian":
            thresh = cv2.adaptiveThreshold(
                gray_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                C,
            )
        elif method == "otsu":
            _, thresh = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif method == "sauvola":
            thresh_sauvola = threshold_sauvola(gray_image, window_size=block_size)
            thresh = gray_image > thresh_sauvola
            thresh = thresh.astype("uint8") * 255
        elif method == "niblack":
            thresh_niblack = threshold_niblack(gray_image, window_size=block_size)
            thresh = gray_image > thresh_niblack
            thresh = thresh.astype("uint8") * 255
        elif method == "local":
            thresh_local = threshold_local(gray_image, block_size, offset=C)
            thresh = gray_image > thresh_local
            thresh = thresh.astype("uint8") * 255
        else:
            _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        # Check if the image is mostly black and invert it if necessary
        mean_color = cv2.mean(gray_image)[0]
        if mean_color < 60:
            thresh = cv2.bitwise_not(thresh)

        return thresh
    except Exception as e:
        logging.error(f"Error in thresholding: {e}")
        return image


def _getSkewAngle(cvImage) -> float:
    """Calculates the skew angle of an image.

    Args:
    cvImage (ndarray): Input image.

    Returns:
    float: Skew angle.
    """
    try:
        # Prep image, copy, convert to gray scale, blur, and threshold
        newImage = cvImage.copy()
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Apply dilate to merge text into lines/paragraphs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=5)

        # Find all contours
        contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Ensure at least one contour was found
        if len(contours) == 0:
            logging.warning("No contours found in image.")
            return 0.0

        # Find largest contour and surround in min area box
        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)

        # Determine the angle
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle
    except Exception as e:
        logging.error(f"Failed to calculate skew angle: {e}")
        return 0.0


def _rotateImage(cvImage, angle: float):
    """Rotates the image around its center.

    Args:
    cvImage (ndarray): Input image.
    angle (float): Angle to rotate the image.

    Returns:
    ndarray: Rotated image.
    """
    try:
        newImage = cvImage.copy()
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(
            newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return newImage
    except Exception as e:
        logging.error(f"Failed to rotate image: {e}")
        return cvImage


def deskew(cvImage):
    """Deskews the input image.

    Args:
    cvImage (ndarray): Input image.

    Returns:
    ndarray: Deskewed image.
    """
    try:
        angle = _getSkewAngle(cvImage)
        return _rotateImage(cvImage, -1.0 * angle)
    except Exception as e:
        logging.error(f"Failed to deskew image: {e}")
        return cvImage


def get_quadrilateral_corners(image):
    """Finds the corners of the largest quadrilateral in the image.

    Args:
    image (ndarray): Input image.

    Returns:
    list of tuple: List of four points representing the corners of the
    largest quadrilateral.
    """
    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        thresh = thresholding(gray, method="otsu")

        # Find the contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Find the quadrilateral corners
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        return [tuple(point) for point in box]
    except Exception as e:
        logging.error(f"Error in finding quadrilateral corners: {e}")
        return None


def correct_distortion(image):
    """Corrects the distortion of the input image using homography techniques.

    Args:
    image (ndarray): Input image.

    Returns:
    ndarray: Image with corrected distortion.
    """
    try:
        # Get the source points (corners of the largest quadrilateral)
        src_points = get_quadrilateral_corners(image)

        # Define the destination points as the corners of a rectangle
        dst_points = [
            (0, 0),
            (image.shape[1] - 1, 0),
            (image.shape[1] - 1, image.shape[0] - 1),
            (0, image.shape[0] - 1),
        ]

        # Convert points to numpy arrays
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Find the homography matrix
        matrix, _ = cv2.findHomography(src_points, dst_points)

        # Perform the perspective warp
        corrected_image = cv2.warpPerspective(
            image, matrix, (image.shape[1], image.shape[0])
        )

        return corrected_image
    except Exception as e:
        logging.error(f"Error in distortion correction: {e}")
        return image


def adjust_contrast_brightness(
    image, method="simple", alpha=1.0, beta=0, clip_limit=2.0, tile_grid_size=(8, 8)
):
    """Adjusts the contrast and brightness of the input image using various techniques.

    Args:
    image (ndarray): Input image.
    method (str): Method to use for adjustment ('simple', 'hist_eq', 'clahe').
    alpha (float): Contrast control (1.0-3.0) for 'simple' method.
    beta (int): Brightness control (0-100) for 'simple' method.
    clip_limit (float): Threshold for contrast limiting for 'clahe' method.
    tile_grid_size (tuple of int): Size of grid for block-wise
    histogram equalization for 'clahe' method.

    Returns:
    ndarray: Image with adjusted contrast and brightness.
    """
    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if method == "simple":
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        elif method == "hist_eq":
            adjusted = cv2.equalizeHist(gray)
            adjusted = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)
        elif method == "clahe":
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            adjusted = clahe.apply(gray)
            adjusted = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)
        else:
            logging.warning(f"Unknown adjustment method: {method}.")
            return image

        return adjusted
    except Exception as e:
        logging.error(f"Error in contrast and brightness adjustment: {e}")
        return image


def denoise_image(
    image,
    h=10,
    h_for_color_components=10,
    template_window_size=7,
    search_window_size=21,
):
    """
    Denoises the input image using Non-Local Means Denoising algorithm.

    Args:
    image (ndarray): Input image.
    h (float): Parameter regulating filter strength. Big h value removes noise
    effectively but removes image details. Smaller h value preserves details but also
    preserves some noise.
    h_for_color_components (float): Same as h but for color images only.
    template_window_size (int): Size (in odd pixels) of the window used to compute the
    weighted average for a given pixel. Should be odd. (recommended: 7)
    search_window_size (int): Size (in pixels) of the window used to search for pixels
    with similar color. (recommended: 21)

    Returns:
    ndarray: Denoised image.
    """
    try:
        # Apply fastNlMeansDenoisingColored
        dst = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h,
            h_for_color_components,
            template_window_size,
            search_window_size,
        )
        return dst
    except Exception as e:
        logging.error(f"Error in denoising image: {e}")
        return image
