from clearvision.imageproc.toolkit import (
    adjust_contrast_brightness,
    denoise_image,
    deskew,
    thresholding,
)


def clean_text_areas(image):
    image = denoise_image(image)

    image = deskew(image, method="moments")
    image = adjust_contrast_brightness(image, method="clahe")

    image = thresholding(image, method="otsu")

    return image
