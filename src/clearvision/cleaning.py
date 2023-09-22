from clearvision.imageproc.toolkit import (
    adjust_contrast_brightness,
    denoise_image,
    deskew,
    thresholding,
)


def prep(image, dmethod="moments", cmethod="clahe", tmethod="otsu"):
    image = denoise_image(image)

    image = deskew(image, dmethod)
    image = adjust_contrast_brightness(image, cmethod)

    image = thresholding(image, tmethod)

    return image
