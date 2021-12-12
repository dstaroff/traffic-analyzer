import cv2
import numpy as np


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    width, height = image.shape[1], image.shape[0]
    resized_image = cv2.resize(image, (mask.shape[1], mask.shape[0]))

    for c in range(3):
        resized_image[:, :, c] = np.where(
                mask == 1,
                resized_image[:, :, c] *
                (1 - alpha) + alpha * color[c],
                resized_image[:, :, c]
                )

    return cv2.resize(resized_image, (width, height))
