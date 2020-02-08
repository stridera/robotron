"""
Collection of util functions
"""
import cv2

def crop(image, coords):
    """
    Crops out the gamebox from the full image

    Arguments:
        image {ndarray} -- Input image

    Returns:
        ndarray -- The cropped image
    """
    (left, top, right, bottom) = coords
    return image[left:right, top:bottom]

def overlay_text(image, text, location, size=3, weight=8, color=(255, 255, 255)):
    """
    Write text on an image

    Arguments:
        image {ndarray} -- The image
        text {string} -- [description]
        location {list} -- List including x, y locations

    Keyword Arguments:
        size {int} -- Size (default: {3})
        weight {int} -- Font Weight (default: {8})
        color {tuple} -- Text Color(default: {(255, 255, 255)})

    Returns:
        ndarray -- Altered image.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (x, y) = location
    line_height = 40
    for i, line in enumerate(text.split('\n')):
        y = y + i * line_height
        cv2.putText(image, line, (x, y), font, size, color, weight)
    return image
