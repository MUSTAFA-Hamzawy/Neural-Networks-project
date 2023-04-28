# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
from rembg import remove

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def preprocess(image):
    # Removing the background from the given Image
    removed_bg = remove(image)
    # Convert the image to grayscale
    grey = cv2.cvtColor(removed_bg, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding using Otsu's algorithm
    thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Apply Canny edge detection to the thresholded image
    edged = cv2.Canny(thresh, 75, 200)
    # Define a kernel for morphological closing
    kernal = np.ones((3, 3), np.uint8)
    # Apply morphological closing to fill gaps and small holes in the edges
    thresh_closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernal, iterations=3)
    
    return thresh_closed