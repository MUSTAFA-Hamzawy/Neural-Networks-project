# Importing Required Modules
from functions import *
from rembg import remove
from PIL import Image, ImageOps
import cv2
from skimage import data, io


input_path = "./test.png"
image = cv2.imread(input_path)
result = preprocess(image)
show_images([result], "After Edge Detection");
