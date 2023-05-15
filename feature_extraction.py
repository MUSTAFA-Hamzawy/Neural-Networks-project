import skimage.feature as ft

def extract_features(image):

    return ft.hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),visualize=False)
    