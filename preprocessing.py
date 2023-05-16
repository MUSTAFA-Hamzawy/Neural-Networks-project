import cv2
import numpy as np

def shadow_removalDialte(img):
    img2 = cv2.resize(img, (256, 256))
    
    img_hls = cv2.cvtColor(img2, cv2.COLOR_BGR2HLS)

    # define lower and upper bounds for blue color in HLS format
    lower_blue = np.array([0, 0, 60])
    upper_blue = np.array([20, 255, 255])
    # create a mask for blue color in HLS format
    mask = cv2.inRange(img_hls, lower_blue, upper_blue)
    # daialation
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(mask,kernel,iterations = 8)
    # apply the mask to the original image
    result = cv2.bitwise_and(img2, img2, mask=mask)
    return result

def preprocessing(img,org_img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance contrast and remove lighting effects
    equalized = cv2.equalizeHist(gray)

    # Find the contours of the binary image
    contours, hierarchy = cv2.findContours(equalized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour on the original image
    cv2.drawContours(org_img, [max_contour], 0, (0, 255, 0), 2)

    # Show the image with the largest contour drawn
    # Create a bounding box around the hand
    x, y, w, h = cv2.boundingRect(max_contour)

    # Crop the image to the bounding box around the hand
    hand = equalized[y:y+h, x:x+w]
    
    # if width of the image is less than height then rotate the image by 90 degree
    # if w < h:
    #     hand = cv2.rotate(hand, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # apply median filter to remove noise
    Noise_Reduction = cv2.medianBlur(hand, 5)
    # resize the image to 256x256
    img3 = cv2.resize(Noise_Reduction, (256, 256))

    return img3