Here are some examples of how to use the different descriptors in OpenCV:

1. SIFT:
```
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('SIFT', img)
cv2.waitKey(0)
```

2. SURF:
```
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()
kp, des = surf.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('SURF', img)
cv2.waitKey(0)
```

3. ORB:
```
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('ORB', img)
cv2.waitKey(0)
```

4. BRISK:
```
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

brisk = cv2.BRISK_create()
kp, des = brisk.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('BRISK', img)
cv2.waitKey(0)
```

5. FREAK:
```
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

freak = cv2.xfeatures2d.FREAK_create()
kp, des = freak.compute(gray, kp)

img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('FREAK', img)
cv2.waitKey(0)
```

6. AKAZE:
```
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

akaze = cv2.AKAZE_create()
kp, des = akaze.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('AKAZE', img)
cv2.waitKey(0)
```

7. KAZE:
```
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kaze = cv2.KAZE_create()
kp, des = kaze.detectAndCompute(gray, None)

img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('KAZE', img)
cv2.waitKey(0)
```

8. LATCH:
```
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

latch = cv2.xfeatures2d.LATCH_create()
kp, des = latch.compute(gray, kp)

img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('LATCH', img)
cv2.waitKey(0)
```
9. HoG:
```
import cv2

img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hog = cv2.HOGDescriptor()
kp = hog.compute(gray)

img = cv2.drawKeypoints(gray, kp, img)
cv2.imshow('HoG', img)
cv2.waitKey(0)
```

Note that in these examples, 'image.jpg' refers to the file name of the image you want to use. The descriptors will extract keypoints and descriptors from the image and draw them on the image for visualization purposes.