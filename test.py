import tensorflow as tf
import cv2
import numpy as np
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))


print(cv2.__version__)
image = cv2.imread("Hist.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Over the Clouds", image)
cv2.imshow("Over the Clouds - gray", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Read image
image = cv2.imread("man.jpg")
img = np.float32(image) / 255.0

# Calculate gradient
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# Python Calculate gradient magnitude and direction ( in degrees )
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

cv2.imshow("Gx", gx)
cv2.imshow("Gy", gy)
cv2.imshow("mag", mag)
cv2.imshow("original", image)
cv2.waitKey(0)

