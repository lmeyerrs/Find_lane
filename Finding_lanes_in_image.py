import matplotlib.pyplot as plt
import numpy as np
import cv2



#opening image
image = cv2.imread("test_images/solidWhiteRight.jpg")
# image = cv2.imread("test_images/solidWhiteCurve.jpg")
# image = cv2.imread("test_images/solidYellowCurve.jpg")
# image = cv2.imread("test_images/solidYellowLeft.jpg")
#image = cv2.imread("test_images/solidYellowCurve2.jpg")

print(image.shape)
cv2.imshow("Original image", image)
img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
shape = img_gray.shape
width = shape[1]
height = shape[0]
cv2.imshow("Gray Scale", img_gray)
status = cv2.imwrite('test_images/img_gray.jpg', img_gray)


# Apply Gaussian Blur
kernel_sizel = 3
blur_image = cv2.GaussianBlur(img_gray, (kernel_sizel, kernel_sizel), 0)
cv2.imshow("Gaussian Blur", blur_image)
status = cv2.imwrite('test_images/blur_image.jpg', blur_image)

# Use Canny Detector
low_threshold = 50
high_treshold = 150
canny_image = cv2.Canny(blur_image, low_threshold, high_treshold)
cv2.imshow("Canny Image", canny_image)
status = cv2.imwrite('test_images/canny_image.jpg', canny_image)

# create mask edges
mask_image = np.zeros_like(canny_image)
ignore_mask_color = 255

# Apply a poligon region
low_left = (150, height)
high_left = (450, 320)
low_right = (490, 320)
high_right = (width, height)
vertices = np.array([[(low_left, high_left, low_right, high_right)]], dtype=np.int32)
cv2.fillPoly(mask_image, vertices, ignore_mask_color)
cv2.imshow("Masked Image", mask_image)
status = cv2.imwrite('test_images/mask_image.jpg', mask_image)

# Apply mask after canny image
masked_image_result = cv2.bitwise_and(mask_image, canny_image)
cv2.imshow("Result Image", masked_image_result)
status = cv2.imwrite('test_images/masked_image_result.jpg', masked_image_result)

# Applying Hough Transform on image result from bitwise and operation
rho = 1     # distance resolution grid
theta = np.pi/180       # angular resolution in radians
hough_threshold = 15    # minimum number of votes
min_line_len = 20       # The minimum quantity of pixel necessary to recognize a line
max_line_gap = 80      # maximum gap between pixels
line_image = np.copy(image)

# Run Hough transform
lines = cv2.HoughLinesP(masked_image_result, rho, theta, hough_threshold, np.array([]), min_line_len, max_line_gap)
#print(lines)

# Adding lines to line_images
for line in lines:
    for x1, y1, x2, y2 in line:
        # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

cv2.imshow("Line Image", line_image)
status = cv2.imwrite('test_images/line_image.jpg', line_image)


cv2.waitKey()
cv2.destroyAllWindows()

