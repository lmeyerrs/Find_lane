import cv2
import matplotlib.pyplot as plt
import numpy as np

# Opening video
# cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")
cap = cv2.VideoCapture("test_videos/solidYellowLeft.mp4")
# cap = cv2.VideoCapture("test_videos/challenge.mp4")

# Extracting relevant information from video and use it to configure video output
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# video_output = cv2.VideoWriter("test_videos_output/challenge.mp4", fourcc, 25, (width, height))
video_output = cv2.VideoWriter("test_videos_output/solidYellowLeft.mp4", fourcc, 25, (width, height))
# video_output = cv2.VideoWriter("test_videos_output/solidWhiteRight.mp4", fourcc, 25, (width, height))

# Test if no error to open
if not cap.isOpened():
    print("Error opening file")


while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian Blur
        kernel_sizel = 3
        blur_image = cv2.GaussianBlur(img_gray, (kernel_sizel, kernel_sizel), 0)

        # Use Canny Detector
        low_threshold = 50
        high_threshold = 150
        canny_image = cv2.Canny(blur_image, low_threshold, high_threshold)

        # create mask edges
        mask_image = np.zeros_like(canny_image)
        ignore_mask_color = 255

        # Apply a polygon region
        # low_left = (150, height)
        low_left = (190, height)
        high_left = (450, 320)
        low_right = (490, 320)
        high_right = (width, height)
        vertices = np.array([[(low_left, high_left, low_right, high_right)]], dtype=np.int32)
        cv2.fillPoly(mask_image, vertices, ignore_mask_color)

        # Apply mask after canny image
        masked_image_result = cv2.bitwise_and(mask_image, canny_image)

        # Applying Hough Transform on image result from bitwise and operation
        rho = 1  # distance resolution grid
        theta = np.pi / 180  # angular resolution in radians
        hough_threshold = 15  # minimum number of votes
        min_line_len = 15  # The minimum quantity of pixel necessary to recognize a line
        max_line_gap = 30  # maximum gap between pixels
        line_image = np.copy(frame)

        # Run Hough transform
        lines = cv2.HoughLinesP(masked_image_result, rho, theta, hough_threshold, np.array([]), min_line_len,
                                max_line_gap)

        # Adding lines to line_images
        for line in lines:
            for x1, y1, x2, y2 in line:
                # cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        cv2.imshow("Line Image", line_image)
        video_output.write(line_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cv2.waitKey()
cap.release()
cv2.destroyAllWindows()