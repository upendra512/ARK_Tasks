import cv2 
import numpy as np
import math
# Using Sobel Operator
image=cv2.imread('table.png',cv2.IMREAD_GRAYSCALE)
#performing edge detection 
gradients_sobelx = cv2.Sobel(image,-1,1,0)
gradients_sobely= cv2.Sobel(image,-1,0,1)
gradients_sobelxy= cv2.addWeighted(gradients_sobelx,0.5,gradients_sobely ,0.5,0)

# cv2.imshow('Sobel x',gradients_sobelx)
# cv2.imshow('Sobel y',gradients_sobely)
resize=cv2.resize(gradients_sobelxy,(500,500))
cv2.imshow('Sobel xy',resize)

# Using Laplacian Operator
gradients_laplacian= cv2.Laplacian(image,-1)
cv2.imshow('Laplacian',gradients_laplacian)
# Using Canny Edge Operator
canny_output = cv2.Canny(image, 80, 100)
resized = cv2.resize(canny_output,(500,500))
cv2.imshow('Canny',resized)
# Hough Line Transform
image1 = cv2.imread('table.png', cv2.IMREAD_GRAYSCALE)
# Binarize image
_, binary_image = cv2.threshold(image1,150,255,cv2.THRESH_BINARY)
cdst = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

# Standard Hough Line Transform
lines = cv2.HoughLines(canny_output, 1, np.pi / 180, 150, None, 0, 0)
# print(lines)
# Draw the lines
if lines is not None:
    for i in range(len(lines)):
        rho, theta = lines[i ][0]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

# Show results
# cv2.imshow("Detected Lines - Standard Hough Line Transform", cdst)
# resized_image = cv2.resize(cdst, (500,500))
cv2.imshow('Image',resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
