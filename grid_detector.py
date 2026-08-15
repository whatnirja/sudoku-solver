import cv2

image = cv2.imread("test3.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.adaptiveThreshold(
  blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

perimeter = cv2.arcLength(largest_contour, True)
approx_corners = cv2.approxPolyDP(largest_contour, 0.02 * perimeter, True)

image_with_corners = image.copy()
cv2.drawContours(image_with_corners, [approx_corners], -1, (0, 255, 0), 3)

cv2.imshow("Sudoku image", image_with_corners)
print("Number of corners found:", len(approx_corners))
cv2.waitKey(0)
cv2.destroyAllWindows()