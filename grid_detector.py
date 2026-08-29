import cv2
import numpy as np

def order_corners(points):
  points = points.reshape(4,2)
  ordered = np.zeros((4,2), dtype="float32")
  
  sum_pts = points.sum(axis=1)
  ordered[0] = points[np.argmin(sum_pts)]
  ordered[2] = points[np.argmax(sum_pts)]

  diff_pts = np.diff(points, axis=1)
  ordered[1] = points[np.argmin(diff_pts)]
  ordered[3] = points[np.argmax(diff_pts)]

  return ordered

def detect_and_warp_grid(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
      blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    grid_contour = None 
    max_area = 0
    height, width, channels = image.shape
    total_area = height * width

    for c in contours:
      area = cv2.contourArea(c)
      # if area < 1000:
      #   continue
      if area < total_area * 0.1:
         continue

      x, y, w, h = cv2.boundingRect(c)
      aspect_ratio = w/float(h)

      if not (0.9 <= aspect_ratio <= 1.1):
        continue 

      rect = cv2.minAreaRect(c)
      box = cv2.boxPoints(rect)
      box = np.intp(box)

      # print(f"Candidate — area: {area:.0f}, aspect ratio: {aspect_ratio:.2f}")

      if area > max_area:
          grid_contour = box
          max_area = area

    if grid_contour is None:
       return None

    ordered = order_corners(grid_contour)

    side = 450
    destination = np.array([
      [0,0],
      [ side - 1, 0],
      [ side - 1, side -1],
      [ 0, side -1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered, destination)
    warped = cv2.warpPerspective(image, matrix, (side, side))

    image_with_corners = image.copy()
    cv2.drawContours(image_with_corners, [grid_contour], -1, (0, 255, 0), 3)

    # print("Number of corners found:", len(grid_contour))

    return warped

def split_into_cells(warped, side=450):
    cell_size = side // 9
    cells = []
    margin = 5
    for row in range(9):
      for col in range(9):
        y1 = row * cell_size + margin
        y2 = row * cell_size + cell_size - margin
        x1 = col * cell_size + margin
        x2 = col * cell_size + cell_size - margin

        cell = warped[y1:y2, x1:x2]
        cells.append(cell)

    print("Total cells extracted: ", len(cells))
    return cells


if __name__ == "__main__":
    image = cv2.imread("test2.jpg")
    warped = detect_and_warp_grid(image)
    cells = split_into_cells(warped)
    cv2.imshow("Warped Grid", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
