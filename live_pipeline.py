import cv2
from grid_detector import order_corners, detect_and_warp_grid, split_into_cells

cap = cv2.VideoCapture(0)

while True:
  ret, frame = cap.read()
  if not ret:
    break

  warped = detect_and_warp_grid(frame)
  if warped is None:
    cv2.imshow("feed", frame)
  else: 
    cv2.imshow("feed", warped)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()


