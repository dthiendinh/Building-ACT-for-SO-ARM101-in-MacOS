import cv2


WRIST_CAMERA = 0
# FRONT_CAMERA = find_camera_index("0x1130004a545233")
CAMERA_DEVICE = WRIST_CAMERA

cam = cv2.VideoCapture(CAMERA_DEVICE)

if not cam.isOpened():
    print(f"Failed to open camera: {CAMERA_DEVICE}")
    exit()

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cam.get(cv2.CAP_PROP_FPS)

print("====================================")
print("SO-101 CAMERA TEST")
print("====================================")
print(f"Device: {CAMERA_DEVICE}")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")
print("Press 'q' to quit")
print("====================================")

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("SO-101 Workspace Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()