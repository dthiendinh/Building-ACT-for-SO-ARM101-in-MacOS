"""Preview a camera without connecting either robot arm."""
import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    cam = cv2.VideoCapture(args.index)
    try:
        if not cam.isOpened():
            raise RuntimeError(f"Failed to open camera {args.index}")
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cam.set(cv2.CAP_PROP_FPS, args.fps)
        print(f"Camera {args.index}: {cam.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} x "
              f"{cam.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}, {cam.get(cv2.CAP_PROP_FPS):.1f} FPS")
        print("Press q to quit")
        while True:
            ret, frame = cam.read()
            if not ret:
                raise RuntimeError("Failed to read camera frame")
            cv2.imshow("SO-101 Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
