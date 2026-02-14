import argparse
import os
import cv2
from src.plate_detector import PlateDetector
from src.utils import ensure_dirs
from src.config import TESSERACT_CMD


def parse_args():
    p = argparse.ArgumentParser(description='Detect and OCR car number plate')
    p.add_argument('--image', required=True, help='Path to input image')
    p.add_argument('--save', default='outputs/result.jpg', help='Path to save plate crop')
    p.add_argument('--show', action='store_true', help='Show plate image')
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs(os.path.dirname(args.save) or '.')

    detector = PlateDetector(tesseract_cmd=TESSERACT_CMD)

    result = detector.detect(args.image)

    if result['plate_image'] is not None:
        cv2.imwrite(args.save, result['plate_image'])
        print(f"Plate text: {result['text']}")

        if args.show:
            cv2.imshow("Plate", result['plate_image'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No plate detected.")


if __name__ == '__main__':
    main()
    #.venv\Scripts\activate
# python -m src.main --image images/car1.jpg --save outputs/result.jpg --show