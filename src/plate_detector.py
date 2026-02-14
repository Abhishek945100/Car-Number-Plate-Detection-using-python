import cv2
import numpy as np
import pytesseract
from .utils import resize_keep_aspect


class PlateDetector:
    def __init__(self, tesseract_cmd=None, debug=False):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.debug = debug

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 50, 200)

        # Strong dilation to merge plate edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        return gray, edged, dilated

    def find_plate_contour(self, dilated):
        contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)

            # allow 4–6 corners (tilted plates)
            if 4 <= len(approx) <= 6:
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w / float(h)

                # filter out non-plate shapes using aspect ratio
                if 2 < aspect < 6:
                    return approx

        return None

    def extract_plate(self, image, plate_cnt):
        x, y, w, h = cv2.boundingRect(plate_cnt)
        return image[y:y+h, x:x+w]

    # ✅ NEW HIGH-ACCURACY OCR SYSTEM
    def ocr_plate(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return ""

        # Convert to gray
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # Resize larger for stronger OCR
        gray = resize_keep_aspect(gray, width=600)

        # Reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive thresholding (much better than OTSU for unclear plates)
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 31, 5
        )

        # Slight dilation to thicken letters
        kernel = np.ones((2, 2), np.uint8)
        th = cv2.dilate(th, kernel, iterations=1)

        # Tesseract settings: better for license plates
        config = "--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        text = pytesseract.image_to_string(th, config=config)

        # Clean text
        return ''.join(filter(str.isalnum, text)).upper()

    def detect(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = resize_keep_aspect(image, width=900)
        gray, edged, dilated = self.preprocess(image)

        plate_cnt = self.find_plate_contour(dilated)

        result = {"plate_image": None, "text": ""}

        if plate_cnt is not None:
            plate_img = self.extract_plate(image, plate_cnt)
            result["plate_image"] = plate_img
            result["text"] = self.ocr_plate(plate_img)

        return result
