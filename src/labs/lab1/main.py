import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
IMAGE_PATH = BASE_DIR / "assets" / "img.png"

def get_shape(file: Path) -> tuple[int, int, int]:
    img = cv2.imread(file) # upload the img
    h, w, c = img.shape # get the shape height, width, channels

    return h,w,c # return tuple

print("Shape (h, w, c): ", *get_shape(IMAGE_PATH), sep=" ") # print the result as unpacked tuple
