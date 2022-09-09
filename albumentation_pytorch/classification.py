# # advantage:
# It is faster than torchvision on very banchmark
# It has support to more number of tasks like segmentation,classification and detections which
# is harder to do in torchvision.

import cv2
import albumentations as A
import numpy as np
from PIL import Image
from utils import plot_examples

image = Image.open("data/2.jpg")

transform = A.Compose(
    [
        A.Resize(width=1920, height=1080),
        A.RandomCrop(width=1280, height=720),
        A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=0.5),
                A.ColorJitter(p=0.5)
            ], p=0.5
        ),

    ]
)

image_list = [image]
image = np.array(image)
for i in range(20):
    augmentations = transform(image=image)
    augmented_image = augmentations['image']
    image_list.append(augmented_image)

plot_examples(image_list)
