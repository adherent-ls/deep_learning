import cv2
import numpy as np

from base.data.base_transformer import BaseTransformer


class Resize(BaseTransformer):
    def __init__(self, max_size=(640, 640), limit_max_w=1000, is_pad=True, pad=0, **kwargs):
        super(Resize, self).__init__(**kwargs)
        self.max_width = max_size[0]
        self.max_height = max_size[1]
        self.is_pad = is_pad
        self.pad = pad
        self.limit_max_w = limit_max_w

    def forward(self, image, label):

        max_w = self.max_width
        max_h = self.max_height

        new_images = []
        new_labels = []
        h, w, c = image.shape

        radio = max(w / self.max_width, h / self.max_height)
        new_w, new_h = int(w / radio), int(h / radio)

        max_w = max(max_w, new_w + self.pad)
        max_h = max(max_h, new_h)

        if new_w > self.limit_max_w:
            return None

        image = cv2.resize(image, (new_w, new_h))
        if self.is_pad:
            new_h, new_w, c = image.shape
            pad_w, pad_h = (max_w - new_w) // 2, (max_h - new_h) // 2
            new_image = np.zeros((max_h, max_w, c))
            new_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = image
            return [pad_h, pad_w], new_image, label
        else:
            return new_image, label


class ResizeXW(BaseTransformer):
    def __init__(self, max_size=(640, 640), is_pad=True, pad=0, w_p=2.0, **kwargs):
        super(ResizeXW, self).__init__(**kwargs)
        self.w_p = w_p
        self.max_width = max_size[0]
        self.max_height = max_size[1]
        self.is_pad = is_pad
        self.pad = pad

    def forward(self, images):

        max_w = self.max_width
        max_h = self.max_height

        new_images = []
        for i, image in enumerate(images):
            h, w, c = image.shape

            radio = max(w / self.max_width, h / self.max_height)
            new_w, new_h = int(w / radio), int(h / radio)

            max_w = max(max_w, new_w * self.w_p + self.pad)
            max_h = max(max_h, new_h)

            new_images.append(cv2.resize(image, (new_w * self.w_p, new_h)))
        if self.is_pad:
            image_pad = []
            for i, image in enumerate(new_images):
                new_h, new_w, c = image.shape
                pad_w, pad_h = (max_w - new_w) // 2, (max_h - new_h) // 2
                new_image = np.zeros((max_h, max_w, c))
                new_image[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = image
                image_pad.append([pad_h, pad_w])

                new_images[i] = new_image

            return image_pad, np.array(new_images)
        else:
            return new_images
