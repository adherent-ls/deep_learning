import cv2
import math
import numpy as np

from base.data.base_transform import ImageTransform


def resize_norm_img(img,
                    image_shape,
                    padding=True,
                    interpolation=cv2.INTER_LINEAR):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    if not padding:
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=interpolation)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


class ImageResizeNormal(ImageTransform):
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 eval_mode=False,
                 padding=True):
        super().__init__()
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.eval_mode = eval_mode
        self.padding = padding

    def forward(self, image):
        norm_img, valid_ratio = resize_norm_img(image, self.image_shape,
                                                self.padding)
        return norm_img
