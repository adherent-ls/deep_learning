from PIL import Image, ImageOps

from base.data.base_transform import ImageTransform


class ImageAdoptResize(ImageTransform):
    """
    in proportion scale by max_size, 限制在max_size范围内
    >>> resize = ImageAdoptResize(max_size=(32, 256))
    """

    def __init__(self, max_size=(32, 256)):
        super().__init__()
        self.max_height, self.max_width = max_size

    def forward(self, image: Image):
        max_w = self.max_width
        max_h = self.max_height

        w, h = image.size

        radio = max(w / max_w, h / max_h)
        new_w, new_h = int(w / radio), int(h / radio)

        resized_image = image.resize((new_w, new_h), Image.BICUBIC)

        # px, py = (max_w - new_w), (max_h - new_h)
        # if px < 0 or py < 0:
        #     return resized_image
        #
        # if px % 2 == 0:
        #     px = (px // 2, px // 2)
        # else:
        #     px = (px // 2, px // 2 + 1)
        # if py % 2 == 0:
        #     py = (py // 2, py // 2)
        # else:
        #     py = (py // 2, py // 2 + 1)
        # border = (px[0], py[0], px[1], py[1])
        # padded_image = ImageOps.expand(resized_image,
        #                                border=border,
        #                                fill=0)
        return resized_image
