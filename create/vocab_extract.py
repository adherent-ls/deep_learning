import cv2
import io
import numpy as np
import six
from PIL import Image

# 读取图像
image_path = '/home/data/data_old/SynthText/2/ant+hill_4_0.jpg'
image = cv2.imread(image_path)

# OpenCV图像转换为字节
_, buffer = cv2.imencode('.jpg', image)
image_bytes = buffer.tobytes()

# 将字节写入io.BytesIO对象
image_io = io.BytesIO(image_bytes)
buf = six.BytesIO()
buf.write(image_bytes)
buf.seek(0)

# 你可以选择使用PIL库打开图像并显示
pil_image = Image.open(image_io)
# pil_image.show()
print(pil_image)