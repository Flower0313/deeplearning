import os

from pyzbar import pyzbar
from PIL import Image

## 打开图片
# img = Image.open(r"T:\deeplearning\telegram\qrcode.jpg")
#
## 扫描图片中的二维码
# codes = pyzbar.decode(img)
#
## 判断是否为二维码
# if len(codes) > 0 and codes[0].type == 'QRCODE':
#    print('这是一个二维码')
# else:
#    print('这不是一个二维码')

folder_path = r"T:\deeplearning\imgs\\"
for filename in os.listdir(folder_path):
    img = Image.open(folder_path + filename)
    codes = pyzbar.decode(img)
    if len(codes) > 0 and codes[0].type == 'QRCODE':
        print('这是一个二维码')
        os.remove(folder_path + filename)
