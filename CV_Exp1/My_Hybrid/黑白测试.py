import numpy as np
from PIL import Image
img = Image.open('./resources/flw.jpg')
img = np.array(img)
img1 = img.astype ( np.float32 ) / 255.0
print(min(img1.shape))
print(img1.shape)
if min(img1.shape)==2:
    img1 = np.expand_dims(img1,axis = 3)
    print("ok")
print(img1.shape)
print(img1.ndim)
