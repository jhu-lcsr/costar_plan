import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image

def Show(img):
    #ax = plt.axes([0,0,1,1])
    plt.axis('off')
    plt.tight_layout()
    img = plt.imshow(img, interpolation="nearest")
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

def Title(msg):
    plt.title(msg, fontsize=16)

def GetJpeg(img, tmp_filename=".tmp.jpg"):
    '''
    Save a numpy array as a Jpeg, then get it out as a binary blob
    '''
    im = Image.fromarray(np.uint8(img))
    output = io.BytesIO()
    im.save(output, format="JPEG")
    return output.getvalue()

def JpegToNumpy(jpeg):
    stream = io.BytesIO(jpeg)
    im = Image.open(stream)
    return np.asarray(im, dtype=np.uint8)
