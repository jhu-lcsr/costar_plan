"""
Copyright (c) 2018, Johns Hopkins University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Johns Hopkins University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL JOHNS HOPKINS UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import io
from PIL import Image
import skimage.transform as transform
try:
    # don't require tensorflow for reading
    #import tensorflow as tf
    tf = None
except ImportError:
    tf = None


def GetJpeg(img):
    '''
    Save a numpy array as a Jpeg, then get it out as a binary blob
    '''
    im = Image.fromarray(np.uint8(img))
    output = io.BytesIO()
    im.save(output, format="JPEG", quality=80)
    return output.getvalue()


def GetPng(img):
    '''
    Save a numpy array as a Jpeg, then get it out as a binary blob
    '''
    im = Image.fromarray(img)
    output = io.BytesIO()
    # enabling optimized file size
    # increases saving time to ~0.4 seconds per image.
    #im.save(output, format="PNG", optimize=True)
    im.save(output, format="PNG")
    return output.getvalue()


def JpegToNumpy(jpeg):
    if tf is not None:
        # make sure to call tf.enable_eager_execution() at the start of your program
        image = tf.image.decode_jpeg(jpeg)
    else:
        stream = io.BytesIO(jpeg)
        image = Image.open(stream)
    return np.asarray(image, dtype=np.uint8)

def _resize_img(img, size):
    return transform.resize(img, (size, size), mode='constant')

def resize_imgs(imgs, max_size):
    ''' Resize an ndarray of images '''
    shp = imgs.shape
    if len(shp) == 4 and shp[-1] == 3 and \
        (shp[1] > max_size or shp[2] > max_size):
            imgs = [_resize_img(i, max_size) for i in imgs]
            imgs = np.array(imgs)
    return imgs

def ConvertImageListToNumpy(data, format='numpy', data_format='NHWC', dtype=np.uint8):
    """ Convert a list of binary jpeg or png files to numpy format.

    # Arguments

    data: a list of binary jpeg images to convert
    format: default 'numpy' returns a 4d numpy array,
        'list' returns a list of 3d numpy arrays
    """
    length = len(data)
    images = []
    for raw in data:
        img = JpegToNumpy(raw)
        if data_format == 'NCHW':
            img = np.transpose(img, [2, 0, 1])
        images.append(img)
    if format == 'numpy':
        images = np.array(images, dtype=dtype)
    return images
