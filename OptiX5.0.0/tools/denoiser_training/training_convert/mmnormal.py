import glob
import os
import sys
import time
import pickle
import numpy as np
import array
import math

import PIL
from PIL import Image

import OpenEXR
import Imath

import numpy as np

rgb_files = sorted(glob.glob(os.path.join(sys.argv[1],'*.exr')))

def remap(x):
    return x * 0.5

for i in range(len(rgb_files)):
    print rgb_files[i]

    rgbfile = OpenEXR.InputFile(rgb_files[i])
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    dw = rgbfile.header()['dataWindow']
    w = dw.max.x - dw.min.x + 1
    h = dw.max.y - dw.min.y + 1

    rgb = rgbfile.channels("RGB", pt)
    img = np.zeros((h, w, 3), np.float32)
    img[...,0] = np.nan_to_num(np.array(np.fromstring(rgb[0], dtype = np.float32).reshape(h, w)))
    img[...,1] = np.nan_to_num(np.array(np.fromstring(rgb[1], dtype = np.float32).reshape(h, w)))
    img[...,2] = np.nan_to_num(np.array(np.fromstring(rgb[2], dtype = np.float32).reshape(h, w)))

    img = remap(img)

    numi_0 = rgb_files[i].rfind(".")
    output = sys.argv[2] + os.path.basename(rgb_files[i][0:numi_0])
    np.save(output, img)
