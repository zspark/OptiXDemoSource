import glob
import os
import sys
import numpy as np
import array
import math

import PIL
from PIL import Image

import numpy as np

rgb_files = sorted(glob.glob(os.path.join(sys.argv[1],'*.png')))

for i in range(len(rgb_files)):
    print rgb_files[i]

    rgb = Image.open(rgb_files[i]).convert('RGB')
    w,h = rgb.size
    img = np.nan_to_num(np.array(rgb,dtype='float32')/255. - 0.5)

    numi_0 = rgb_files[i].rfind(".")
    output = sys.argv[2] + os.path.basename(rgb_files[i][0:numi_0])
    np.save(output, img)
