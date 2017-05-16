'''
Created on 16 May 2017

@author: FIRAT
'''
from PIL import Image
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    img = img.convert('1')
    data = np.asarray( img, dtype="int32" ).reshape(-1)
#     print(data)
    return data

#testing
imageY1 = load_image('dl_data/1.png')
imageY2 = load_image('dl_data/1a.png')
imageY3 = load_image('dl_data/1b.png')
imageY4 = load_image('dl_data/1c.png')
imageY5 = load_image('dl_data/1d.png')
imageY6 = load_image('dl_data/1e.png')
imageY7 = load_image('dl_data/1f.png')
imageY8 = load_image('dl_data/1g.png')
imageY9 = load_image('dl_data/1h.png')
imageY10 = load_image('dl_data/1i.png')

imageA1 = load_image('dl_data/2a.png')
imageA2 = load_image('dl_data/2b.png')
imageA3 = load_image('dl_data/2c.png')
imageA4 = load_image('dl_data/2d.png')
imageA5 = load_image('dl_data/2e.png')
imageA6 = load_image('dl_data/2f.png')
imageA7 = load_image('dl_data/2g.png')
imageA8 = load_image('dl_data/2h.png')
imageA9 = load_image('dl_data/2i.png')
imageA10 = load_image('dl_data/3.png')

imageS1 = load_image('dl_data/3a.png')
imageS2 = load_image('dl_data/3b.png')
imageS3 = load_image('dl_data/3c.png')
imageS4 = load_image('dl_data/3d.png')
imageS5 = load_image('dl_data/3e.png')
imageS6 = load_image('dl_data/3f.png')
imageS7 = load_image('dl_data/3g.png')
imageS8 = load_image('dl_data/3h.png')
imageS9 = load_image('dl_data/3i.png')

imageI1 = load_image('dl_data/4.png')
imageI2 = load_image('dl_data/4a.png')
imageI3 = load_image('dl_data/4b.png')
imageI4 = load_image('dl_data/4c.png')
imageI5 = load_image('dl_data/4d.png')
imageI6 = load_image('dl_data/4e.png')
imageI7 = load_image('dl_data/4f.png')
imageI8 = load_image('dl_data/4g.png')
imageI9 = load_image('dl_data/4h.png')
imageI10 = load_image('dl_data/4i.png')