import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.draw import line_nd
import skimage.morphology as morphology
from PIL import ImageEnhance
from PIL import  Image
import glob
import sys
import os 

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from swc_handler import *
from file_io import *
from path_util import *
from datasets.swc_processing import *


def convert_color_g(img):
    r = np.zeros(img.shape, dtype=np.uint8)
    b = np.zeros(img.shape, dtype=np.uint8)
    dst = cv.merge(([b, img, r]))
    return dst


def convert_color_r(img):
    g = np.zeros(img.shape, dtype=np.uint8)
    b = np.zeros(img.shape, dtype=np.uint8)
    dst = cv.merge(([img, g, b]))
    return dst


def convert_color_w(img):
    img = img[:, :, np.newaxis]
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img


def img_contrast(img, contrast):
    image = Image.fromarray(img)
    enh_con = ImageEnhance.Contrast(image)
    img_con = enh_con.enhance(contrast)
    img_con = np.array(img_con)
    return img_con


def checkGT(swcDir, imgDir, segDir, saveDir, imgShape, somaShape):

    for i, swcFile in enumerate(glob.glob(os.path.join(swcDir, '*.swc'))):
        filePrefix = get_file_prefix(swcFile) 
        swcImg = swc_to_image(parse_swc(swcFile), imgshape=imgShape)
        
        somaSeg = load_image(os.path.join(segDir, f'{filePrefix}.v3draw'))[0]
        img = load_image(os.path.join(imgDir, f'{filePrefix}.v3draw'))[0]
        center = [dim // 2 for dim in imgShape]
        swcImg = np.clip(swcImg, 0, 170)
        swcImg[center[0] - somaShape[0] // 2: center[0] + somaShape[0] // 2,
              center[1] - somaShape[1] // 2: center[1] + somaShape[1] // 2,
              center[2] - somaShape[2] // 2: center[2] + somaShape[2] // 2] += somaSeg * 255
    
        plt.figure(figsize=(10,4), dpi=600)
        plt.subplot(121)
        plt.imshow(cv.addWeighted(convert_color_w(img_contrast(np.max(img, axis=0), contrast=5.0)), 0.5,
               convert_color_r(np.max(swcImg, axis=0)), 0.5, 0))
        plt.subplot(122)
        plt.imshow(cv.addWeighted(convert_color_w(img_contrast(np.max(img, axis=1), contrast=5.0)), 0.5,
               convert_color_r(np.max(swcImg, axis=1)), 0.5, 0))
        plt.show()

        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
            os.chmod(saveDir, 0o777)
        if not os.path.exists(os.path.join(saveDir, f'{filePrefix}.png')):
            plt.savefig(os.path.join(saveDir, f'{filePrefix}.png'))


if __name__ == '__main__':
    swcDir = '/PBshare/SEU-ALLEN/Users/Gaoyu/neuronSegSR/dataset/swc/256/final'
    imgDir = '/PBshare/SEU-ALLEN/Users/Gaoyu/neuronSegSR/dataset/img/256/raw/'
    segDir = '/PBshare/SEU-ALLEN/Users/Gaoyu/neuronSegSR/dataset/img/256/somaSeg/'
    saveDir = '/PBshare/SEU-ALLEN/Users/Gaoyu/neuronSegSR/dataset/img/256/checkGT'
    imgShape = [128, 256, 256]
    somaShape = [64, 128, 128]
    checkGT(swcDir, imgDir, segDir, saveDir, imgShape, somaShape)
