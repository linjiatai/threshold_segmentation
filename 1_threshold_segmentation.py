from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import cv2
from skimage import morphology
import os

palette = [0]*6
palette[0:3] = [255,255,255]
palette[3:6] = [255,0,0]
palette[6:9] = [0,0,0]

# 使用前要修改这两个地址
IHC_dir = '/home/linjiatai/14TB/天津/天津标注code/ROIs/'
HE_dir = '/home/linjiatai/14TB/天津/天津标注code/ROIs_HE/'
save_dir = 'output/'

files = os.listdir(IHC_dir)

def gen_bg_mask(orig_img):
    orig_img = np.asarray(orig_img)
    img_array = np.array(orig_img).astype(np.uint8)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    binary = np.uint8(binary)    
    dst = morphology.remove_small_objects(binary!=255,min_size=10000,connectivity=1)
    dst = morphology.remove_small_objects(dst==False,min_size=10000,connectivity=1)
    bg_mask = np.zeros(orig_img.shape[:2])
    bg_mask[dst==True]=1
    return bg_mask


for file in files:
    fname = file[:-11]
    img = cv2.imread(os.path.join(IHC_dir,file))
    img_HE = cv2.imread(os.path.join(HE_dir,fname+'_HE.jpg'))
    tissue = gen_bg_mask(img_HE)
    channel_B, channel_G, channel_R = cv2.split(img)
    # 师弟可以好好试试一下这个阈值
    threshold = [75,75,85]
    _,channel_B = cv2.threshold(channel_B, threshold[0], 255, cv2.THRESH_BINARY)
    _,channel_G = cv2.threshold(channel_G, threshold[1], 255, cv2.THRESH_BINARY)
    _,channel_R = cv2.threshold(channel_R, threshold[2], 255, cv2.THRESH_BINARY)

    mask = channel_R*channel_G*channel_B

    tmp = morphology.remove_small_objects(mask==255,min_size=150,connectivity=1)
    mask = morphology.remove_small_objects(tmp==False,min_size=150,connectivity=1)
    kernel_1 = np.ones((7, 7), np.uint8)
    kernel_2 = np.ones((7, 7), np.uint8)

    mask = cv2.dilate(np.uint8(mask), kernel_2, iterations=1)

    mask = cv2.erode(np.uint8(mask), kernel_1, iterations=1)

    tmp = morphology.remove_small_objects(mask==1,min_size=150,connectivity=1)
    mask = morphology.remove_small_objects(tmp==False,min_size=150,connectivity=1)
    mask = (mask)*1+1
    mask[tissue==1]=0
    mask = Image.fromarray(np.uint8(mask), 'P')
    mask.putpalette(palette)
    mask.save(save_dir+file[:-4]+'.png')

test = 1
