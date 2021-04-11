import cv2
'''
im_gray = cv2.imread('love-me-do-vox.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE) 
(thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
thresh = 127 
im_binary = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)
cv2.imwrite('binary_image.jpg', im_bw) 
'''

from PIL import Image 
import numpy as np
import os

curr_dir = os.getcwd()
vox_dir = curr_dir + '\\vox\\'
mask_dir = curr_dir + '\\masks\\'
print("vox is ?", vox_dir)
it = 1

for possible_masks in os.listdir(vox_dir):
    print("on " + str(it) + " of " + str(len(vox_dir) +1))
    it = it +1
    
    working_dir = vox_dir + possible_masks
    #print("WORKING DIR IS ", working_dir)
    vox_name = '\\' + working_dir[working_dir.rindex('\\'):]
    vox_name = vox_name.replace("\\","")
    #print("getting", vox_name)
    col = Image.open(working_dir) #read image 
    gray = col.convert('L')  #conversion to gray scale 
    bw = gray.point(lambda x: 0 if x<50 else 255, '1')  #binarization 
    bw.save(mask_dir + vox_name + "-mask.jpg") #save it 
