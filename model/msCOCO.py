import os
import scipy.io
from pycocotools.coco import COCO
import pylab


# coco annotation files
pylab.rcParams['figure.figsize'] = (8.0, 10.0)   
dataDir='/worktmp/hxkhkh/data/coco/MSCOCO'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
anncaptionFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)

# initialize COCO api 
coco=COCO(annFile)
coco_caps=COCO(anncaptionFile)   
cats = coco.loadCats(coco.getCatIds())
cats_id = [item['id'] for item in cats]
cats_names = [item['name']for item in cats]


all_images_data = coco.imgs

image_id = 42
example_image_data = coco.imgs[image_id]

annId_img = coco.getAnnIds( imgIds=image_id, iscrowd=False) 
anns_image = coco.loadAnns(annId_img)

for item in range(len(anns_image)):
    
    item_catId = anns_image[item]['category_id']
    item_catinfo = coco.loadCats(item_catId)[0]
    
    item_catName = item_catinfo['name']
    item_supercatName  = item_catinfo['supercategory']   


for item in anns_image:
    mask_temp = coco.annToMask(item )
    
from matplotlib import pyplot as plt
plt.imshow(mask_temp)

example_image_feature_path = "/run/media/hxkhkh/khazar_data_1/khazar/features/coco/MSCOCO/train/COCO_val2014_000000000042"


import pickle
def load_data (filepath):
    infile = open(filepath ,'rb')
    data = pickle.load(infile)
    infile.close()
    return data

import numpy
example_image_data = load_data (example_image_feature_path)
example_image_FM = numpy.reshape(example_image_data, [14,14, 512]) 
plt.imshow(numpy.average(example_image_FM, axis = -1))

from PIL import Image
import requests
img_id = 42
img_info = coco.loadImgs([img_id])[0]
img_file_name = img_info["file_name"]
img_url = img_info["coco_url"]
im = Image.open(requests.get(img_url, stream=True).raw)
plt.axis("off")
plt.imshow(numpy.asarray(im))