from collections.abc import Iterable
import types

import pydicom as dc 
import pandas as pd 
import cv2

from pathlib import Path
import os 
from glob import glob

balanced_df = pd.read_csv("balanced.csv",index_col=False)
# we will need to randome selecte the number rows for not keep it sorted 
random_label_df = balanced_df.sample(n=2316,random_state =70,replace=True)
# https://www.kaggle.com/code/snnclsr/roi-extraction-using-opencv
def crop_coords(img):
    """
    Crop ROI from image.
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)


def normalization_(img):
    """
    WindowCenter and normalize pixels in the breast ROI.
    return: numpy array of the normalized image
    """
    # Convert to float to avoid overflow or underflow losses.
    image_2d = img.astype(float)
    image_2d_scaled = (np.maximum(image_2d,0) / image_2d.max()) * 255.0
    # Convert to uint
    normalized = np.uint8(image_2d_scaled)
    return normalized

def Processing_img(Path_dir , label_csv , image_id):
    data_id = label_csv.iloc[image_id]
    image = str(data_id.patient_id)+ "/" + str(data_id.image_id) 
    Image_path = os.path.join(Path_dir,image) + ".dcm" 
    read_image = dc.read_file(Image_path)
    Array_pixel = read_image.pixel_array
#     if read_image.PhotometricInterpretation == "MONOCHROME1":
#         Array_pixel = np.amax(Array_pixel) - Array_pixel
    (x,y,w,h) = crop_coords(Array_pixel)
    image_crop = Array_pixel[y:y+h,x:x+w]
    normalize_img =normalization_(image_crop)
    # Resize the image to the final shape. 
    img_final = cv2.resize(normalize_img, (120, 120)).astype(np.float32)
    return img_final , data_id

## set the clean dataFrame balanced_df
sums = 0 
sums_square = 0 ### to computer Normalize means and STD to use later 
Normalizer = 120*120
Processed_data = "./Processed"
for idx_img , Patient_ID in enumerate(tqdm(random_label_df.image_id)):
    ### Extract label Patient 
    image_processed , label_id = Processing_img(Path_train,random_label_df,image_id=idx_img)
    Image_id = str(label_id.image_id)
    label = label_id.cancer
    Train_or_Val = "train" if idx_img < 1800 else "val" ## total samples are 2316 we split it by 
    ## creat save folder 
    Save_path = Path(Processed_data + "/"+ Train_or_Val +"/"+ str(label))
    Save_path.mkdir(parents=True,exist_ok=True)
    #         ### the struct dircotry is 
#         ## Train/
#         #        lable_0/
#         #              Image_ID
#         #        labek_1/
#         #              Image_ID 
#         #####
    np.save(os.path.join(Save_path,Image_id),image_processed)
    if Train_or_Val == "train":
        sums += np.sum(image_processed) / Normalizer 
        sums_square += (image_processed ** 2).sum() / Normalizer     

