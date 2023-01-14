# RSNA-End-to-End-Predicting-Cancer-probability-
#### Problem Description <a class="anchor" id="Problem Description"></a>
   * **problem** : In this competition our goal is to predict the presence or absence of cancer in mammography images. 
   * **helped Notebooks**: in our journey i will use some hopefull ideas from other notebook and i will mention them in description 
   * **Tasks we will cover in this section** 
       1. Per-Processing images 
          - understand the data 
          - explore data from diffrent view perspective 
          - Technic to processes data to feed into the mode 
               1. read images in each case Patient_ID 
               2. Resize the image and Crop the ROI ( region of intersted 
               3. save the process the image in npy format 
               4. extrat the image label from Train.Csv file 
               5. Virtualize few sample 
* **Run** 
    to the train step following this command 
    
    ```python
     python train.py --stage 'train' --gpus 0 --Epochs 200
```                
   
