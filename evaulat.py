import torch
import torchvision
from torchvision import transforms
import torchmetrics
import pydicom as dc
import numpy as np
import glob

## Loading the model check point weight 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_model = "./processed/logs_weights_ex/lightning_logs/version_0/checkpoints/"
last_epoch = glob.glob(path_model+'*.ckpt')[-1]
model_load = CancerModel.load_from_checkpoint(last_epoch)
model_load.to(device)
model_load.eval();

predicted_ = []
labels_ = []
with torch.no_grad():
    for image , label in tqdm(Loading_DataFolder_Val):
        ## here we will need to load the data into device by hand 
        image = image.to(device).float().unsqueeze(0)
        label = label
        predicted= torch.sigmoid(model_load(image)[0]).cpu()
        predicted_.append(predicted)
        labels_.append(label)
    print(image.shape) 
    print(predicted.shape)
            
Tensor_Pre = torch.tensor(predicted_)        
Tensor_Lab = torch.tensor(labels_).int() 


# Make predictions on test images, write out sample submission
patient_ID = []
Score_ID= []
#my_formatter = "{0:.2f}"
def predict(image_fps, min_conf=0.90):
        with torch.no_grad():
            for image_id in tqdm(image_fps):
                ds = dc.read_file(image_id)
                image = ds.pixel_array  / 255 ##normalize 
                image_2d = image.astype(float)
                image = cv2.resize(image , (120,120)).astype(np.float32)
                image = np.expand_dims(image,axis=0)
                patient_id = os.path.splitext(os.path.basename(image_id))[0]
                patient_ID.append(patient_id)
                image= torch.from_numpy(image)
                image = image.to(device).float().unsqueeze(0)
                results = torch.sigmoid(model_load(image)[0]).cpu()
                score =  results[0].item()
                Score = str(score) if float(score) > 0.60 else str(score) 
                Score_ID.append(Score)

predict(test_image_fps)

