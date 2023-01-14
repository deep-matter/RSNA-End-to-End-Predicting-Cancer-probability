import torch
import torchvision
from torchvision import transforms
import torchmetrics
import numpy as np
import glob
import matplotlib.pyplot as plt 
import pydicom as dc

#### Loader data function 
def loader(path):
    return np.load(path).astype(np.float32)
#### set the transfoms augmentation 
transform = transforms.Compose([
    transforms.ToTensor(),
    ])
validation_set = Path("./Processed/val/")

load_data = torchvision.datasets.DatasetFolder(root=validation_set , loader=loader,extensions="npy",transform=transform)

#### building the model 

class CancerModel_val(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet152()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=1)
        
        ### get the features map from the model 2
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])
        
    def forward(self, input_):
        feature_map = self.feature_map(input_)
        pooling_avg = torch.nn.functional.adaptive_avg_pool2d(feature_map , output_size=(1,1))
        flatten_pool = torch.flatten(pooling_avg)
        Predict_= self.model.fc(flatten_pool)
        return feature_map ,Predict_ 


### intialze the model 
Model_Cancer = CancerModel_val()
Model_Cancer.load_from_checkpoint("./processed/logs_weights_ex4/lightning_logs/version_0/checkpoints/epoch=89-step=16000.ckpt",strict=False)
Model_Cancer.eval();

### compute the Grad-Cam function 

def Cam(image,model):
    with torch.no_grad():
        image = image[0]
        feature_map , pred = Model_Cancer(image.unsqueeze(0))
    print(feature_map.shape)
    feature = feature_map.reshape((2048,4*4))
    weights = list(model.model.fc.parameters())[0]
    weights_param = weights[0].detach()
    cam = torch.matmul(weights_param,feature) 
    print(cam.shape)
    cam_img = cam.reshape(4,4).cpu()
    return cam_img , torch.sigmoid(pred)

def virtulizer(img,cam ,pred):
    img = img[0]
    cam_img =transforms.functional.resize(cam.unsqueeze(0),(120,120))[0]
    fig ,axis = plt.subplots(1,2 , figsize=(4,4))
    axis[0].imshow(img,cmap="bone")
    axis[1].imshow(img,cmap="bone")
    axis[1].imshow(cam_img,cmap="jet")
    if pred >0.5 :
        plt.title("positive")
    else:
        plt.title("negative")

##### testing 
# 1. compute 
img = load_data[-5][0]
map_activation , Predicted = Cam(img.unsqueeze(0),Model_Cancer)

if __name__ == '__main__':
    
    virtulizer(img,map_activation,Predicted)
