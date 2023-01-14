import torch
import torchvision
from torchvision import transforms
import torchmetrics
import numpy as np
import pydicom as dc
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--gpus', default = [0])
    parser.add_argument('--Epochs', default =50)
    args = parser.parse_args()
    return args

### Loading the numpy format as .npy
def Loader_Data(path):
    return np.load(path).astype(np.float32)
### Augmenetation pipline from transform Objet trochvision 

Transform_aug_Train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
    ])
Transform_aug_Val = transforms.Compose([
    transforms.ToTensor(),
    ])
Loading_DataFolder_Train = torchvision.datasets.DatasetFolder(root="./Processed/train",
                                                              loader=Loader_Data , 
                                                              extensions="npy",
                                                              transform=Transform_aug_Train)
Loading_DataFolder_Val= torchvision.datasets.DatasetFolder(root="./Processed/val",
                                                           loader=Loader_Data , 
                                                           extensions="npy",
                                                           transform=Transform_aug_Val)
batch_size = 4
Training_Loader = torch.utils.data.DataLoader(Loading_DataFolder_Train,batch_size=batch_size, shuffle=True)
Validation_Loader = torch.utils.data.DataLoader(Loading_DataFolder_Val,batch_size=batch_size, shuffle=False)
print(f"lenght of training data is :{len(Training_Loader)}\nlenght of validationi s: {len(Validation_Loader)}")


###### Creating the model 
class CancerModel(pl.LightningModule):
    def __init__(self,weight=1):
        super().__init__()
        ### the PL of model 
        self.model = torchvision.models.resnet152()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=1)
        ### setup the Optimizer and loss functions 
        self.Optimizer = torch.optim.Adam(self.model.parameters(),lr = 1e-4)
        self.Loss_F = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([weight]))
         # simple accuracy computation
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        
    def forward(self, input_):
        Predicted_label = self.model(input_)
        return Predicted_label
        
    def training_step(self,batch , batch_idx):
        Image , label = batch 
        label= label.float()
        Predicted_label = self(Image)[:,0]
        loss = self.Loss_F(Predicted_label,label)
        # Log loss and batch accuracy
        self.log("Train Loss", loss,sync_dist=True)
        self.log("Step Train Acc", self.train_acc(torch.sigmoid(Predicted_label), label.int()))
        return loss
    
    def training_epoch_end(self, outs):
        # After one epoch compute the whole train_data accuracy
        self.log("Train Acc", self.train_acc.compute(),sync_dist=True)
        
    #######
    ### here we did the same as Traiing PL we changed only the input_data disttro
    #######
    def validation_step(self,batch , batch_idx):
        Image , label = batch 
        label = label.float()
        Predicted_label = self(Image)[:,0]
        loss = self.Loss_F(Predicted_label,label)
        # Log loss and batch accuracy
        self.log("Val Loss", loss,sync_dist=True)
        self.log("Step Val Acc", self.train_acc(torch.sigmoid(Predicted_label), label.int()))
        return loss
    
    def validation_epoch_end(self, outs):
        # After one epoch compute the whole train_data accuracy
        self.log("Val Acc", self.train_acc.compute(),sync_dist=True)  
        
    def configure_optimizers(self):
        #Caution! You always need to return a list here (just pack your optimizer into one :))
        return [self.Optimizer]  

Model_ = CancerModel() ### instanace the model from the class 
Check_Point_Callbacks = ModelCheckpoint(
    monitor="Val Acc", 
    save_top_k=12,
    mode="max")  

# Create the trainer
# Change the gpus parameter to the number of available gpus on your system. Use 0 for CPU training

 #TODO
 

if __name__ == '__main__':

    args = make_parse()
    
    Trainer = pl.Trainer(accelerator='gpu', devices=args.gpus, logger=TensorBoardLogger(save_dir= "./processed/logs_weights_ex"), log_every_n_steps=1,
                     callbacks=Check_Point_Callbacks,                    
                     max_epochs=args.Epochs)   
    Trainer.fit(Model_,Training_Loader,Validation_Loader)        