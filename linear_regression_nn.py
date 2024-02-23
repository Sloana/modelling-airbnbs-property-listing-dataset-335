import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.datasets import load_diabetes
import torch.nn.functional as F
import yaml
import joblib
import json
import os
import time
# from torch.utils.tensorboard import SummaryWriter
with open('C:/Users/laura/OneDrive/Desktop/Data Science/nn_config.yaml', 'r') as stream:
    data_loaded = yaml.safe_load(stream)
def get_nn_config():
    return data_loaded['parameters']
print(get_nn_config())
class DiabetesDataset():
    def __init__(self):
        super().__init__()
        self.X,self.y=load_diabetes(return_X_y=True)
    def __getitem__(self,idx):
        return(torch.tensor(self.X[idx]).float(), torch.tensor(self.y[idx]).float())
    def __len__(self):
        return(len(self.X))

dataset=DiabetesDataset()

batch_size=4
train_loader=DataLoader(dataset,shuffle=True, batch_size=batch_size)
example=next(iter(train_loader))
print(example)
features, labels=example
features=features.reshape(batch_size, -1)


class linearRegression(torch.nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.linear_layer=torch.nn.Linear(10,1)
    
    def forward(self, features):
        return self.linear_layer(features)
    
model=linearRegression()
print(model(features))
def train(model, epochs=10):
    lr1= get_nn_config()['learning_rate']
    optimiser=torch.optim.SGD(model.parameters(),lr=lr1)
    # writer=SummaryWriter()
    batch_idx=0
    for batch in train_loader:
        features, labels=batch
        prediction=model(features)
        loss=F.mse_loss(prediction,labels)
        loss.backward()
        print(loss)
        optimiser.step()
        optimiser.zero_grad()
        # writter.add_scalar('loss',loss.item(),batch_idx)
class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
         #define layers
        self.linear_layer=torch.nn.Linear(10,16)
        self.linear_layer2=torch.nn.Linear(16,1)
    def forward(self, X):
        X=self.linear_layer(X)
        X=F.relu(X)
        X=self.linear_layer2(X)
        return X
    
    
def chech_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    return x
        
metrics=[' RMSE_loss ','R_squared','training_duration', 'inference_latency']
# def save_models(folder):
#     joblib.dump(chech_model(), folder+"model.pt")

#     with open(folder+"hyperparameters.json", 'w') as f:
#         json.dump(get_nn_config(), f)

#     with open(folder+"metrics.json", 'w') as f1:
#         json.dump(metrics, f1)


folder= r"C:/Users/laura/OneDrive/Desktop/Data Science/neural networks/regression/"
# save_models(folder)   


if __name__=='__main__':

    folder_name = 'folder' + time.strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(folder_name)
    dataset=DiabetesDataset()
    train_loader=DataLoader(dataset,shuffle=True,batch_size=8)
    model=NN()
    train(model)
    torch.save(model.state_dict(),'model.pt')
 