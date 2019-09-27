#IMPORTS

import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pylab as plt
import numpy as np
import time
from numpy import random
import sys, os

# This function implements PERIODIC BOUNDARY conditions for convolutional neural nets
# df ----------> density field
# padding -----> number of elements to pad

def periodic_padding(df, padding):
    
    #y       z
    #y     z
    #y   z
    #y z
    #0 x x x x

    right = df[:,:,-padding:, :, :]
    left  = df[:,:,:padding, :, :]
    df = torch.cat((df,    left), dim=2)
    df = torch.cat((right, df),   dim=2)

    top    = df[:,:,:, -padding:, :]
    bottom = df[:,:,:, :padding, :]
    df = torch.cat((df,  bottom), dim=3)
    df = torch.cat((top, df),     dim=3)

    front = df[:,:,:,:,-padding:]
    back  = df[:,:,:,:,:padding]
    df = torch.cat((df,    back),  dim=4)
    df = torch.cat((front, df),    dim=4)

    return df

    """
    df_original = df.clone()
    dims = df_original.shape
    for i in xrange(df.shape[0]):
        for j in xrange(df.shape[1]):
            for k in xrange(df.shape[2]):
                a = df[i,j,k]
                b = df_original[(i-padding)%dims[0], 
                                (j-padding)%dims[1], 
                                (k-padding)%dims[2]]
                if a!=b:
                    raise Exception('padding not properly done!!!')
    """


import periodic_padding as PD


#ARCHITECTURE

class CNN1(nn.Module):
    
    # Constructor
    def __init__(self, out_1=12, out_2=16, out_3=24, out_4=36, out_5=48, out_6=64, out_7=100, out_8=128, out_9=256, params=2):
        super(CNN1, self).__init__()

        self.cnn1 = nn.Conv3d(in_channels=1,     out_channels=out_1, kernel_size=(7,7,4), 
                              stride=1, padding=0) 
        self.cnn2 = nn.Conv3d(in_channels=out_1, out_channels=out_2, kernel_size=(7,7,5), 
                              stride=1, padding=0)
        self.cnn3 = nn.Conv3d(in_channels=out_2, out_channels=out_3, kernel_size=4, 
                              stride=2, padding=0)
        self.cnn4 = nn.Conv3d(in_channels=out_3, out_channels=out_4, kernel_size=5, 
                              stride=1, padding=0)
        self.cnn5 = nn.Conv3d(in_channels=out_4, out_channels=out_5, kernel_size=5, 
                              stride=1, padding=0)
        self.cnn6 = nn.Conv3d(in_channels=out_5, out_channels=out_6, kernel_size=5, 
                              stride=1, padding=0)
        self.cnn7 = nn.Conv3d(in_channels=out_6, out_channels=out_7, kernel_size=(5,5,4), 
                              stride=1, padding=0)
        self.cnn8 = nn.Conv3d(in_channels=out_7, out_channels=out_8, kernel_size=4, 
                              stride=2, padding=0)
        self.cnn9 = nn.Conv3d(in_channels=out_8, out_channels=out_9, kernel_size=4, 
                              stride=2, padding=0)
        
        self.BN1 = nn.BatchNorm3d(num_features=out_1)
        self.BN2 = nn.BatchNorm3d(num_features=out_2)
        self.BN3 = nn.BatchNorm3d(num_features=out_3)
        self.BN4 = nn.BatchNorm3d(num_features=out_4)
        self.BN5 = nn.BatchNorm3d(num_features=out_5)
        self.BN6 = nn.BatchNorm3d(num_features=out_6)
        self.BN7 = nn.BatchNorm3d(num_features=out_7)
        self.BN8 = nn.BatchNorm3d(num_features=out_8)
        self.BN9 = nn.BatchNorm3d(num_features=out_9)
        
        #self.AvgPool1 = nn.AvgPool3d(kernel_size=2)

        self.fc1 = nn.Linear(out_9 * 5 * 5 * 2, params) 
        #self.fc2 = nn.Linear(250, params)
        #self.fc3 = nn.Linear(250, params)

        #self.dropout1   = nn.Dropout(p=0.2)
        #self.dropout2	= nn.Dropout(p=0.5)
	
        #self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
    
    # Prediction
    def forward(self, x):
        out = self.cnn1(PD.periodic_padding(x,1)) #first convolutional layer
        out = self.BN1(out)
        out = self.LeakyReLU(out)
        #out = self.dropout1(out)
        #out = self.AvgPool1(out)
        out = self.cnn2(PD.periodic_padding(out,1)) #second convolutional layer
        out = self.BN2(out)
        out = self.LeakyReLU(out)
        #out = self.dropout1(out)
        #out = self.AvgPool2(out)
        out = self.cnn3(PD.periodic_padding(out,1)) #third convolutional layer
        out = self.BN3(out)
        out = self.LeakyReLU(out)
        #out = self.dropout1(out)
        #out = self.AvgPool3(out)
        out = self.cnn4(PD.periodic_padding(out,1)) #fourth conv. layer
        out = self.BN4(out)
        out = self.LeakyReLU(out)
        #out = self.dropout2(out)
        #out = self.AvgPool4(out)
        out = self.cnn5(PD.periodic_padding(out,1)) # fifth conv. layer
        out = self.BN5(out)
        out = self.LeakyReLU(out)
        #out = self.dropout2(out)
        out = self.cnn6(PD.periodic_padding(out,1)) # sixth conv. layer
        out = self.BN6(out)
        out = self.LeakyReLU(out)
        #out = self.dropout2(out)
        out = self.cnn7(PD.periodic_padding(out,1)) # seventh conv. layer
        out = self.BN7(out)
        out = self.LeakyReLU(out)
        #out = self.dropout2(out)
        out = self.cnn8(PD.periodic_padding(out,1)) # eighth conv. layer
        out = self.BN8(out)
        out = self.LeakyReLU(out)
        out = self.cnn9(PD.periodic_padding(out,1)) # ninth conv. layer
        out = self.BN9(out)
        #out = self.LeakyReLU(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out) # first fully connected layer
        #out = self.LeakyReLU(out)
        #out = self.dropout2(out)
        #out = self.fc2(out) # second fully connected layer
        #out = self.LeakyReLU(out)
        #out = self.dropout2(out)
        #out = self.fc3(out) # third fully connected layer
        
        return out
        
#ROTATION FUNCTION for data augmentation

def random_rotation(dataset, rotation):
    
    if rotation ==0:
        choice = np.rot90(dataset, 1, (0,1)) #rotating along x-axis only 
    elif rotation ==1:
        choice = np.rot90(dataset, 2, (0,1)) 
    elif rotation ==2:
        choice = np.rot90(dataset, 3, (0,1))
    elif rotation ==3:
        choice = np.rot90(dataset, 4, (0,1)) #the identity

        
    return choice



#define DATA SET CLASS

parameters = [0,4]  #[0,1,2,3,4] ---> [Om, Ob, h, ns, s8] #set the parameters 
length =2000
offset=0
num_rot=4 									#set the number of rotations



class Data_set(Dataset):
    def __init__(self, offset, length, num_rot, parameters):
    
        rotations = np.zeros((length, num_rot), dtype=np.int32)

        for i in range(length):
            rotations[i]= np.random.choice(num_rot, num_rot, replace = False)


        cp=np.loadtxt('/mnt/ceph/users/fvillaescusa/Ana/latin_hypercube/latin_hypercube_params.txt')
        cp=cp[:,parameters]
        #print(cp[0,:])
        mean, std = np.mean(cp, axis=0), np.std(cp,  axis=0)
        self.mean, self.std = mean, std
        cp = (cp - mean)/std #normalize the labels
        #print(cp[0,:])
        cp = cp[offset:offset+length]
        #print(cp[0,:])
        #print('cp shape', cp.shape)
        self.cosmo_params = np.zeros((length*num_rot, cp.shape[1]), dtype=np.float32)
        count = 0
        for i in range(length):
            for j in range(num_rot):
                self.cosmo_params[count] = cp[i]
                count += 1
        self.cosmo_params = torch.tensor(self.cosmo_params, dtype=torch.float)
        #print(self.cosmo_params.size())
        #print(self.cosmo_params[3990:4010,:])
        #print(len(self.cosmo_params))
        #print(mean, std)

        
        # read all 3D prisms
        self.cubes = np.zeros((length*num_rot, 64, 64, 33), dtype=np.float32)
        
        count = 0 
        for i in range(length):
            f_df = '/mnt/ceph/users/fvillaescusa/Ana/Fourier_grid/density_fields/df_Fourier_64_%d.npy'%(i)
            df = np.load(f_df)
            #df = (df - np.min(df))/(np.max(df)-np.min(df)) 
            for j in range(num_rot):
                df_rotated = random_rotation(df, rotations[i,j])
                self.cubes[count] = df_rotated
                count += 1
        min_df = np.min(self.cubes)
        max_df = np.max(self.cubes)
        self.cubes = (self.cubes - min_df)/(max_df - min_df) #normalize the images
        self.cubes = torch.tensor(self.cubes)
        #print(self.images.size())
        self.len = length*num_rot
    def __getitem__(self, index):
        return self.cubes[index].unsqueeze(0), self.cosmo_params[index]
    
    def __len__(self):
        return self.len

#define a device
device = torch.device('cpu')

if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')



# SPLIT THE DATA into training, vlidation and test sets
print('splitting the dataset...')

#manually

train = Data_set(0, 1800, num_rot, parameters)
print('train set')
validation = Data_set(1800, 100, num_rot, parameters)
print('validation set')
test = Data_set(1900, 100, num_rot, parameters)
print('test set')



'''
#save for future use

torch.save(train, 'fourier_trainset_norm_4rot.pt') 			#remember to RENAME as necessary
torch.save(validation, 'fourier_validset_norm_4rot.pt') 		#remember to RENAME as necessary
torch.save(test, 'fourier_testset_norm_4rot.pt')  			#remember to RENAME as necessary
'''

#Load data

print('Loading the data...')

train = torch.load('fourier_trainset_norm_4rot.pt')
print('train set')
validation = torch.load('fourier_validset_norm_4rot.pt')
print('validation set')
test = torch.load('fourier_testset_norm_4rot.pt')
print('test set') 

#TRAIN the model

print('preparing to train...')

batch_size = 20    #size of batches
epochs     = 50    #number of epochs
lr         = .001  #learning rate

fout   = 'data/losses_CNN_9_fourier_1fcl_9_26_norm_4rot.txt'   #file to save the losses,                   #RENAME
fmodel = 'best_models/CNN_9_fourier_1fcl_9_26_norm_4rot.pt'   #file to save the weights of the best model, #RENAME

#fixed   = True
#padding = False
num_workers = 1

#cosmo_dict = {0:'dOm', 1:'dOb', 2:'dh', 3:'dns', 4:'ds8'}
cosmo_dict = {0:'dOm', 4:'ds8'}

print('preparing dataset loaders')

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=num_workers)

num_train_batches = len(train_loader)
num_valid_batches = len(valid_loader)

# If output file exists, delete it
if os.path.exists(fout):  os.system('rm %s'%fout)

# define model, loss and optimizer 
print('Initializing...')
#model = arquitecture.FC1(grid=32, params=len(parameters))
model = CNN1(params=len(parameters))
model.to(device=device)

if os.path.exists(fmodel):
    print("Loading model : Are we sure we want to load now ?")
    model.load_state_dict(torch.load(fmodel))
    model.to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, 
                                                       patience=15, min_lr=1e-6)
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
pytorch_total_params = sum(p.numel() for p in model.parameters())
print('total number of parameters in the model =',pytorch_total_params)


# get validation loss
print('Computing initial validation loss')
model.eval()
min_valid_loss = 0.0
for x_val, y_val in valid_loader:
    with torch.no_grad():
        x_val = x_val.to(device=device)
        y_val = y_val.to(device=device)
        y_pred2 = model(x_val)
        min_valid_loss += criterion(y_pred2, y_val).item()

min_valid_loss /= num_valid_batches
print('Initial valid loss = %.3e'%min_valid_loss)
first_model = 5 #random number to get it started

# train model
start = time.time()
print('Starting training...')
for epoch in range(epochs):
    train_loss, valid_loss = 0.0, 0.0
        
    # do training
    batch_num = 0
    model.train()
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        x_train=x_train.to(device)
        y_train=y_train.to(device)
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        #if batch_num==50:  break
        batch_num += 1
    train_loss /= batch_num

    # do validation
    model.eval()
    error = torch.zeros(len(parameters))
    for x_val, y_val in valid_loader:
        with torch.no_grad():
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            y_pred2 = model(x_val)
            valid_loss += criterion(y_pred2, y_val).item()
            error += torch.sum((y_pred2.cpu() - y_val.cpu())**2, dim=0)
            
    valid_loss /= num_valid_batches
    error /= len(validation)

    scheduler.step(valid_loss)
    
    
    # save model if it is better
    

    best_model = np.abs(train_loss - valid_loss)
    if best_model<first_model:
        print('saving model...')
        torch.save(model.state_dict(), fmodel)
        first_model = best_model
    
    
        # print some information
    print('%03d ---> train loss = %.3e : valid loss = %.3e'\
          %(epoch, train_loss, valid_loss))
    for i,j in enumerate(parameters):
        print('%03s = %.3f'%(cosmo_dict[j], error[i]))
    
    # save losses to file
    f = open(fout, 'a')
    f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))
    f.close()
    
stop = time.time()
print('Time (m):', "{:.4f}".format((stop-start)/60.0))




#TEST THE MODEL

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, 
                                          shuffle=True, num_workers=num_workers)

test_results   = 'results_CNN_9_fourier_1fcl_9_26_norm_4rot.txt'  				#RENAME

#mean, std = [0.3, 0.05, 0.7, 1.0, 0.8], [0.11547004, 0.011547, 0.11547004, 0.11547004, 0.11547004]
mean, std = [0.3, 0.8], [0.11547004, 0.11547004] #computed beforhenad
print(mean,std)


# If output file exists, delete it
if os.path.exists(test_results):  os.system('rm %s'%test_results)

num_params = len(parameters)

# get the pretrained model
##model = arquitecture.CNN1(params=num_params)
#model = arquitecture.FC1(params=num_params)
model.load_state_dict(torch.load(fmodel))

# get parameters from trained network
print('Testing results...')
model.eval()
results = np.zeros((len(test), 2*num_params), dtype=np.float32)
offset = 0
for x_test, y_test in test_loader:
    with torch.no_grad():
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        y_pred3 = model(x_test)
        length = len(x_test)
        results[offset:offset+length, 0:num_params]            = mean + y_test.cpu().numpy()*std
        results[offset:offset+length, num_params:2*num_params] = mean + y_pred3.cpu().numpy()*std
        offset += length
np.savetxt(test_results, results)
#print(y_test)

for i,j in enumerate(parameters):
    print('%s = %.3f'%(cosmo_dict[j], error[i]))

