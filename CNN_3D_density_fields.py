
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





#PERIODIC PADDING

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


#ROTATION FUNCTION for data augmentation

def random_rotation(dataset, rotation):
    
    if rotation ==0:
        choice = np.rot90(dataset, 1, (0,1)) #this represents the x-axis rotations
    elif rotation ==1:
        choice = np.rot90(dataset, 1, (0,2)) #this represents the y-axis rotations
    elif rotation ==2:
        choice = np.rot90(dataset, 1, (1,2)) #this represents the z-axis rotations
    # let's get the combined rotations:
    elif rotation ==3:
        choice = np.rot90(dataset, 2, (0,1)) #this represents the x-axis only rotations
    elif rotation ==4:
        choice = np.rot90(dataset, 3, (0,1))
    #combine rotations from x-axis rotations:
    elif rotation ==5:
        choice = np.rot90(np.rot90(dataset, 2, (0,1)), 1, (0,2)) # additional rotation along y-axis
    elif rotation ==6:
        choice= np.rot90(np.rot90(dataset, 3, (0,1)), 1, (0,2)) # additional rotation along y-axis
    elif rotation ==7:
        choice = np.rot90(np.rot90(dataset, 2, (0,1)), 1, (1,2)) # additional rotation along z-axis
    elif rotation ==8:
        choice = np.rot90(np.rot90(dataset, 3, (0,1)), 1, (1,2)) # additional rotation along z-axis
    #now we go back and do the y-axis only rotations:
    elif rotation ==9:
         choice = np.rot90(dataset, 2, (0,2)) #this represents the y-axis only rotations
    elif rotation ==10:
         choice= np.rot90(dataset, 3, (0,2))
    #combine rotations from y-axis rotations:
    elif rotation ==11:
         choice= np.rot90(np.rot90(dataset, 2, (0,2)), 1, (0,1)) # additional rotation along x-axis
    elif rotation ==12:
        choice = np.rot90(np.rot90(dataset, 3, (0,2)), 1, (0,1)) # additional rotation along x-axis
    elif rotation ==13:
        choice = np.rot90(np.rot90(dataset, 2, (0,2)), 1, (1,2)) # additional rotation along z-axis
    elif rotation ==14:
        choice = np.rot90(np.rot90(dataset, 3, (0,2)), 1, (1,2)) # additional rotation along z-axis
    # now we go back and do z-axis only rotations:
    elif rotation ==15:
         choice = np.rot90(dataset, 2, (1,2)) #this represents the z-axis rotations
    elif rotation ==16:
        choice = np.rot90(dataset, 3, (1,2))
    #combine rotations from z-axis rotations:
    elif rotation ==17:
        choice = np.rot90(np.rot90(dataset, 2, (1,2)), 1, (0,1)) # additional rotation along x-axis
    elif rotation ==18:
        choice = np.rot90(np.rot90(dataset, 3, (1,2)), 1, (0,1)) 
    elif rotation ==19:
        choice = np.rot90(np.rot90(dataset, 2, (1,2)), 1, (0,2))# additional rotation along y-axis
    elif rotation ==20:
        choice = np.rot90(np.rot90(dataset, 3, (1,2)), 1, (0,2))
    #we have a single 1x1 rotation
    elif rotation ==21:
        choice = np.rot90(np.rot90(dataset, 1, (0,1)), 1, (1,2))# 1 rotation along x and 1 along z 
    #we have a single 3x3 rotation
    elif rotation ==22:
        choice = np.rot90(np.rot90(dataset, 3, (0,1)), 3, (1,2))# 3 rotation along x and 3 along y 
    #let's include the identity
    elif rotation ==23:
        choice = dataset #no rotations
    
    return choice




#ARCHITECTURE

#3-D constructor
class CNN1(nn.Module):
    
    def __init__(self, out_1=4, out_2=8, out_3=32, out_4=64, out_5=128, out_6=256, params=2):
        super(CNN1, self).__init__()

        self.cnn1 = nn.Conv3d(in_channels=1,     out_channels=out_1, kernel_size=5, 
                              stride=1, padding=0) 
        self.cnn2 = nn.Conv3d(in_channels=out_1, out_channels=out_2, kernel_size=5, 
                              stride=1, padding=0)
        self.cnn3 = nn.Conv3d(in_channels=out_2, out_channels=out_3, kernel_size=4, 
                              stride=2, padding=0)
        self.cnn4 = nn.Conv3d(in_channels=out_3, out_channels=out_4, kernel_size=3, 
                              stride=3, padding=0)
        self.cnn5 = nn.Conv3d(in_channels=out_4, out_channels=out_5, kernel_size=3, 
                              stride=3, padding=0)
        #self.cnn6 = nn.Conv3d(in_channels=out_5, out_channels=out_6, kernel_size=3, 
                              #stride=3, padding=0)

        self.BN1 = nn.BatchNorm3d(num_features=out_1)
        self.BN2 = nn.BatchNorm3d(num_features=out_2)
        self.BN3 = nn.BatchNorm3d(num_features=out_3)
        self.BN4 = nn.BatchNorm3d(num_features=out_4)
        self.BN5 = nn.BatchNorm3d(num_features=out_5)
        #self.BN6 = nn.BatchNorm3d(num_features=out_6)
        
        self.AvgPool1 = nn.AvgPool3d(kernel_size=2)
        self.AvgPool2 = nn.AvgPool3d(kernel_size=2)
        #self.AvgPool3 = nn.AvgPool3d(kernel_size=2)
        #self.AvgPool4 = nn.AvgPool3d(kernel_size=2)

        self.fc1 = nn.Linear(out_5 * 2 * 2 * 2, 250) 
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, params)
	
        self.dropout   = nn.Dropout(p=0.5)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
       
    
    # Prediction
    def forward(self, x):
        out = self.cnn1(periodic_padding(x,2)) #first convolutional layer
        out = self.BN1(out)
        out = self.LeakyReLU(out)
        out = self.AvgPool1(out)
        out = self.cnn2(periodic_padding(out,2)) #second convolutional layer
        out = self.BN2(out)
        out = self.LeakyReLU(out)
        out = self.AvgPool2(out)
        out = self.cnn3(periodic_padding(out,1)) #third convolutional layer
        out = self.BN3(out)
        out = self.LeakyReLU(out)
        #out = self.AvgPool3(out)
        out = self.cnn4(periodic_padding(out,2)) #fourth conv. layer
        out = self.BN4(out)
        out = self.LeakyReLU(out)
        #out = self.AvgPool4(out)
        out = self.cnn5(periodic_padding(out,1)) # fifth conv. layer
        out = self.BN5(out)
        out = self.LeakyReLU(out)
        #out = self.cnn6(periodic_padding(out,1)) # sixth conv. layer
        #out = self.BN6(out)
        #out = self.LeakyReLU(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out) # first fully connected layer
        out = self.LeakyReLU(out)
        out = self.dropout(out)
        out = self.fc2(out) # second fully connected layer
        out = self.LeakyReLU(out)
        out = self.dropout(out)
        out = self.fc3(out) # third fully connected layer
        
        
        return out


#DEFINE DATA SET CLASS

parameters = [0,4]  						#[0,1,2,3,4] ---> [Om, Ob, h, ns, s8] 
length =2000 							#dataset size
offset=0
total_choice = 24 						#number of available rotaions
num_rot=12							#number of rotations to apply

class Data_set(Dataset):
    def __init__(self, offset, length, num_rot, parameters):
        
        rotations = np.zeros((length, num_rot), dtype=np.int32)
        
        for i in range(length):
            rotations[i] = np.random.choice(total_choice, num_rot)
        
        #read file with value of the cosmological parameters
        cp = np.loadtxt('/mnt/ceph/users/fvillaescusa/Ana/latin_hypercube/latin_hypercube_params.txt')
        cp = cp[:,parameters]
        mean, std = np.mean(cp, axis=0), np.std(cp,  axis=0)
        self.mean, self.std = mean, std
        cp = (cp - mean)/std 					#normalize the labels
        cp = cp[offset:offset+length]
        self.cosmo_params = np.zeros((length*num_rot, cp.shape[1]), dtype=np.float32)
        count = 0
        for i in range(length):
            for j in range(num_rot):
                self.cosmo_params[count] = cp[i]
                count += 1
        self.cosmo_params = torch.tensor(self.cosmo_params, dtype=torch.float)


        # read all 3D cubes
        self.cubes = np.zeros((length*num_rot, 64, 64, 64), dtype=np.float32)
        count = 0 
        for i in range(length):
            f_df = '/mnt/ceph/users/fvillaescusa/Ana/latin_hypercube/%d/df_m_64_z=0.npy'%(i)
            df = np.load(f_df)
            for j in range(num_rot):
                df_rotated = random_rotation(df, rotations[i,j])
                self.cubes[count] = df_rotated
                count += 1
        min_df = np.min(self.cubes)
        max_df = np.max(self.cubes)
        self.cubes = (self.cubes - min_df)/(max_df - min_df) #normalize the cubes
        self.cubes = torch.tensor(self.cubes)
        self.len = length*num_rot
     
        
    def __getitem__(self, index):
        return self.cubes[index].unsqueeze(0), self.cosmo_params[index]
    
    def __len__(self):
        return self.len


#DEFINE A DEVICE

device = torch.device('cpu')

if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')


#SPLIT THE DATA INTO TRAINING, VALIDATION, AND TEST SETS

#using a data splitter
datasetsize=length*num_rot
lengths = [int(datasetsize*0.8), int(datasetsize*0.1), int(datasetsize*0.1)]
torch.manual_seed(datasetsize)
print('splitting the dataset...')
           
train, validation, test = torch.utils.data.random_split(Data_set(offset = offset, length=length, parameters=parameters, num_rot=num_rot), lengths)


#TRAIN THE MODEL

print('preparing to train...')

batch_size = 20							#size of batches
epochs     = 200    	 					#number of epochs
lr         = 5e-5   						#learning rate

fout   = 'losses/3d_losses__5cnn_3fcl_b99_12_rot.txt'      	#losses for plotting 	RENAME
fmodel = 'best_models/3d_5cnn_3fcl_b99_12_rot.pt'   		#best model		RENAME 

num_workers = 1

#cosmo_dict = {0:'dOm', 1:'dOb', 2:'dh', 3:'dns', 4:'ds8'}	#dictionary to hold parameter names
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
model = CNN1(params=len(parameters)) 				#model
model.to(device=device)

if os.path.exists(fmodel):  
    print("Loading model : Are we sure we want to load now ?")
    model.load_state_dict(torch.load(fmodel))
    model.to(device=device)
    #print(model.state_dict())

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)	#optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
								patience=5, verbose=True)	#reduce learning rate


criterion = nn.MSELoss()					#criterion
#criterion = nn.L1Loss()
pytorch_total_params = sum(p.numel() for p in model.parameters())
print('total number of parameters in the model =',pytorch_total_params)


#get validation loss
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

#train model
start = time.time()
print('Starting training...')
for epoch in range(epochs):
    train_loss, valid_loss = 0.0, 0.0
        
    #do training
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

    #save model if it is better
    if valid_loss<min_valid_loss:
        print('SAVING MODEL...')
        torch.save(model.state_dict(), fmodel)
        min_valid_loss = valid_loss

    #print some information
    print('%03d ---> train loss = %.3e : valid loss = %.3e'\
          %(epoch, train_loss, valid_loss))
    for i,j in enumerate(parameters):
        print('%03s = %.3f'%(cosmo_dict[j], error[i]))
    
    #save losses to file
    f = open(fout, 'a')
    f.write('%d %.5e %.5e\n'%(epoch, train_loss, valid_loss))
    f.close()
    
stop = time.time()
print('Time (m):', "{:.4f}".format((stop-start)/60.0))




#TEST THE MODEL

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, 
                                          shuffle=True, num_workers=num_workers)

test_results   = 'predictions/results_3d_5cnn_3fcl_b99_12_rots_200epochs_11_24.txt'  		#RENAME

#mean, std = [0.3, 0.05, 0.7, 1.0, 0.8], [0.11547004, 0.011547, 0.11547004, 0.11547004, 0.11547004] #computed beforhenad
mean, std = [0.3, 0.8], [0.11547004, 0.11547004] 					#computed beforhenad


#If output file exists, delete it
if os.path.exists(test_results):  os.system('rm %s'%test_results)

num_params = len(parameters)

#get the pretrained model
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
np.savetxt(test_results, results)				#save prediction results

for i,j in enumerate(parameters):
    print('%s = %.3f'%(cosmo_dict[j], error[i]))		#print frinal error on paramaters

