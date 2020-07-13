
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

neurons = 1000

class CNN1(nn.Module):
    
    def __init__(self, out_1=1, out_2=1, out_3=2, out_4=4, out_5=4, out_6=8,out_7=8, out_8=16, out_9=16, out_10=32, out_11=24, out_12=128, params=1):
        super(CNN1, self).__init__()

        self.cnn1 = nn.Conv3d(in_channels=1,     out_channels=out_1, kernel_size=3, 
                              stride=1, padding=0) 
        self.cnn2 = nn.Conv3d(in_channels=out_1, out_channels=out_2, kernel_size=3, 
                              stride=1, padding=0)
        self.cnn3 = nn.Conv3d(in_channels=out_2, out_channels=out_3, kernel_size=2, 
                              stride=2, padding=0)
        self.cnn4 = nn.Conv3d(in_channels=out_3, out_channels=out_4, kernel_size=3, 
                              stride=1, padding=0)
        self.cnn5 = nn.Conv3d(in_channels=out_4, out_channels=out_5, kernel_size=3, 
                              stride=2, padding=0)
        self.cnn6 = nn.Conv3d(in_channels=out_5, out_channels=out_6, kernel_size=3, 
                              stride=1, padding=0)
        self.cnn7 = nn.Conv3d(in_channels=out_6, out_channels=out_7, kernel_size=3, 
                              stride=2, padding=0)
        self.cnn8 = nn.Conv3d(in_channels=out_7, out_channels=out_8, kernel_size=3, 
                              stride=1, padding=0)
        self.cnn9 = nn.Conv3d(in_channels=out_8, out_channels=out_9, kernel_size=3, 
                              stride=2, padding=0)
        self.cnn10 = nn.Conv3d(in_channels=out_9, out_channels=out_10, kernel_size=3, 
                              stride=1, padding=0)
        self.cnn11 = nn.Conv3d(in_channels=out_10, out_channels=out_11, kernel_size=3, 
                              stride=2, padding=0)
        self.cnn12 = nn.Conv3d(in_channels=out_11, out_channels=out_12, kernel_size=3, 
                              stride=1, padding=0)

        self.BN1 = nn.BatchNorm3d(num_features=out_1)
        self.BN2 = nn.BatchNorm3d(num_features=out_2)
        self.BN3 = nn.BatchNorm3d(num_features=out_3)
        self.BN4 = nn.BatchNorm3d(num_features=out_4)
        self.BN5 = nn.BatchNorm3d(num_features=out_5)
        self.BN6 = nn.BatchNorm3d(num_features=out_6)
        self.BN7 = nn.BatchNorm3d(num_features=out_7)
        self.BN8 = nn.BatchNorm3d(num_features=out_8)
        self.BN9 = nn.BatchNorm3d(num_features=out_9)
        self.BN10 = nn.BatchNorm3d(num_features=out_10)
        self.BN11 = nn.BatchNorm3d(num_features=out_11)
        self.BN12 = nn.BatchNorm3d(num_features=out_12)

        
        #self.AvgPool1 = nn.AvgPool3d(kernel_size=2)
        #self.AvgPool2 = nn.AvgPool3d(kernel_size=2)



        self.fc1 = nn.Linear(out_12 * 2 * 2 * 2, neurons) 
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, params)
	
        self.dropout   = nn.Dropout(p=0.2)
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
        out = self.cnn1(periodic_padding(x,1)) #first convolutional layer
        #out = self.BN1(out)
        out = self.LeakyReLU(out)
        #out = self.AvgPool1(out)
        out = self.cnn2(periodic_padding(out,1)) #second convolutional layer
        out = self.BN2(out)
        out = self.LeakyReLU(out)
        #out = self.AvgPool2(out)
        out = self.cnn3(periodic_padding(out,1)) #third convolutional layer
        out = self.BN3(out)
        out = self.LeakyReLU(out)
        #out = self.AvgPool3(out)
        out = self.cnn4(periodic_padding(out,1)) #fourth conv. layer
        out = self.BN4(out)
        out = self.LeakyReLU(out)
        #out = self.AvgPool4(out)
        out = self.cnn5(periodic_padding(out,1)) # fifth conv. layer
        out = self.BN5(out)
        out = self.LeakyReLU(out)
        out = self.cnn6(periodic_padding(out,1)) # sixth conv. layer
        out = self.BN6(out)
        out = self.LeakyReLU(out)
        out = self.cnn7(periodic_padding(out,1)) #seventh conv. layer
        out = self.BN7(out)
        out = self.LeakyReLU(out)
        out = self.cnn8(periodic_padding(out,1)) #eighth conv. layer
        out = self.BN8(out)
        out = self.LeakyReLU(out)
        out = self.cnn9(periodic_padding(out,1)) # ninth conv. layer
        out = self.BN9(out)
        out = self.LeakyReLU(out)
        out = self.cnn10(periodic_padding(out,1)) # tenth conv. layer
        out = self.BN10(out)
        out = self.LeakyReLU(out)
        out = self.cnn11(periodic_padding(out,1)) # 11th conv. layer
        out = self.BN11(out)
        out = self.LeakyReLU(out)
        out = self.cnn12(periodic_padding(out,1)) # 12th conv. layer
        out = self.BN12(out)
        out = self.LeakyReLU(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out) # first fully connected layer
        out = self.LeakyReLU(out)
        #out = self.dropout(out)
        out = self.fc2(out) # second fully connected layer
        out = self.LeakyReLU(out)
        #out = self.dropout(out)
        out = self.fc3(out) # third fully connected layer
        
        return out

parameters = [0] 						#[0,1,2,3,4] ---> [Om, Ob, h, ns, s8] 
length =3000 							#dataset size
offset=0
total_choice = 24 						#number of available rotaions
num_rot=1							#number of rotations to apply
        
#rotations = np.zeros((length, num_rot), dtype=np.int32)        
#for i in range(length):
	#rotations[i] = np.random.choice(total_choice, num_rot)

#DEFINE DATA SET CLASS


class Data_set(Dataset):
    def __init__(self, offset, length, parameters):
        
        #read file with value of the cosmological parameters
        cp = np.load('/mnt/ceph/users/fvillaescusa/Ana/Gaussian_density_fields_3D/data/A_values_3D.npy')
        #cp = cp[:,parameters]
        mean, std = np.mean(cp, axis=0), np.std(cp,  axis=0)
        self.mean, self.std = mean, std
        cp = (cp - mean)/std 					#normalize the labels
        #cp = cp[offset:offset+length]
        self.cosmo_params = np.zeros((length, 1), dtype=np.float32)
        count = 0
        for i in range(length):
            self.cosmo_params[count] = cp[offset+i]
            count += 1
        self.cosmo_params = torch.tensor(self.cosmo_params, dtype=torch.float)


        # read all 3D cubes
        self.cubes = np.zeros((length, 32, 32, 32), dtype=np.float32)
        f_df = '/mnt/ceph/users/fvillaescusa/Ana/Gaussian_density_fields_3D/data/Gaussian_df_3D.npy'
        df = np.load(f_df)
        df = np.log(1+df)	#normalize
        count = 0 
        for i in range(length):
            self.cubes[count] = df[offset+i]
            count += 1
        min_df = np.min(self.cubes)
        max_df = np.max(self.cubes)
        self.cubes = (self.cubes - min_df)/(max_df - min_df) #normalize the cubes
        self.cubes = torch.tensor(self.cubes)
        self.len = length
     
        
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

'''
#using a data splitter
datasetsize=length
lengths = [int(datasetsize*0.8), int(datasetsize*0.1), int(datasetsize*0.1)]
torch.manual_seed(datasetsize)
print('splitting the dataset...')
           
train, validation, test = torch.utils.data.random_split(Data_set(offset = offset, length=length, parameters=parameters), lengths)

print('saving the datasets...')

torch.save(train, 'May/data/3d_test_3000_trainset.pt') #remember to rename as necessary
torch.save(validation, 'May/data/3d_test_3000_validset.pt') #remember to rename as necessary
torch.save(test, 'May/data/3d_test_3000_testset.pt')  #remember to rename as necessary


'''
#LOAD PREVIOUSLY SAVED data sets if applicable
print('loading datasets...')
train = torch.load('May/data/3d_test_3000_trainset.pt')
print("train")
validation = torch.load('May/data/3d_test_3000_validset.pt')
print("valid")
test = torch.load('May/data/3d_test_3000_testset.pt')
print("test")


#TRAIN THE MODEL

print('preparing to train...')

batch_size   = 16						#size of batches
epochs       = 1000   	 					#number of epochs
lr           = 1e-4						#learning rate
weight_decay = 1e-6   						#weight decay
date         = '06_23' 						#date for saving

fout   = 'May/losses/3d_test_losses_12cnn_3fcl_%dn_%s.txt'%(neurons, date)    #losses for plotting 	RENAME
fmodel = 'May/best_models/3d_test_12cnn_3fcl_%dn_%s.pt'%(neurons,date)   	#best model		RENAME 
num_workers = 1

#cosmo_dict = {0:'dOm', 1:'dOb', 2:'dh', 3:'dns', 4:'ds8'}	#dictionary to hold parameter names
cosmo_dict = {0:'A'}

print('preparing dataset loaders')
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True, num_workers=num_workers)

num_train_batches = len(train_loader)
num_valid_batches = len(valid_loader)

#If output file exists, delete it
if os.path.exists(fout):  os.system('rm %s'%fout)
#if os.path.exists(fmodel):  os.system('rm %s'%fmodel)
# define model, loss and optimizer 
print('Initializing...')
model = CNN1(params=len(parameters)) 				#model
#print('There are ', torch.cuda.device_count(), 'GPUs')
#model = nn.DataParallel(model)					#parallelize
model.to(device=device)

if os.path.exists(fmodel):  
    print("Loading model : Are we sure we want to load now ?")
    model.load_state_dict(torch.load(fmodel))
    model.to(device=device)
    #print(model.state_dict())

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)	#optimizer
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=35, 
							verbose=True, eps=1e-7) #reduce learning rate

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
        for i in range(batch_size):
        	flip = np.random.choice(2, 1)
        	if flip ==0:
        		x_flip = np.fliplr(x_train[i,0,:,:])
        		#print('flipped', x_flip)
        	else:
        		x_flip = x_train[i,0,:,:]
        		#print('not flipped', x_flip)
        	choice = np.random.choice(total_choice, num_rot)
        	rot = np.ascontiguousarray(random_rotation(x_flip,choice))
        	x_train[i,0,:,:] = torch.from_numpy(rot)
        x_train=x_train.to(device)
        y_train=y_train.to(device)
        #print('choice', choice)
        #print('augmented size', x_train.shape)


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
'''
#TEST THE MODEL

test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, 
                                          shuffle=True, num_workers=num_workers)

test_results   = 'May/predictions/results_3d_test_12cnn_1000n_06_23.txt'  		#RENAME

cp = np.load('/mnt/ceph/users/fvillaescusa/Ana/Gaussian_density_fields_3D/data/A_values_3D.npy')
mean, std = [np.mean(cp)], [np.std(cp)]



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
test_error = torch.zeros(len(parameters))
for x_test, y_test in test_loader:
    with torch.no_grad():
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        y_pred3 = model(x_test)
        length = len(x_test)
        results[offset:offset+length, 0:num_params]            = mean + y_test.cpu().numpy()*std
        results[offset:offset+length, num_params:2*num_params] = mean + y_pred3.cpu().numpy()*std
        offset += length
        test_error += torch.sum((y_pred3.cpu() - y_test.cpu())**2, dim=0)

test_error /= len(test)   
np.savetxt(test_results, results)			#save prediction results

for i,j in enumerate(parameters):
    print('%s = %.3f'%(cosmo_dict[j], test_error[i]))		#print test error on paramaters
    
print('results saved as', test_results)
'''
