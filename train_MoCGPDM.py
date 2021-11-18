import torch
import numpy as np
import time
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.distributions.normal import Normal
import pickle
from MoGPDM import MoGPDM
import matplotlib.animation as animation
import random
import argparse
from os import walk


	# set hyperparameters

seed = 100
num_data = 10          # set the number of sequences
deg = 30               # set which folder you want to use
d = 5                  # set the size of the latent space


save_folder = 'model/'

#  	only if you want to save some videos of the performance

#	Writer = animation.writers['ffmpeg']
#	writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)


dtype=torch.float64
device=torch.device('cpu')
Y_names = []
U_names = []


# load observation data names
folder = 'DATA/CTRL/8x8_rng_swing_'+str(deg)+'_deg/'
print('\n ### DATA FOLDER: '+folder)

for i in range(41):
    Y_names.append(folder+'state_samples_rng_swing_'+str(i)+'.csv')
    U_names.append(folder+'input_samples_rng_swing_'+str(i)+'.csv')



    # define the model 

D = 192
u_dim = 6
param_dict = {}
param_dict['D'] = D
param_dict['d'] = d
param_dict['u_dim'] = u_dim
param_dict['N_max'] = 450
model = MoGPDM(**param_dict)

	# set some pre-trained weights for a better training performance (you may need other weights for different set of data)

model.map_cluster_kernel_length = torch.nn.Parameter(torch.tensor([27.7606, 23.0840, 21.5175, 20.5224, 20.5270] ,dtype=model.dtype, device=model.device))
model.latent_cluster_kernel_length = torch.nn.Parameter(torch.tensor([73.0439, 44.8699, 49.7291, 34.5981, 35.1434, 72.5992, 45.6354, 50.0329,
        34.9401, 35.0878, 20.0000, 20.1708, 20.0671, 20.0000, 20.1708, 20.0671],dtype=model.dtype, device=model.device))

    # set the training set

training_set = np.random.choice(len(Y_names), size=num_data, replace=False)
print('training_set is ', training_set)
np.savetxt(save_folder+'training_set.csv', np.array(training_set), delimiter=',')

	# read from

training_set = np.genfromtxt(save_folder+'training_set.csv', delimiter=',').astype(int)
for i in training_set.astype(int):
    Y_data = np.loadtxt(Y_names[i], delimiter = ',')
    U_data = np.loadtxt(U_names[i], delimiter = ',')
    n = Y_data.shape[0]

    	# 	comment the following line if you want to train it with the full size
        #   instead of the reduced(ore concentrated) version.

    Y_data = (Y_data.reshape(n,64,3)-Y_data[:,(189, 190, 191)].reshape(n, 1, 3)).reshape(n,192)

    model.add_data(Y_data,U_data)


# init model makes a few trainings to search for stable configurations (sometimes it finds singular kernels)

keep_trying = True
init_loss = 0
while keep_trying:
    try:
        model.init_X()
        model.init_clusters('KMeans','Trajectories')
        model.init_param()
        model.init_means()
        Y = torch.tensor(model.get_Y(), dtype = model.dtype, device = model.device)
        N = len(Y)
        init_loss = model.gpdm_loss(Y,N,1).detach().numpy()
        loss = model.train_experts(num_opt_steps=100, num_print_steps=20, lr=0.01, balance=1, train_X = False, optimizer_type = 'adam')
        model.train_Gates( num_opt_steps = 2, num_print_steps = 1, lr=0.001, balance=100)
        keep_trying = False
    except:
        pass

    # 	start the training

loss_tracking = [init_loss]
Y = torch.tensor(model.get_Y(), dtype = model.dtype, device = model.device)
N = len(Y)
loss_tracking.append(model.gpdm_loss(Y,N,1).detach().numpy())
print(loss_tracking)
np.savetxt(save_folder+'loss_tracking.csv', np.array(loss_tracking), delimiter=',')
model.init_K_inv()
model.init_means()
model.train_gibbs_update(num_opt_steps = 1, num_print_steps = 1, update_latent = False, update = 'likelihood')
model.save(save_folder)
model.init_means()
loss_tracking.append(model.gpdm_loss(Y,N,1).detach().numpy())
print(loss_tracking)

		# 	training loop
for i in range(15):
    print('i = ', i)
    try:
    		#	train with the adam optimizer
        loss = model.train_experts(num_opt_steps = 50, num_print_steps=10, lr=0.01, balance=1, train_X = True, optimizer_type = 'adam')
        loss_tracking.append(model.gpdm_loss(Y,N,1).detach().numpy())
        print(loss_tracking)
        model.train_gibbs_update(num_opt_steps = 1, num_print_steps = 1, update_latent = False, update = 'likelihood')
        model.train_Gates( num_opt_steps = 2, num_print_steps = 1, lr=0.001, balance=100)
        model.save(save_folder)
        loss_tracking.append(model.gpdm_loss(Y,N,1).detach().numpy())
        print(loss_tracking)
        np.savetxt(save_folder+'loss_tracking.csv', np.array(loss_tracking), delimiter=',')
        model.init_means()
    except:
            #   failed to train, reload model
            #   load log_dict
        log_dict = pickle.load(open(save_folder+'log_dict.pt', 'rb' ) )
        # init GPDM object
        D = log_dict['D']
        d = log_dict['d']
        param_dict = {}
        param_dict['D'] = D
        param_dict['d'] = d
        param_dict['u_dim'] = log_dict['u_dim']
        param_dict['N_max'] = log_dict['N_max']
        param_dict['sigma_n_num_Y'] = log_dict['sigma_n_num_Y']
        param_dict['sigma_n_num_X'] = log_dict['sigma_n_num_X']
        model = MoGPDM(**param_dict)
        # load observation and control data
        model.observations_list = log_dict['observations_list']
        model.controls_list = log_dict['controls_list']
        model.keys_map_clusters = log_dict['keys_map_clusters']
        model.map_cluster_idx = log_dict['map_cluster_idx']
        model.keys_latent_clusters = log_dict['keys_latent_clusters']
        model.latent_cluster_idx = log_dict['latent_cluster_idx']
        model.latent_cluster_kernel_length = log_dict['latent_cluster_kernel_length']
        model.map_cluster_kernel_length = log_dict['map_cluster_kernel_length']
        # init model
        model.init_X()
        model.init_param()
        # load parameters
        state_dict = torch.load(save_folder+'Mogpdm.pt')
        model.load_state_dict(state_dict)
        model.init_K_inv()
        model.init_means()
        model.train_gibbs_update(num_opt_steps = 2, num_print_steps = 1, update_latent = False)
        model.save(save_folder)
        model.init_means()



loss = model.train_experts(num_opt_steps=5, num_print_steps=1, lr=0.0001, balance=1, train_X = False, optimizer_type = 'adam')
loss_tracking.append(model.gpdm_loss(Y,N,1).detach().numpy())
print(loss_tracking)
np.savetxt(save_folder+'loss_tracking.csv', np.array(loss_tracking), delimiter=',')
model.train_Gates( num_opt_steps = 3, num_print_steps = 1, lr=0.001, balance=100)
model.save(save_folder)
model.init_K_inv()
model.init_means()

plt.plot(loss_tracking)
plt.ylabel( 'log-likelihood during training')
plt.show()




