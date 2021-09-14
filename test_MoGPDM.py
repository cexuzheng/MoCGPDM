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


		 # load a model from save_folder


# load log_dict
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




#		load test data

deg = 30

folder = 'DATA/CTRL/8x8_rng_swing_'+str(deg)+'_deg/'
Y_list_1 = []
U_list_1 = []
for i in range(41,51):
    Y_data = np.loadtxt(folder+'state_samples_rng_swing_'+str(i)+'.csv', delimiter=',')
    n = Y_data.shape[0]
    #	only if the dimension was reduced during training
    Y_data = (Y_data.reshape(n,64,3)-Y_data[:,(189, 190, 191)].reshape(n, 1, 3)).reshape(n,192)
    Y_list_1.append(Y_data)
    U_list_1.append(np.loadtxt(folder+'input_samples_rng_swing_'+str(i)+'.csv', delimiter=','))




n_t = len(U_list_1)
errors = np.zeros((n_t,Y_list_1[0].shape[0]))
model = model

for which in range(n_t):
	#	 find the neares latent variable
    index = np.argmin( np.linalg.norm(model.Y-Y_list_1[which][0], axis = 1) )
    X0 = model.X[index]
    index = np.argmin( np.linalg.norm(model.Y-Y_list_1[which][1], axis = 1) )
    X1 = model.X[index]
    #	rollout 
    X_hat3, Y_hat3, time_model3 = model.rollout(U_list_1[which].shape[0], torch.tensor(U_list_1[which]), X0, X1 = X1  )
    #	compute the error
    Y_hat3 = Y_hat3.cpu().detach().numpy()
    Yhat3 = np.array(Y_hat3).reshape(len(Y_hat3),64,3)
    uu = np.cumsum(U_list_1[which][:,:3], axis = 0)
    Yhat3 = (Yhat3 + uu.reshape(len(Yhat3),1,3)).reshape(len(Y_hat3),192)
    errors[which, :] = 100*np.linalg.norm(Y_hat3-Y_list_1[which], axis = 1)/np.linalg.norm(Y_list_1[which], axis = 1)


#	chech if there are some failed predictions
idx = np.where( np.isnan(errors) )
for a in range(idx[0].shape[0]):
    errors[idx[0][a], idx[1][a]] = errors[idx[0][a], idx[1][a]-1]

errors_1 = np.copy(errors)
mean_errors_1 = np.mean(errors_1, axis = 0)
std_1 = np.sqrt(np.sum( (mean_errors_1-errors_1)**2, axis = 0)/(Y_list_1[0].shape[0]))
m_1 = np.mean(errors_1, axis = 1)
s_1 = np.sqrt(np.sum( (np.mean(m_1)-errors_1)**2)/(len(Y_list_1[0])*Y_list_1[0].shape[0]))
    

#		show performance

xx = np.arange(X_list_1[0].shape[0])

plt.figure(1)
plt.errorbar(xx,mean_errors_1, yerr = std_1, fmt = 'or', linewidth = 1, ms = 2)
plt.ylabel('error %')
plt.xlabel('time step')
plt.ylim([0,53])
plt.show()


plt.figure(1)
plt.errorbar([0],np.mean(m_1), yerr = 1.96*s_1, fmt = 'or', linewidth = 2, ms = 4)
#plt.legend(['0.1', '1', '10','100'])
plt.plot([0], np.mean(m_1)+1.96*s_1, '_r')
plt.plot([0], np.mean(m_1)-1.96*s_1, '_r')
plt.xlim([-1, 1])
plt.ylim([0,53])
plt.grid(axis = 'y')
plt.show()


#		show video


def update_line_with_test(i, Yhat1, line1, Ytest1, line_test1):
    line1.set_data(np.array([Yhat1[i,:,0],Yhat1[i,:,1]]))
    line1.set_3d_properties(Yhat1[i,:,2])
    line_test1.set_data(np.array([Ytest1[i,:,0],Ytest1[i,:,1]]))
    line_test1.set_3d_properties(Ytest1[i,:,2])
    return line1



which = np.random.choice(len(model.controls_list ) )
###
X_test_list1 = model.get_latent_sequences()
U_list1 = model.controls_list
X_hat1, Y_hat1, time_model2 = model.rollout(U_list1[which].shape[0], torch.tensor(U_list1[which]), torch.tensor(X_test_list1[which][0,:]), X1 = torch.tensor(X_test_list1[which][1,:] )  )
X_hat1 = X_hat1.cpu().detach().numpy()



fig = plt.figure(1, figsize = (16,9))

Yhat1 = np.array(Y_hat1).reshape(len(Y_hat1),64,3)
Yhat1 = Yhat1 + uu.reshape(len(Yhat1),1,3)
ax1 = fig.add_subplot(projection="3d")
ax1.set_xlim3d([-1.5, 1.5])
ax1.set_xlabel('X')
ax1.set_ylim3d([-1.5,1.5])
ax1.set_ylabel('Y')
ax1.set_zlim3d([-1.5, 1.5])
ax1.set_zlabel('Z')
ax1.set_title(' balance = 0.1 ')
line1 = ax1.plot(Yhat1[0,:,0],Yhat1[0,:,1],Yhat1[0,:,2],'bo', ms = 2)[0]
Ytest1 = np.array(model.observations_list[which]).reshape(len(Y_hat1), 64, 3) + uu.reshape(len(Y_hat1),1,3)
line_test1 = ax1.plot(Ytest1[0,:,0],Ytest1[0,:,1],Ytest1[0,:,2],'ro', ms = 1)[0]
plt.legend(['predicted', 'reference'])

line_ani = animation.FuncAnimation(
    fig, update_line_with_test, len(Yhat1), fargs=(Yhat1, line1, Ytest1, line_test1), interval=10,save_count = 1000)

try:
	line_ani.save('images/compare_i_'+str(which)+'rollout_.mp4', writer=writer)
except:
	pass

plt.show()




