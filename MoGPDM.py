
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch.distributions.normal import Normal
import pickle
import sys
from decimal import Decimal


class MoGPDM(torch.nn.Module):
    """
    Gaussian Process Dynamical Model
    Attributes
    ----------
    dtype : torch.dtype
        data type of torch tensors
    device : torch.device
        device on which a tensors will be allocated
    D : int
        observation space dimension
    d : int
        desired latent space dimension
    N_max : int
        maximum number of elements per cluster
    y_log_lengthscales_dict : torch.nn.Parameter
        log(lengthscales) of Y GP kernel
    y_log_lambdas_dict : torch.nn.Parameter
        log(signal inverse std) of Y GP kernel
    y_log_sigma_n_dict : torch.nn.Parameter
        log(noise std) of Y GP kernel
    x_log_lengthscales_dict : torch.nn.Parameter
        log(lengthscales) of X GP kernel
    x_log_lambdas_dict : torch.nn.Parameter
        log(signal inverse std) of X GP kernel
    x_log_sigma_n_dict : torch.nn.Parameter
        log(noise std) of X GP kernel
    x_log_lin_coeff_dict : torch.nn.Parameter
        log(linear coefficients) of X GP kernel
    X : torch.nn.Parameter
        latent states
    sigma_n_num_X : double
        additional noise std for numerical issues in X GP
    sigma_n_num_Y : double
        additional noise std for numerical issues in Y GP
    observations_list : list(double)
        list of observation sequences
    controls_list : list
        list of observed control actions
    u_dim : int
        control input dimension
    keys_map_clusters : set
        number of GP_Map_Experts
    map_cluster_idx : ndarray
        ndarray with the cluster index of all the mapping pairs
    keys_latent_clusters : set
        number of GP_latent_Experts
    latent_cluster_idx : ndarray
        ndarray with the cluster index of all the latent pairs
    """
    def __init__(self, D, d, u_dim, N_max, alpha = 1/100, sigma_n_num_Y=0., sigma_n_num_X=0.,
                dtype=torch.float64, device=torch.device('cpu')):
        """
        Parameters
        ----------
        D : int
            observation space dimension
        d : int
            latent space dimension
        N_max : int
            number of maximum elements per Expert
        sigma_n_num_Y : double (optional)
            additional noise std for numerical issues in X GP
        sigma_n_num_X : double (optional)
            additional noise std for numerical issues in X GP
        dtype: torch.dtype (optional)
            data type of torch tensors
        device: torch.device (optional)
            device on which a tensors will be allocated
        """
        super().__init__()

        # torch parameters
        self.dtype = dtype
        self.device = device
        # observation dimension
        self.D = D
        # desired latent dimension
        self.d = d
        # maximum number of points per cluster
        self.N_max = N_max
        # Set Y-kernel parameters
        self.y_log_lengthscales_dict = torch.nn.ParameterDict({})
        self.y_log_lambdas_dict = torch.nn.ParameterDict({})
        self.y_log_sigma_n_dict = torch.nn.ParameterDict({})
        # Set X-kernel parameters
        self.x_log_lengthscales_dict = torch.nn.ParameterDict({})
        self.x_log_lambdas_dict = torch.nn.ParameterDict({})
        self.x_log_sigma_n_dict = torch.nn.ParameterDict({})
        self.x_log_lin_coeff_dict = torch.nn.ParameterDict({})
        # additional noise variance for numerical issues
        self.sigma_n_num_Y = sigma_n_num_Y
        self.sigma_n_num_X = sigma_n_num_X
        # inti cluster parameters
        self.alpha = alpha
        self.map_cluster_kernel_length = torch.nn.Parameter(torch.ones(d,dtype=self.dtype, device=self.device)*10)
        self.latent_cluster_kernel_length = torch.nn.Parameter(torch.ones(2*d+u_dim,dtype=self.dtype, device=self.device)*10)
        # Initialize observations
        self.observations_list = []
        self.num_sequences = 0
        # control input dimension
        self.u_dim = u_dim
        self.controls_list = []

    def set_evaluation_mode(self):
        """
        Set the model in evaluation mode
        """
        for p in self.parameters():
            p.requires_grad = False
    

    def set_training_mode(self):
        """
        Set the model in training mode
        """
        for p in self.parameters():
            p.requires_grad = True


    def add_data(self, Y, U):
        """
        Add observation  and control data to self.observations_list
        Parameters
        ----------
        Y : double
            observation data (dimension: N x D)
        U : double
            observation data (dimension: N x u_dim)
        """
        if Y.shape[1]!=self.D:
            raise ValueError('Y must be a N x D matrix collecting observation data!')
        if U.shape[1]!=self.u_dim:
            raise ValueError('U must be a N x u_dim matrix collecting observation data!')
        if U.shape[0]!=Y.shape[0]:
            raise ValueError('Y and U must have the same number N of data!')
        self.observations_list.append(Y)
        self.controls_list.append(U)
        self.num_sequences = self.num_sequences+1
        print('Num. of sequences = '+str(self.num_sequences)+' [Data points = '+str(np.concatenate(self.observations_list, 0).shape[0])+']')

    def get_y_kernel(self, X1, X2, j, flg_noise=True):
        """
        Compute the latent mapping kernel (GP Y)
        Parameters
        ----------
        X1 : tensor(dtype)
            1st GP input points
        X2 : tensor(dtype)
            2nd GP input points
        j : string
            key for the corresponding expert
        flg_noise : boolean (optional)
            add noise to kernel matrix
        Return
        ------
        K_y(X1,X2)
        """
        return self.get_rbf_kernel(X1, X2, self.y_log_lengthscales_dict[str(j)], self.y_log_sigma_n_dict[str(j)], self.sigma_n_num_Y, flg_noise)


    def get_u_y_kernel(self, X1, X2, z, j):
        u =  torch.exp(-self.get_weighted_distances(X1, X2, self.y_log_lengthscales_dict[str(z)]))
        u[j] += torch.exp(self.y_log_sigma_n_dict[str(z)])**2 + self.sigma_n_num_Y**2
        return u.cpu().detach().numpy()


    def get_x_kernel(self, X1, X2, j, flg_noise=True):
        """
        Compute the latent dynamic kernel (GP X)
        Parameters
        ----------
        X1 : tensor(dtype)
            1st GP input points
        X2 : tensor(dtype)
            2nd GP input points
        j : string or integer
            key for the corresponding expert
        flg_noise : boolean (optional)
            add noise to kernel matrix
        Return
        ------
        K_x(X1,X2)
        """
        return self.get_rbf_kernel(X1, X2, self.x_log_lengthscales_dict[str(j)], self.x_log_sigma_n_dict[str(j)], self.sigma_n_num_X, flg_noise) + \
               self.get_lin_kernel(X1, X2, self.x_log_lin_coeff_dict[str(j)])

    def get_u_x_kernel(self, X1, X2, z, j):
        u =  torch.exp(-self.get_weighted_distances(X1, X2, self.x_log_lengthscales_dict[str(z)])) + \
            self.get_lin_kernel(X1, X2, self.x_log_lin_coeff_dict[str(z)])
        u[j] += torch.exp(self.x_log_sigma_n_dict[str(z)])**2 + self.sigma_n_num_X**2
        return u.cpu().detach().numpy()


    def get_rbf_kernel(self, X1, X2, log_lengthscales_par, log_sigma_n_par, sigma_n_num=0, flg_noise=True):
        """
        Compute RBF kernel on X1, X2 points (without considering signal variance)
        Parameters
        ----------
        X1 : tensor(dtype)
            1st GP input points
        X2 : tensor(dtype)
            2nd GP input points
        log_lengthscales_par : tensor(dtype)
            log(lengthscales) RBF kernel
        log_sigma_n_par : tensor(dtype)
            log(noise std)  RBF kernel
        sigma_n_num : double
            additional noise std for numerical issues
        flg_noise : boolean (optional)
            add noise to kernel matrix
        Return
        ------
        K_rbf(X1,X2)
        """
        if flg_noise:
            N = X1.shape[0]
            return torch.exp(-self.get_weighted_distances(X1, X2, log_lengthscales_par)) + \
                torch.exp(log_sigma_n_par)**2*torch.eye(N, dtype=self.dtype, device=self.device) + sigma_n_num**2*torch.eye(N, dtype=self.dtype, device=self.device)
        else:
            return torch.exp(-self.get_weighted_distances(X1, X2, log_lengthscales_par))

    def get_weighted_distances(self, X1, X2, log_lengthscales_par):
        """
        Computes (X1-X2)^T*Sigma^-2*(X1-X2) where Sigma = diag(exp(log_lengthscales_par))
        Parameters
        ----------
        X1 : tensor(dtype)
            1st GP input points
        X2 : tensor(dtype)
            2nd GP input points
        log_lengthscales_par : tensor(dtype)
            log(lengthscales)
        Return
        ------
        dist = (X1-X2)^T*Sigma^-2*(X1-X2)
        """
        lengthscales = torch.exp(log_lengthscales_par)
        X1_sliced = X1/lengthscales
        X1_squared = torch.sum(X1_sliced.mul(X1_sliced), dim=1, keepdim=True)
        X2_sliced = X2/lengthscales
        X2_squared = torch.sum(X2_sliced.mul(X2_sliced), dim=1, keepdim=True)
        dist = X1_squared + X2_squared.transpose(dim0=0, dim1=1) -2*torch.matmul(X1_sliced,X2_sliced.transpose(dim0=0, dim1=1))
        return dist


    def get_lin_kernel(self, X1, X2, log_lin_coeff_par):
        """
        Computes linear kernel on X1, X2 points: [X1,1]^T*Sigma*[X2,1] where Sigma=diag(exp(log_lin_coeff_par))
        Parameters
        ----------
        X1 : tensor(dtype)
            1st GP input points
        X2 : tensor(dtype)
            2nd GP input points
        log_lin_coeff_par : tensor(dtype)
            log(linear coefficients)
        Return
        ------
        K_lin(X1,X2)
        """
        Sigma = torch.diag(torch.exp(log_lin_coeff_par)**2)
        X1 = torch.cat([X1,torch.ones(X1.shape[0],1, dtype=self.dtype, device=self.device)],1)
        X2 = torch.cat([X2,torch.ones(X2.shape[0],1, dtype=self.dtype, device=self.device)],1)
        return torch.matmul(X1, torch.matmul(Sigma, X2.transpose(0,1)))


    def get_y_neg_log_likelihood(self, Y, X, j):
        """
        Compute latent negative log-likelihood Ly
        Parameters
        ----------
        Y : tensor(dtype)
            observation matrix
        X : tensor(dtype)
            latent state matrix
        Return
        ------
        L_y = D/2*log(|K_y(X,X)|) + 1/2*trace(K_y^-1*Y*W_y^2*Y) - N*log(|W_y|)
        """
        K_y = self.get_y_kernel(X,X, j)
        U = torch.cholesky(K_y, upper=True)
        U_inv = torch.inverse(U)
        K_y_inv = torch.matmul(U_inv,U_inv.t())
        log_det_K_y = 2*torch.sum(torch.log(torch.diag(U)))
        W = torch.diag(torch.exp(self.y_log_lambdas_dict[str(j)]))
        W2 = torch.diag(torch.exp(self.y_log_lambdas_dict[str(j)])**2)
        log_det_W = 2*torch.sum(self.y_log_lambdas_dict[str(j)])
        return self.D/2*log_det_K_y + 1/2*torch.trace(torch.chain_matmul(K_y_inv,Y,W2,Y.transpose(0,1))) - X.shape[0]*log_det_W

    def get_x_neg_log_likelihood(self, Xout, Xin, j):
        """
        Compute dynamics map negative log-likelihood Lx
        Parameters
        ----------
        Xout : tensor(dtype)
            dynamics map output matrix
        Xin : tensor(dtype)
            dynamics map input matrix
        Return
        ------
        L_x = d/2*log(|K_x(Xin,Xin)|) + 1/2*trace(K_x^-1*Xout*W_x^2*Xout) - (N-dyn_back_step)*log(|W_x|)
        """
        K_x = self.get_x_kernel(Xin,Xin, j)
        U = torch.cholesky(K_x, upper=True)
        U_inv = torch.inverse(U)
        K_x_inv = torch.matmul(U_inv,U_inv.t())
        log_det_K_x = 2*torch.sum(torch.log(torch.diag(U)))
        W = torch.diag(torch.exp(self.x_log_lambdas_dict[str(j)]))
        W2 = torch.diag(torch.exp(self.x_log_lambdas_dict[str(j)])**2)
        log_det_W = 2*torch.sum(self.x_log_lambdas_dict[str(j)])
        return self.d/2*log_det_K_x + 1/2*torch.trace(torch.chain_matmul(K_x_inv,Xout,W2,Xout.transpose(0,1))) - Xin.shape[0]*log_det_W

    def get_Xin_Xout_matrices(self, U=None, X=None):
        """
        Compute GP input and output matrices (Xin, Xout) for GP X
        Parameters
        ----------
        U : tensor(dtype) (optional)
            control input matrix
        X : tensor(dtype) (optional)
            latent state matrix
        Return
        ------
        Xin : GP X input matrix
        Xout : GP X output matrix
        start_indeces : list of sequences' start indeces
        """
        if X==None:
            X = self.X
        if U==None:
            U = self.controls_list
        X_list = []
        U_list = []
        x_start_index = 0
        start_indeces = []
        for j in range(len(self.observations_list)):
            sequence_length = self.observations_list[j].shape[0]
            X_list.append(X[x_start_index:x_start_index+sequence_length,:])
            U_list.append(torch.tensor(U[j], dtype=self.dtype, device=self.device))
            start_indeces.append(x_start_index)
            x_start_index = x_start_index+sequence_length

        Xin = torch.cat((X_list[0][1:-1,:], X_list[0][0:-2,:], U_list[0][1:-1,:]), 1)
        Xout = X_list[0][2:,:] - X_list[0][1:-1,:]
        for j in range(1,len(self.observations_list)):
            Xin = torch.cat((Xin, torch.cat((X_list[j][1:-1,:], X_list[j][0:-2,:], U_list[j][1:-1,:]), 1)),0)
            Xout = torch.cat((Xout, X_list[j][2:,:] - X_list[j][1:-1,:]),0)
        return Xin, Xout, start_indeces


    def init_X(self, cluster_type = 'KMeans'):
        """
        Initalize latent variables matrix with PCA
        """
        self.Y = self.get_Y()
        self.pca = PCA(n_components=self.d)
        X0 = self.pca.fit_transform(self.Y)
        # X0 = np.matmul(X0,(np.diag(1/np.sqrt(pca.singular_values_))))
        # set latent variables as parameters
        self.X = torch.nn.Parameter(torch.tensor(X0, dtype=self.dtype, device=self.device), requires_grad=True)

    def init_param(self):
        for z in self.keys_map_clusters:
            self.y_log_lengthscales_dict[str(z)] = torch.nn.Parameter(torch.rand(self.d,dtype=self.dtype, device=self.device)*0.5 + 0.5)
            self.y_log_lambdas_dict[str(z)] = torch.nn.Parameter(torch.rand(self.D,dtype=self.dtype, device=self.device)*0.5 + 0.5)
            self.y_log_sigma_n_dict[str(z)] = torch.nn.Parameter(torch.rand(1,dtype=self.dtype, device=self.device)*0.5 + 0.5)
        for z in self.keys_latent_clusters:
            self.x_log_lengthscales_dict[str(z)] = torch.nn.Parameter(torch.rand(self.d*2+self.u_dim,dtype=self.dtype, device=self.device)*0.5 + 0.5)
            self.x_log_lambdas_dict[str(z)] = torch.nn.Parameter(torch.rand(self.d,dtype=self.dtype, device=self.device)*0.5 + 0.5)
            self.x_log_sigma_n_dict[str(z)] = torch.nn.Parameter(torch.rand(1,dtype=self.dtype, device=self.device)*0.5 + 0.5)
            self.x_log_lin_coeff_dict[str(z)] = torch.nn.Parameter(torch.rand(self.d*2+self.u_dim+1,dtype=self.dtype, device=self.device)*0.5 + 0.5)

    def init_clusters(self, mapping_cluster_type = 'KMeans', rollout_cluster_type = 'Trajectories'):
        """
        Initialize the clusters of the mapp and the Xin
        Parameters
        ----------
        cluster_type : string
            can be ('KMeans', 'Trajectories', 'Random')
        """
        # mapping clusters
        N_y = self.X.shape[0]
        k_y = int(np.ceil(N_y/self.N_max))
        self.keys_map_clusters = set()
        self.map_cluster_idx = np.zeros(N_y)
        N_x = 0
        for i in range(len(self.observations_list)):
            N_x += self.observations_list[i].shape[0]-2
        k_x = int(np.ceil(N_x/self.N_max))
        self.keys_latent_clusters = set()
        self.latent_cluster_idx = np.zeros(N_x)
        if (mapping_cluster_type == 'KMeans'):
            # KMeans initialization
            kmeans = KMeans(n_clusters = k_y)
            kmeans.fit(self.X.cpu().detach().numpy())
            cluster_idx = 0
            for i in range(k_y):
                if len(np.where(kmeans.labels_ == i)[0])<=self.N_max:
                    # the cluster is smaller than the N_max condition
                    self.map_cluster_idx[np.where(kmeans.labels_ == i)] = cluster_idx
                    self.keys_map_clusters.add(cluster_idx)
                    #init expert parameters
                    self.y_log_lengthscales_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d,dtype=self.dtype, device=self.device))
                    self.y_log_lambdas_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.D,dtype=self.dtype, device=self.device))
                    self.y_log_sigma_n_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(1,dtype=self.dtype, device=self.device))
                    cluster_idx += 1
                else:
                    #the cluster must be splitted to reach that N_max condition
                    aux = np.where(kmeans.labels_ == i)[0]
                    np.random.shuffle(aux)
                    n = int(np.ceil(len(aux)/self.N_max))
                    for j in range(n):
                        self.map_cluster_idx[aux[j::n]] = cluster_idx
                        self.keys_map_clusters.add(cluster_idx)
                        #init expert parameters
                        self.y_log_lengthscales_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d,dtype=self.dtype, device=self.device))
                        self.y_log_lambdas_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.D,dtype=self.dtype, device=self.device))
                        self.y_log_sigma_n_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(1,dtype=self.dtype, device=self.device))
                        cluster_idx += 1
        elif (mapping_cluster_type == 'Trajectories'):
            cluster_idx = 0
            from_idx = 0
            to_idx = 0
            start_current_idx = 0
            currentN = 0
            for i in range(len(self.observations_list)):
                trajectory_N = self.observations_list[i].shape[0]
                if(currentN+trajectory_N > self.N_max):
                    # mapping expert
                    self.map_cluster_idx[start_current_idx:start_current_idx+currentN] = cluster_idx
                    self.keys_map_clusters.add(cluster_idx)
                    self.y_log_lengthscales_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d,dtype=self.dtype, device=self.device))
                    self.y_log_lambdas_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.D,dtype=self.dtype, device=self.device))
                    self.y_log_sigma_n_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(1,dtype=self.dtype, device=self.device))
                    #update indicators
                    start_current_idx += currentN
                    currentN = trajectory_N
                    from_idx = to_idx
                    to_idx = to_idx+trajectory_N-2
                    cluster_idx += 1
                elif( i == len(self.observations_list) -1):
                    # mapping expert
                    self.map_cluster_idx[start_current_idx:start_current_idx+currentN+trajectory_N] = cluster_idx
                    self.keys_map_clusters.add(cluster_idx)
                    self.y_log_lengthscales_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d,dtype=self.dtype, device=self.device))
                    self.y_log_lambdas_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.D,dtype=self.dtype, device=self.device))
                    self.y_log_sigma_n_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(1,dtype=self.dtype, device=self.device))
                else:
                    currentN += trajectory_N
                    to_idx += trajectory_N-2
        elif (mapping_cluster_type == 'Random'):
            aux = np.shuffle(np.arange(N_y))
            for i in range(k_y):
                self.map_cluster_idx[aux[i::k_y]] = i
                self.keys_map_clusters.add(i)
                #init expert parameters
                self.y_log_lengthscales_dict[str(i)] = torch.nn.Parameter(torch.ones(self.d,dtype=self.dtype, device=self.device))
                self.y_log_lambdas_dict[str(i)] = torch.nn.Parameter(torch.ones(self.D,dtype=self.dtype, device=self.device))
                self.y_log_sigma_n_dict[str(i)] = torch.nn.Parameter(torch.ones(1,dtype=self.dtype, device=self.device))
        else:
            print("mapping_cluster_type must be KMeans, Trajectories or Random")
        if(rollout_cluster_type == 'KMeans'):
            # KMeans initialization
            Xin,_,__ = self.get_Xin_Xout_matrices()
            kmeans = KMeans(n_clusters = k_x)
            kmeans.fit(Xin.cpu().detach().numpy())
            cluster_idx = 0
            for i in range(k_x):
                if len(np.where(kmeans.labels_ == i)[0])<=self.N_max:
                    # the cluster is smaller than the N_max condition
                    self.latent_cluster_idx[np.where(kmeans.labels_ == i)] = cluster_idx
                    self.keys_latent_clusters.add(cluster_idx)
                    #init expert parameters
                    self.x_log_lengthscales_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d*2+self.u_dim,dtype=self.dtype, device=self.device))
                    self.x_log_lambdas_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d,dtype=self.dtype, device=self.device))
                    self.x_log_sigma_n_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(1,dtype=self.dtype, device=self.device))
                    self.x_log_lin_coeff_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d*2+self.u_dim+1,dtype=self.dtype, device=self.device))
                    cluster_idx += 1
                else:
                    #the cluster must be splitted to reach that N_max condition
                    aux = np.where(kmeans.labels_ == i)[0]
                    np.random.shuffle(aux)
                    n = int(np.ceil(len(aux)/self.N_max))
                    for j in range(n):
                        self.latent_cluster_idx[aux[j::n]] = cluster_idx
                        self.keys_latent_clusters.add(cluster_idx)
                        #init expert parameters
                        self.x_log_lengthscales_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d*2+self.u_dim,dtype=self.dtype, device=self.device))
                        self.x_log_lambdas_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d,dtype=self.dtype, device=self.device))
                        self.x_log_sigma_n_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(1,dtype=self.dtype, device=self.device))
                        self.x_log_lin_coeff_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d*2+self.u_dim+1,dtype=self.dtype, device=self.device))
                        cluster_idx += 1
        elif(rollout_cluster_type == 'Trajectories'):
            cluster_idx = 0
            from_idx = 0
            to_idx = 0
            start_current_idx = 0
            currentN = 0
            for i in range(len(self.observations_list)):
                trajectory_N = self.observations_list[i].shape[0]
                if(currentN+trajectory_N > self.N_max):
                    # latent rollout expert
                    self.latent_cluster_idx[from_idx:to_idx] = cluster_idx
                    self.keys_latent_clusters.add(cluster_idx)
                    self.x_log_lengthscales_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d*2+self.u_dim,dtype=self.dtype, device=self.device))
                    self.x_log_lambdas_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d,dtype=self.dtype, device=self.device))
                    self.x_log_sigma_n_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(1,dtype=self.dtype, device=self.device))
                    self.x_log_lin_coeff_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d*2+self.u_dim+1,dtype=self.dtype, device=self.device))
                    #update indicators
                    start_current_idx += currentN
                    currentN = trajectory_N
                    from_idx = to_idx
                    to_idx = to_idx+trajectory_N-2
                    cluster_idx += 1
                elif( i == len(self.observations_list) -1):
                    # latent rollout expert
                    self.latent_cluster_idx[from_idx:to_idx+trajectory_N-2] = cluster_idx
                    self.keys_latent_clusters.add(cluster_idx)
                    self.x_log_lengthscales_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d*2+self.u_dim,dtype=self.dtype, device=self.device))
                    self.x_log_lambdas_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d,dtype=self.dtype, device=self.device))
                    self.x_log_sigma_n_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(1,dtype=self.dtype, device=self.device))
                    self.x_log_lin_coeff_dict[str(cluster_idx)] = torch.nn.Parameter(torch.ones(self.d*2+self.u_dim+1,dtype=self.dtype, device=self.device))
                else:
                    currentN += trajectory_N
                    to_idx += trajectory_N-2
        elif(rollout_cluster_type == 'Random'):
            aux = np.shuffle(np.arange(N_x))
            for i in range(k_x):
                self.latent_cluster_idx[aux[i::k_x]] = i
                self.keys_latent_clusters.add(i)
                #init expert parameters
                self.x_log_lengthscales_dict[str(i)] = torch.nn.Parameter(torch.ones(self.d*2+self.u_dim,dtype=self.dtype, device=self.device))
                self.x_log_lambdas_dict[str(i)] = torch.nn.Parameter(torch.ones(self.d,dtype=self.dtype, device=self.device))
                self.x_log_sigma_n_dict[str(i)] = torch.nn.Parameter(torch.ones(1,dtype=self.dtype, device=self.device))
                self.x_log_lin_coeff_dict[str(i)] = torch.nn.Parameter(torch.ones(self.d*2+self.u_dim+1,dtype=self.dtype, device=self.device))
        else:
            print("rollout_cluster_type must be KMeans, Trajectories or Random")
        print("starting computing K_inv")
        self.init_K_inv()

    def init_means(self):
        self.meanY = {}
        for z in self.keys_map_clusters:
            idx = np.where(self.map_cluster_idx == z)
            self.meanY[str(z)] = np.mean(self.Y[idx], axis = 0)
        self.meanX = {}
        _,Xout,__ = self.get_Xin_Xout_matrices()
        for z in self.keys_latent_clusters:
            idx = np.where(self.latent_cluster_idx == z)
            self.meanX[str(z)] = np.mean( Xout[idx].cpu().detach().numpy() , axis = 0)


    def init_K_inv(self):
        self.Ky_inv = {}
        self.log_det_K_y = {}
        for j in self.keys_map_clusters:
            idx = np.where(self.map_cluster_idx == j)
            Ky = self.get_y_kernel(self.X[idx],self.X[idx], j).cpu().detach().numpy()
            self.Ky_inv[str(j)] = np.linalg.inv(Ky)
            U = np.linalg.cholesky(Ky)
            self.log_det_K_y[str(j)] = 2*np.sum(np.log(np.diag(U)))
        self.Kx_inv = {}
        self.log_det_K_x = {}
        Xin,_,__ = self.get_Xin_Xout_matrices()
        for j in self.keys_latent_clusters:
            idx = np.where(self.latent_cluster_idx == j)
            Kx = self.get_x_kernel(Xin[idx],Xin[idx], j).cpu().detach().numpy()
            self.Kx_inv[str(j)] = np.linalg.inv(Kx)
            U = np.linalg.cholesky(Kx)
            self.log_det_K_x[str(j)] = 2*np.sum(np.log(np.diag(U)))


    def init_K_y(self, X, j):
        Ky = self.get_y_kernel(X,X,j).cpu().detach().numpy()
        self.Ky_inv[str(j)] = np.linalg.inv(Ky)
        U = np.linalg.cholesky(Ky)
        self.log_det_K_y[str(j)] = 2*np.sum(np.log(np.diag(U)))

    def init_K_x(self, Xin, j):
        Kx = self.get_x_kernel(Xin,Xin, j).cpu().detach().numpy()
        self.Kx_inv[str(j)] = np.linalg.inv(Kx)
        U = np.linalg.cholesky(Kx)
        self.log_det_K_x[str(j)] = 2*np.sum(np.log(np.diag(U)))


    def single_rank_matrix_inverse(self, A_inv, u, v):
        """
        Computes the inverse of (A+np.dot(u, v.T)) given the inverse of A
        Parameters
        ----------
        A_inv : square ndarray of 2 dimensions (n, n)
            the inverse of A
        u : ndarray of shape (1, n)
        v : ndarray of shape (1, n)
        """
        return A_inv-np.dot( np.dot(A_inv, u).reshape(len(u), 1) , np.dot(v.T, A_inv).reshape(1, len(v)))/(1+np.dot(v.T, np.dot(A_inv, u)))


    def submatrix_inverse(self, A_inv, i, u, j, experts = 'mapping'):
        """
        Computes the inverse of the submatrix of A if you take out the i-th row and column
        """
        u = -np.copy(u)
        if (len(u) == len(A_inv)-1):
            u = np.concatenate((u[:i],np.array([0]),u[i:] ))

        # u = u, v = (0 0 .. 1 .. 0 0)
        if experts == 'mapping':
            self.log_det_K_y[str(j)] += np.log(1+np.dot(A_inv[i,:], u.squeeze()))
        else:
            self.log_det_K_x[str(j)] += np.log(1+np.dot(A_inv[i,:], u.squeeze()))
        A_inv = A_inv-np.dot( np.dot(A_inv, u).reshape(len(u), 1) , A_inv[:,i].reshape(1, len(u)))/(1+ np.dot(A_inv, u)[i])

        # u = (0 0 .. 1 .. 0 0), v = u
        if experts == 'mapping':
            self.log_det_K_y[str(j)] += np.log(1+np.dot(u.squeeze(), A_inv[:,i]))
        else:
            self.log_det_K_x[str(j)] += np.log(1+np.dot(u.squeeze(), A_inv[:,i]))
        A_inv = A_inv-np.dot( A_inv[:,i].reshape(len(u), 1) , np.dot(u.T, A_inv).reshape(1, len(u)))/(1+np.dot(u.reshape(1,len(u)), A_inv[:,i].reshape(len(u),1)))

        if experts == 'mapping':
            self.log_det_K_y[str(j)] += np.log(A_inv[i,i])
        else:
            self.log_det_K_x[str(j)] += np.log(A_inv[i,i])
        return np.delete(np.delete(A_inv, i, axis = 0), i, axis = 1)


    def hypermatrix_inverse(self, B_inv, i, u, j, experts = 'mapping'):
        """
        Computes the inverse of a supermatrix assuming it is a covariance (ones in the diagonal)
        Parameters
        ----------
        B_inv : ndarray 2-dim matrix
            The inverse of the previous
        i : int
            The final position of the new row and column
        u : ndarray
            vector of the actual row that we are adding
        """
        ii_element = u[i].squeeze()
        u = np.copy(u)
        u[i] = 0
        aux = np.zeros( (len(B_inv)+1,len(B_inv)+1) )
        aux[:i,:i] = B_inv[:i,:i]
        aux[:i, i+1:] = B_inv[:i, i:]
        aux[i+1:,:i] = B_inv[i:, :i]
        aux[i+1:,i+1:] = B_inv[i:,i:]
        aux[i,i] = 1/ii_element
        if experts == 'mapping':
            self.log_det_K_y[str(j)] += np.log(ii_element)
        else:
            self.log_det_K_x[str(j)] += np.log(ii_element)

        # u = u, v = (0 0 ..1.. 0 0)
        if experts == 'mapping':
            self.log_det_K_y[str(j)] += np.log(1+np.dot(aux[i,:], u.squeeze()))
        else:
            self.log_det_K_x[str(j)] += np.log(1+np.dot(aux[i,:], u.squeeze()))
        A_inv = aux-np.dot( np.dot(aux, u).reshape(len(u), 1) , aux[i,:].reshape(1, len(u)))/(1+ np.dot(aux[i,:], u))

        # u = (0 0 ..1.. 0 0), v = u
        if experts == 'mapping':
            self.log_det_K_y[str(j)] += np.log(1+np.dot(u.squeeze(), A_inv[:,i]))
        else:
            self.log_det_K_x[str(j)] += np.log(1+np.dot(u.squeeze(), A_inv[:,i]))
        A_inv = A_inv-np.dot( A_inv[:,i].reshape(len(u), 1) , np.dot(u.reshape(1,len(u)), A_inv).reshape(1, len(u)))/(1+np.dot(u.reshape(1,len(u)), A_inv[:,i].reshape(len(u),1)))

        return A_inv


    def map_cluster_kernel(self, X1, X2):
        return torch.exp( -0.5*torch.matmul( (X1-X2)**2, self.map_cluster_kernel_length ) )

    def latent_cluster_kernel(self, X1, X2):
        return torch.exp( -0.5*torch.matmul( (X1-X2)**2, self.latent_cluster_kernel_length ) )

    def map_experts_weight(self, X):
        p = torch.zeros( len(self.keys_map_clusters) )
        K = self.map_cluster_kernel( X, self.X )
        i = 0
        for j in self.keys_map_clusters:
            idx = np.where(j == self.map_cluster_idx)
            p[i] = torch.sum( K[idx])
            i+=1
        return p/torch.sum(p).detach()

    def latent_experts_weight(self,X):
        p = torch.zeros( len(self.keys_latent_clusters) )
        Xin,_,__ = self.get_Xin_Xout_matrices()
        K = self.latent_cluster_kernel( X, Xin )
        i = 0
        for j in self.keys_latent_clusters:
            idx = np.where(j == self.latent_cluster_idx)
            p[i] = torch.sum( K[idx])
            i+=1
        return p/torch.sum(p).detach()

    def pred_x_to_y(self, X, Y, z):
        idx = np.where(self.map_cluster_idx == z)
        Ky_star = self.get_y_kernel(X.unsqueeze(0),self.X[idx], z, False).cpu().detach().numpy()
        mu = Ky_star.dot(self.Ky_inv[str(z)]).dot(self.Y[idx])+self.meanY[str(z)]
        sigma = np.abs( self.get_u_y_kernel(X.unsqueeze(0), X.unsqueeze(0), z = z, j = 0) - Ky_star.dot(self.Ky_inv[str(z)]).dot(Ky_star.T) )
        return mu, sigma*np.exp(self.y_log_lambdas_dict[str(z)].cpu().detach().numpy())**-2

    def map_cluster_likelihood(self, z, X, Y):
        idx = np.where(self.map_cluster_idx == z)
        Ky_star = self.get_y_kernel(X.unsqueeze(0),self.X[idx], z, False).cpu().detach().numpy()
        mu = Ky_star.dot(self.Ky_inv[str(z)]).dot(self.Y[idx]-self.meanY[str(z)]) + self.meanY[str(z)]
        sigma = np.abs( self.get_u_y_kernel(X.unsqueeze(0), X.unsqueeze(0), z = z, j = 0) - Ky_star.dot(self.Ky_inv[str(z)]).dot(Ky_star.T) )
        diag_sd = np.squeeze( np.sqrt(sigma*np.exp(self.y_log_lambdas_dict[str(z)].cpu().detach().numpy())**-2) )
        dec = Decimal(diag_sd[0]*2*np.pi)
        for i in range(1, self.D):
            dec = dec*Decimal(diag_sd[i]*2*np.pi)
        return Decimal(float(np.squeeze(np.exp(-1/2*np.dot( (Y-mu)**2 , diag_sd )))))/np.sqrt(dec)

    def latent_cluster_likelihood(self, z, X, Y):
        Xin,Xout,__ = self.get_Xin_Xout_matrices()
        idx = np.where(self.latent_cluster_idx == z)
        aux = np.squeeze(self.get_x_kernel(X.unsqueeze(0),Xin[idx], z, False).cpu().detach().numpy())
        mu = np.dot( aux, np.dot(self.Kx_inv[str(z)], Xout[idx].cpu().detach().numpy())-self.meanX[str(z)])+self.meanX[str(z)]
        sigma = np.abs( self.get_x_kernel(X.unsqueeze(0), X.unsqueeze(0), z, False).cpu().detach().numpy() - np.dot(aux, np.dot(self.Kx_inv[str(z)], aux)) )
        diag_var = np.squeeze(np.sqrt(sigma*np.exp(self.x_log_lambdas_dict[str(z)].cpu().detach().numpy())**-2))
        return np.exp(-1/2*np.dot( (Y.cpu().detach().numpy()-mu)**2 , diag_var ))/np.sqrt(np.prod(diag_var)*(2*np.pi)**self.d)



    def fast_y_log_likelihood(self):
        """
        Compute latent negative log-likelihood Ly
        Parameters
        ----------
        Y : tensor(dtype)
            observation matrix
        X : tensor(dtype)
            latent state matrix
        Return
        ------
        sum_j L_y = sum_j D/2*log(|K_y(X,X)|) + 1/2*trace(K_y^-1*Y*W_y^2*Y) - N*log(|W_y|)
        """
        Y = torch.tensor(self.get_Y(), dtype = self.dtype, device = self.device)
        lossY = torch.tensor(0).float()
        for j in self.keys_map_clusters:
            idx = np.where(self.map_cluster_idx == j)
            K_y_inv = torch.tensor(self.Ky_inv[str(j)])
            log_det_K_y = torch.tensor(self.log_det_K_y[str(j)])
            W = torch.diag(torch.exp(self.y_log_lambdas_dict[str(j)]))
            W2 = torch.diag(torch.exp(self.y_log_lambdas_dict[str(j)])**2)
            log_det_W = 2*torch.sum(self.y_log_lambdas_dict[str(j)])
            lossY += self.D/2*log_det_K_y + 1/2*torch.trace(torch.chain_matmul(K_y_inv,Y[idx],W2,Y[idx].transpose(0,1))) - idx[0].shape[0]*log_det_W

        return -lossY

    def fast_x_log_likelihood(self):
        """
        Compute dynamics map negative log-likelihood Lx
        Parameters
        ----------
        Xout : tensor(dtype)
            dynamics map output matrix
        Xin : tensor(dtype)
            dynamics map input matrix
        Return
        ------
        sum_j L_x = sum_j d/2*log(|K_x(Xin,Xin)|) + 1/2*trace(K_x^-1*Xout*W_x^2*Xout) - (N-dyn_back_step)*log(|W_x|)
        """

        Xin, Xout, start_indeces = self.get_Xin_Xout_matrices()
        lossX = torch.tensor(0, dtype = self.dtype, device = self.device)
        for j in self.keys_latent_clusters:
            idx = np.where(self.latent_cluster_idx == j)
            K_x_inv = torch.tensor(self.Kx_inv[str(j)])
            log_det_K_x = torch.tensor(self.log_det_K_x[str(j)])
            W = torch.diag(torch.exp(self.x_log_lambdas_dict[str(j)]))
            W2 = torch.diag(torch.exp(self.x_log_lambdas_dict[str(j)])**2)
            log_det_W = 2*torch.sum(self.x_log_lambdas_dict[str(j)])
            lossX += self.d/2*log_det_K_x + 1/2*torch.trace(torch.chain_matmul(K_x_inv,Xout[idx],W2,Xout[idx].transpose(0,1))) - idx[0].shape[0]*log_det_W

        return -lossX

    def Gibbs_update_map_clusters(self, update = 'likelihood'):
        N = len(self.X)
        indexes = np.arange(N)
        np.random.shuffle(indexes)
        for i in indexes:
            # map_clusters_update
            z = int(self.map_cluster_idx[i])
            idx = np.where(self.map_cluster_idx == z)
            j = np.where( idx[0] == i )[0][0]
            # erase from the cluster
            prev_Ky_inv = self.Ky_inv[str(z)]
            prev_mean = self.meanY[str(z)]
            prev_log_det_ky = self.log_det_K_y[str(z)]
            self.map_cluster_idx[i] = -1
            if(len(idx[0]) <= 1):
                # this cluster disapears
                try:
                    self.Ky_inv.pop(str(z),None)
                except:
                    self.Ky_inv.pop(str(z))
                try:
                    self.y_log_lengthscales_dict.pop(str(z),None)
                except:
                    self.y_log_lengthscales_dict.pop(str(z))
                try:
                    self.y_log_lambdas_dict.pop(str(z),None)
                except:
                    self.y_log_lambdas_dict.pop(str(z))
                try:
                    self.y_log_sigma_n_dict.pop(str(z),None)
                except:
                    self.y_log_sigma_n_dict.pop(str(z))
                self.keys_map_clusters.discard(z)
            else:
                self.Ky_inv[str(z)] = self.submatrix_inverse(np.copy(prev_Ky_inv), j, self.get_u_y_kernel(self.X[idx],self.X[i].unsqueeze(0), z, j), z, 'mapping')
                self.meanY[str(z)] = (prev_mean-self.Y[i]/len(idx[0]))*(len(idx[0])/(len(idx[0])-1))
                if( np.any(np.isnan(self.Ky_inv[str(z)])) or np.isnan(self.log_det_K_y[str(z)]) ):
                    idx = np.where(self.map_cluster_idx == z)
                    self.init_K_y(self.X[idx], z)

            # choose new cluster
            p = np.zeros(len(self.keys_map_clusters))
            keys = list(self.keys_map_clusters)
            K = self.map_cluster_kernel( self.X[i], self.X ).cpu().detach().numpy()
            if(update == 'loss'):
              for a in range(len(keys)):
                if(  len(np.where(self.map_cluster_idx == int(keys[a]))[0]) >= self.N_max ):
                    p[a] = 0
                else:
                    self.map_cluster_idx[i] = keys[a]
                    before_k = self.Ky_inv[str(keys[a])]
                    before_log_det = self.log_det_K_y[str(keys[a])]
                    if keys[a] == z:
                        self.Ky_inv[str(keys[a])] = prev_Ky_inv
                        self.log_det_K_y[str(keys[a])] = prev_log_det_ky
                        p[a] = self.fast_y_log_likelihood()
                    else:
                        idx = np.where(keys[a] == self.map_cluster_idx)
                        j = np.where( idx[0] == i )[0][0]
                        u = self.get_u_y_kernel(self.X[idx],self.X[i].unsqueeze(0), keys[a], j)
                        self.Ky_inv[str(keys[a])] = self.hypermatrix_inverse(np.copy(before_k), j, u, keys[a], experts = 'mapping')
                        p[a] = self.fast_y_log_likelihood()
                    self.map_cluster_idx[i] = -1
                    self.Ky_inv[str(keys[a])] = before_k 
                    self.log_det_K_y[str(keys[a])] = before_log_det 
            elif(update == 'likelihood'):
              for a in range(len(keys)):
                if(  len(np.where(self.map_cluster_idx == int(keys[a]))[0]) >= self.N_max ):
                    p[a] = 0
                else:
                    idx = np.where(keys[a] == self.map_cluster_idx)
                    p[a] = float(self.map_cluster_likelihood(keys[a], self.X[i], self.Y[i])*Decimal(np.sum( K[idx])))



            # probabilities of each cluster is proportional to p*pz
            # sample new cluster
            p = np.concatenate( (p,np.array(self.alpha/(self.alpha+N-1)).reshape(1)*(2*self.d+1)/self.D) )
            new_z = np.argmax( p )

            if(new_z == len(keys)):
                #creates new cluster
                new_z = max(self.keys_map_clusters)+1
                self.keys_map_clusters.add(new_z)
                self.map_cluster_idx[i] = new_z
                #init expert parameters
                self.y_log_lengthscales_dict[str(new_z)] = torch.nn.Parameter(sum(self.y_log_lengthscales_dict.values())/len(self.y_log_lengthscales_dict))
                self.y_log_lambdas_dict[str(new_z)] = torch.nn.Parameter(sum(self.y_log_lambdas_dict.values())/len(self.y_log_lambdas_dict))
                self.y_log_sigma_n_dict[str(new_z)] = torch.nn.Parameter(sum(self.y_log_sigma_n_dict.values())/len(self.y_log_sigma_n_dict))
                self.Ky_inv[str(new_z)] = np.array(1).reshape(1,1)
                self.meanY[str(new_z)] = self.Y[i]
                self.log_det_K_y[str(new_z)] = 0
            else:
                new_z = keys[new_z]
                self.map_cluster_idx[i] = new_z
                if(new_z == z):
                    self.Ky_inv[str(new_z)] = prev_Ky_inv
                    self.meanY[str(z)] = prev_mean
                    self.log_det_K_y[str(z)] = prev_log_det_ky 
                else:
                    idx = np.where(self.map_cluster_idx == new_z)
                    self.init_K_y(self.X[idx], new_z)
                    self.meanY[str(new_z)] = np.mean(self.Y[idx], axis = 0)
                    idx = np.where(self.map_cluster_idx == z)
                    if(idx[0].size > 0):
                        self.init_K_y(self.X[idx], z)
                        self.meanY[str(z)] = np.mean(self.Y[idx], axis = 0)
        
        # clear the small clusters and join them
        all_idx = np.array([])
        min_cluster = np.max(np.array(list(self.keys_map_clusters)))
        for z in list(self.keys_map_clusters):
            idx = np.where( z == self.map_cluster_idx )
            if len(idx[0]) <= 5:
                all_idx = np.concatenate( [all_idx, idx[0]] ).astype('int32')
                min_cluster = min( z, min_cluster )
                try:
                    self.Ky_inv.pop(str(z),None)
                except:
                    self.Ky_inv.pop(str(z))
                try:
                    self.y_log_lengthscales_dict.pop(str(z),None)
                except:
                    self.y_log_lengthscales_dict.pop(str(z))
                try:
                    self.y_log_lambdas_dict.pop(str(z),None)
                except:
                    self.y_log_lambdas_dict.pop(str(z))
                try:
                    self.y_log_sigma_n_dict.pop(str(z),None)
                except:
                    self.y_log_sigma_n_dict.pop(str(z))
                self.keys_map_clusters.discard(z)
        k_y = np.ceil( len(all_idx) / self.N_max).astype('int32')
        np.random.shuffle(all_idx)
        for i in range(k_y):
            while( min_cluster in self.map_cluster_idx ):
                min_cluster += 1
            idx = all_idx[i::k_y]
            self.map_cluster_idx[idx] = min_cluster
            self.keys_map_clusters.add(min_cluster)
            #init expert parameters
            try:
                self.y_log_lengthscales_dict[str(min_cluster)] = torch.nn.Parameter(sum(self.y_log_lengthscales_dict.values())/len(self.y_log_lengthscales_dict))
                self.y_log_lambdas_dict[str(min_cluster)] = torch.nn.Parameter(sum(self.y_log_lambdas_dict.values())/len(self.y_log_lambdas_dict))
                self.y_log_sigma_n_dict[str(min_cluster)] = torch.nn.Parameter(sum(self.y_log_sigma_n_dict.values())/len(self.y_log_sigma_n_dict))
            except:
                self.y_log_lengthscales_dict[str(min_cluster)] = torch.nn.Parameter(torch.rand(self.d,dtype=self.dtype, device=self.device)*0.5 + 0.5)
                self.y_log_lambdas_dict[str(min_cluster)] = torch.nn.Parameter(torch.rand(self.D,dtype=self.dtype, device=self.device)*0.5 + 0.5)
                self.y_log_sigma_n_dict[str(min_cluster)] = torch.nn.Parameter(torch.rand(1,dtype=self.dtype, device=self.device)*0.5 + 0.5)
            self.meanY[str(min_cluster)] = np.mean(self.Y[idx], axis = 0)

    def Gibbs_update_latent_clusters(self, update = 'likelihood'):
        Xin,Xout,__ = self.get_Xin_Xout_matrices()
        N = len(Xin)
        indexes = np.arange(N)
        np.random.shuffle(indexes)
        for i in indexes:
            # latent variables cluster update
            z = int(self.latent_cluster_idx[i])
            idx = np.where(self.latent_cluster_idx == z)
            j = np.where( idx[0] == i )[0][0]
            # erase from the cluster
            prev_Kx_inv = self.Kx_inv[str(z)]
            prev_mean = self.meanX[str(z)]
            prev_log_det_kx = self.log_det_K_x[str(z)]
            self.latent_cluster_idx[i] = -1
            if(len(idx[0]) <= 1):
                # this cluster disapears
                try:
                    self.Kx_inv.pop(str(z),None)
                except:
                    self.Kx_inv.pop(str(z))
                try:
                    self.x_log_lengthscales_dict.pop(str(z),None)
                except:
                    self.x_log_lengthscales_dict.pop(str(z))
                try:
                    self.x_log_lambdas_dict.pop(str(z),None)
                except:
                    self.x_log_lambdas_dict.pop(str(z))
                try:
                    self.x_log_sigma_n_dict.pop(str(z),None)
                except:
                    self.x_log_sigma_n_dict.pop(str(z))
                try:
                    self.x_log_lin_coeff_dict.pop(str(z),None)
                except:
                    self.x_log_lin_coeff_dict.pop(str(z))
                self.keys_latent_clusters.discard(z)
            else:
                # update Kx_inv
                #idx = np.where(self.latent_cluster_idx == z)
                #Kx = self.get_x_kernel(Xin[idx],Xin[idx], z).cpu().detach().numpy()
                #self.Kx_inv[str(z)] = np.linalg.inv(Kx)
                self.Kx_inv[str(z)] = self.submatrix_inverse(np.copy(prev_Kx_inv), j, np.squeeze(self.get_u_x_kernel(Xin[idx],Xin[i].unsqueeze(0), z, j)), z, 'latent')
                self.meanX[str(z)] = (prev_mean-Xout[i].detach().numpy()/len(idx[0]))*(  len(idx[0]) / (len(idx[0])-1)  )
                if( np.any(np.isnan(self.Kx_inv[str(z)])) or np.isnan(self.log_det_K_x[str(z)]) ):
                    idx = np.where(self.latent_cluster_idx == z)
                    Kx = self.get_x_kernel(Xin[idx],Xin[idx], z).cpu().detach().numpy()
                    self.Kx_inv[str(z)] = np.linalg.inv(Kx)

            # choose new cluster
            p = np.zeros(len(self.keys_latent_clusters))
            keys = list(self.keys_latent_clusters)
            K = self.latent_cluster_kernel( Xin[i], Xin ).cpu().detach().numpy()
            X = Xin[i]
            Y = Xout[i]
            for a in range(len(keys)):
                j = keys[a]
                if(  len(np.where(self.latent_cluster_idx == int(keys[a]))[0]) >= self.N_max ):
                    p[a] = 0
                else:
                    idx = np.where(self.latent_cluster_idx == j)
                    aux = np.squeeze(self.get_x_kernel(X.unsqueeze(0),Xin[idx], j, False).cpu().detach().numpy())
                    mu = np.dot( aux, np.dot(self.Kx_inv[str(j)], Xout[idx].cpu().detach().numpy()-self.meanX[str(j)])) + self.meanX[str(j)]
                    sigma = np.abs( self.get_u_x_kernel(X.unsqueeze(0), X.unsqueeze(0), z = j, j = 0) - np.dot(aux, np.dot(self.Kx_inv[str(j)], aux)) )
                    diag_var = np.squeeze(np.sqrt(sigma*np.exp(self.x_log_lambdas_dict[str(j)].cpu().detach().numpy())**-2))
                    p[a] = np.exp(-1/2*np.dot( (Y.cpu().detach().numpy()-mu)**2 , diag_var ))/np.sqrt(np.prod(diag_var)*(2*np.pi)**self.d)*np.sum(K[idx])/len(idx[0])


            # probabilities of each cluster is proportional to p*pz
            # sample new cluster

            p = np.concatenate( (p,np.array(self.alpha/(self.alpha+N-1)).reshape(1)) )
            new_z = np.argmax(p)

            if(new_z == len(keys)):
                #creates new cluster
                new_z = max(self.keys_latent_clusters)+1
                self.keys_latent_clusters.add(new_z)
                self.latent_cluster_idx[i] = new_z
                #init expert parameters
                self.x_log_lengthscales_dict[str(new_z)] = torch.nn.Parameter(sum(self.x_log_lengthscales_dict.values())/len(self.x_log_lengthscales_dict))
                self.x_log_lambdas_dict[str(new_z)] = torch.nn.Parameter(sum(self.x_log_lambdas_dict.values())/len(self.x_log_lambdas_dict))
                self.x_log_sigma_n_dict[str(new_z)] = torch.nn.Parameter(sum(self.x_log_sigma_n_dict.values())/len(self.x_log_sigma_n_dict))
                self.x_log_lin_coeff_dict[str(new_z)] = torch.nn.Parameter(sum(self.x_log_lin_coeff_dict.values())/len(self.x_log_lin_coeff_dict))
                self.Kx_inv[str(new_z)] = np.array(1).reshape(1,1)
                self.meanX[str(new_z)] = Xout[i].cpu().detach().numpy()
                self.log_det_K_x[str(new_z)] = 0
            else:
                # update cluster
                new_z = keys[new_z]
                self.latent_cluster_idx[i] = new_z
                if(new_z == z):
                    self.Kx_inv[str(new_z)] = prev_Kx_inv
                    self.meanX[str(new_z)] = prev_mean
                    self.log_det_K_x[str(z)] = prev_log_det_kx
                else:
                    idx = np.where(self.latent_cluster_idx == new_z)
                    self.init_K_x(Xin[idx], new_z)
                    idx = np.where(self.latent_cluster_idx == z)
                    if(idx[0].size > 0):
                        self.init_K_x(Xin[idx], z)
        # clear the small clusters and join them
        min_cluster = np.max(np.array(list(self.keys_map_clusters)))
        all_idx = np.array([])
        for z in list(self.keys_latent_clusters):
            idx = np.where( z == self.latent_cluster_idx )
            if len(idx[0]) <= 5:
                all_idx = np.concatenate( [all_idx, idx[0]] ).astype('int32')
                min_cluster = min( z, min_cluster )
                try:
                    self.Kx_inv.pop(str(z),None)
                except:
                    self.Kx_inv.pop(str(z))
                try:
                    self.x_log_lengthscales_dict.pop(str(z),None)
                except:
                    self.x_log_lengthscales_dict.pop(str(z))
                try:
                    self.x_log_lambdas_dict.pop(str(z),None)
                except:
                    self.x_log_lambdas_dict.pop(str(z))
                try:
                    self.x_log_sigma_n_dict.pop(str(z),None)
                except:
                    self.x_log_sigma_n_dict.pop(str(z))
                try:
                    self.x_log_lin_coeff_dict.pop(str(z),None)
                except:
                    self.x_log_lin_coeff_dict.pop(str(z))
                self.keys_latent_clusters.discard(z)
        k_x = np.ceil( len(all_idx) / self.N_max).astype('int32')
        np.random.shuffle(all_idx)
        for i in range(k_x):
            while( min_cluster in self.latent_cluster_idx ):
                min_cluster += 1
            idx = all_idx[i::k_x]
            self.latent_cluster_idx[idx] = min_cluster
            self.keys_latent_clusters.add(min_cluster)
            #init expert parameters
            try:
                self.x_log_lengthscales_dict[str(min_cluster)] = torch.nn.Parameter(sum(self.x_log_lengthscales_dict.values())/len(self.x_log_lengthscales_dict))
                self.x_log_lambdas_dict[str(min_cluster)] = torch.nn.Parameter(sum(self.x_log_lambdas_dict.values())/len(self.x_log_lambdas_dict))
                self.x_log_sigma_n_dict[str(min_cluster)] = torch.nn.Parameter(sum(self.x_log_sigma_n_dict.values())/len(self.x_log_sigma_n_dict))
                self.x_log_lin_coeff_dict[str(min_cluster)] = torch.nn.Parameter(sum(self.x_log_lin_coeff_dict.values())/len(self.x_log_lin_coeff_dict))
            except:
                self.x_log_lengthscales_dict[str(z)] = torch.nn.Parameter(torch.rand(self.d*2+self.u_dim,dtype=self.dtype, device=self.device)*0.5 + 0.5)
                self.x_log_lambdas_dict[str(z)] = torch.nn.Parameter(torch.rand(self.d,dtype=self.dtype, device=self.device)*0.5 + 0.5)
                self.x_log_sigma_n_dict[str(z)] = torch.nn.Parameter(torch.rand(1,dtype=self.dtype, device=self.device)*0.5 + 0.5)
                self.x_log_lin_coeff_dict[str(z)] = torch.nn.Parameter(torch.rand(self.d*2+self.u_dim+1,dtype=self.dtype, device=self.device)*0.5 + 0.5)
            self.meanX[str(min_cluster)] = np.mean( Xout[idx].cpu().detach().numpy() , axis = 0)


    def Gibbs_update_clusters(self):
        self.Gibbs_update_map_clusters()
        print("gibbs updated the Y mapping")

        self.Gibbs_update_latent_clusters()
        print("Gibbs updated the X mapping")
        #for numerical issues, we recalculate the inverse matrix, just in case
        self.init_K_inv()
        self.init_means()


    def get_Y(self):
        """
        Create observation matrix Y from observations_list
        Return
        ------
        Y : observation matrix
        """
        observation = np.concatenate(self.observations_list, 0)
        # self.meanY = np.mean(observation,0)
        return observation

    def gpdm_loss(self, Y, N, balance = 1, train_X = True):
        """
        Calculate GPDM loss function L = Lx + B*Ly
        Parameters
        ----------
        Y : tensor(dtype)
            observation matrix
        X : tensor(dtype)
            latent state matrix
        N : int
            number of data points
        balance : double (optional)
            balance factor B
        Return
        ------
        GPDM loss = Ly + B*Lx
        """
        lossY = torch.tensor(0).float()
        for j in self.keys_map_clusters:
            idx = self.map_cluster_idx == j
            if(train_X):
                lossY += self.get_y_neg_log_likelihood(Y[idx], self.X[idx], j)
            else:
                lossY += self.get_y_neg_log_likelihood(Y[idx], self.X[idx].detach(), j)
        lossX = torch.tensor(0).float()
        Xin, Xout, start_indeces = self.get_Xin_Xout_matrices()
        for j in self.keys_latent_clusters:
            idx = self.latent_cluster_idx == j
            if(train_X):
                lossX += self.get_x_neg_log_likelihood(Xout[idx], Xin[idx], j)
            else:
                lossX += self.get_x_neg_log_likelihood(Xout[idx].detach(), Xin[idx].detach(), j)
        loss = lossY + balance*lossX
        return loss


    def gate_likelihood_loss(self, Y, N, balance = 1):
        lossY = torch.tensor(0).float()
        for i in range(N):
            N = len(self.X)
            # map_clusters_update
            z = int(self.map_cluster_idx[i])
            idx = np.where(self.map_cluster_idx == z)
            j = np.where( idx[0] == i )[0][0]
            i = torch.tensor(i, dtype = torch.long)
            # erase from the cluster
            prev_Ky_inv = self.Ky_inv[str(z)]
            prev_mean = self.meanY[str(z)]
            self.map_cluster_idx[i] = -1
            if(len(idx[0]) == 1):
                # this cluster disappears
                self.keys_map_clusters.discard(z)
            else:
                self.Ky_inv[str(z)] = self.submatrix_inverse(prev_Ky_inv, j, self.get_u_y_kernel(self.X[idx],self.X[i].unsqueeze(0), z, j), z, 'mapping')
                self.meanY[str(z)] = (prev_mean-self.Y[i]/len(idx[0]))*(len(idx[0])/(len(idx[0])-1))
                if(np.any(np.isnan(self.Ky_inv[str(z)]))):
                    idx = np.where(self.map_cluster_idx == z)
                    Ky = self.get_y_kernel(self.X[idx],self.X[idx], z).cpu().detach().numpy()
                    self.Ky_inv[str(z)] = np.linalg.inv(Ky)
            
            p = np.zeros(len(self.keys_map_clusters))
            pz = torch.zeros( len(self.keys_map_clusters) , dtype=self.dtype, device = self.device)
            keys = list(self.keys_map_clusters)
            K = self.map_cluster_kernel( self.X[i].cpu().detach(), self.X.cpu().detach() )
            for a in range(len(keys)):
                idx = np.where(keys[a] == self.map_cluster_idx)
                pz[a] = torch.sum( K[idx])/len( idx[0] )
                p[a] = float( self.map_cluster_likelihood(keys[a], self.X[i], self.Y[i]) )
            # probabilities of each cluster is proportional to p*pz
            #pz = pz/torch.sum(pz).detach()
            if(np.any(np.isinf(p))):
                idx = np.where(np.isinf(p))
                p[:] = 0
                p[idx] = 1
            pz = pz/torch.sum(pz).detach()
            #p = p/np.sum(p)
            lossY += torch.log( torch.dot(torch.tensor(p,dtype=self.dtype, device = self.device), pz) )
            self.map_cluster_idx[i] = z
            self.keys_map_clusters.add(z)
            self.Ky_inv[str(z)] = prev_Ky_inv
            self.meanY[str(z)] = prev_mean

        Xin, Xout, start_indeces = self.get_Xin_Xout_matrices()
        lossX = torch.tensor(0).float()
        for i  in range(Xin.shape[0]):
            z = int(self.latent_cluster_idx[i])
            idx = np.where(self.latent_cluster_idx == z)
            j = np.where( idx[0] == i )[0][0]
            i = torch.tensor(i, dtype = torch.long)
            # erase from the cluster
            prev_Kx_inv = self.Kx_inv[str(z)]
            prev_mean = self.meanX[str(z)]
            self.latent_cluster_idx[i] = -1
            if(len(idx[0]) == 1):
                # this cluster disapears
                self.keys_latent_clusters.discard(z)
            else:
                # update Kx_inv
                #idx = np.where(self.latent_cluster_idx == z)
                #Kx = self.get_x_kernel(Xin[idx],Xin[idx], z).cpu().detach().numpy()
                #self.Kx_inv[str(z)] = np.linalg.inv(Kx)
                self.Kx_inv[str(z)] = self.submatrix_inverse(np.copy(prev_Kx_inv), j, np.squeeze(self.get_u_x_kernel(Xin[idx],Xin[i].unsqueeze(0), z, j)), z, 'latent' )
                self.meanX[str(z)] = (prev_mean-Xout[i].detach().numpy()/len(idx[0]))*(  len(idx[0]) / (len(idx[0])-1)  )
                if( np.any(np.isnan(self.Kx_inv[str(z)])) ):
                    idx = np.where(self.latent_cluster_idx == z)
                    Kx = self.get_x_kernel(Xin[idx],Xin[idx], z).cpu().detach().numpy()
                    self.Kx_inv[str(z)] = np.linalg.inv(Kx)
            p = np.zeros(len(self.keys_latent_clusters))
            pz = torch.zeros( len(self.keys_latent_clusters) , dtype=self.dtype, device = self.device)
            keys = list(self.keys_latent_clusters)
            X = Xin[i].cpu().detach()
            Y = Xout[i]
            K = self.latent_cluster_kernel( X, Xin.cpu().detach() )
            for a in range(len(keys)):
                j = keys[a]
                idx = np.where(self.latent_cluster_idx == j)
                pz[a] = torch.sum( K[idx])/len( idx[0] )
                aux = np.squeeze(self.get_x_kernel(X.unsqueeze(0),Xin[idx], j, False).cpu().detach().numpy())
                mu = np.dot( aux, np.dot(self.Kx_inv[str(j)], Xout[idx].cpu().detach().numpy())) + self.meanX[str(z)]
                sigma = np.abs( self.get_x_kernel(X.unsqueeze(0), X.unsqueeze(0), j, False).cpu().detach().numpy() - np.dot(aux, np.dot(self.Kx_inv[str(j)], aux)) )
                diag_var = np.squeeze( np.sqrt(sigma*np.exp(self.x_log_lambdas_dict[str(j)].cpu().detach().numpy())**-2))
                p[a] = np.exp(-1/2*np.dot( (Y.cpu().detach().numpy()-mu)**2 , diag_var ))/np.sqrt(np.prod(diag_var)*(2*np.pi)**self.d)
            # probabilities of each cluster is proportional to p*pz
            if(np.any(np.isinf(p))):
                idx = np.where(np.isinf(p))
                p[:] = 0
                p[idx] = 1
            pz = pz/torch.sum(pz).detach()
            #p = p/np.sum(p)
            lossX += torch.log(  torch.dot(torch.tensor(p,dtype=self.dtype, device = self.device), pz)  )
            self.latent_cluster_idx[i] = z
            self.Kx_inv[str(z)] = prev_Kx_inv
            self.meanX[str(z)] = prev_mean
            self.keys_latent_clusters.add(z)

        return lossY+balance*lossX


    def train_experts(self, num_opt_steps, num_print_steps, lr=0.01, balance=1, train_X = True, optimizer_type = 'LBFGS'):

        print('\n### Optimize MoGPDM Experts (L-BFGS) ###')
        # create observation matrix
        Y = self.get_Y()
        N = Y.shape[0]
        Y = torch.tensor(Y, dtype=self.dtype, device=self.device)
        # set list of parameters
        '''
        parameters = []
        for z in self.keys_map_clusters:
            parameters.append( self.y_log_lengthscales_dict[str(z)] )
            parameters.append( self.y_log_lambdas_dict[str(z)]      )
            parameters.append( self.y_log_sigma_n_dict[str(z)]      )
        for z in self.keys_latent_clusters:
            parameters.append( self.x_log_lengthscales_dict[str(z)] )
            parameters.append( self.x_log_lambdas_dict[str(z)]      )
            parameters.append( self.x_log_sigma_n_dict[str(z)]      )
            parameters.append( self.x_log_lin_coeff_dict[str(z)]    )
        
        if( train_X ):
            parameters.append( self.X )
        '''
        self.set_training_mode()
        if(optimizer_type == 'LBFGS'):
            optimizer = torch.optim.LBFGS(  self.parameters() , lr=lr, max_iter=20, history_size=7, line_search_fn='strong_wolfe'  )
            #optimizer_X = torch.optim.LBFGS( [self.X], lr=0.1*lr, max_iter=20, history_size=7, line_search_fn='strong_wolfe')
        elif(optimizer_type == 'adam'):
            parameters = []
            for z in self.keys_map_clusters:
                parameters.append( {'params' : self.y_log_lengthscales_dict[str(z)] } )
                parameters.append( {'params' : self.y_log_lambdas_dict[str(z)]      } )
                parameters.append( {'params' : self.y_log_sigma_n_dict[str(z)]      } )
            for z in self.keys_latent_clusters:
                parameters.append( {'params' : self.x_log_lengthscales_dict[str(z)] } )
                parameters.append( {'params' : self.x_log_lambdas_dict[str(z)]      } )
                parameters.append( {'params' : self.x_log_sigma_n_dict[str(z)]      } )
                parameters.append( {'params' : self.x_log_lin_coeff_dict[str(z)]    } )
            if( train_X ):
                parameters.append( {'params' : self.X, 'lr' :1e-2*lr} )
            optimizer = torch.optim.Adam(parameters, lr = lr)

        losses = []
        t_start = time.time()
        for epoch in range(num_opt_steps):
            if(optimizer_type == 'LBFGS'):
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    loss = self.gpdm_loss(Y, N, balance, train_X = train_X)
                    if loss.requires_grad:
                        loss.backward()
                    return loss
                losses.append(closure().item())
                optimizer.step(closure)
            elif(optimizer_type == 'adam'):
                optimizer.zero_grad()
                loss = self.gpdm_loss(Y, N, balance, train_X = train_X)
                loss.backward()
                if torch.isnan(loss):
                    print('Loss is nan')
                    break
                optimizer.step()
                losses.append(loss.item())
            
            #optimizer_X.step(closure)
            if epoch % num_print_steps == 0:
                print('\n Opt. EPOCH:', epoch)
                print('Running loss:', "{:.4e}".format(losses[-1]))
                t_stop = time.time()
                print('Time elapsed:',t_stop-t_start)
                t_start = t_stop
            self.init_means()
        self.init_K_inv()
        return losses

    def train_gibbs_update(self, num_opt_steps, num_print_steps, update_mapping = True, update_latent = True, update = 'loss'):
        """
        Update Gibbs
        Parameters
        ----------
        num_opt_steps : int
            number of optimization steps
        num_print_steps : int
            number of steps between printing info
        """
        print('\n### Update clusters ussing Gibbs method ###')
        # create observation matrix
        # define optimizer
        t_start = time.time()
        self.init_K_inv()
        for epoch in range(num_opt_steps):
            if(update_mapping):
                self.Gibbs_update_map_clusters(update = update)
                print("gibbs updated the Y mapping")
            if(update_latent):
                self.Gibbs_update_latent_clusters()
                print("Gibbs updated the X mapping")
            self.init_K_inv()
            self.init_means()
            if epoch % num_print_steps == 0:
                print('\n Opt. EPOCH:', epoch)
                t_stop = time.time()
                print('Time elapsed:',t_stop-t_start)
                t_start = t_stop


    def train_Gates(self, num_opt_steps, num_print_steps, lr=0.01, balance=1):
        """
        Optimize GPDM with L-BFGS
        Parameters
        ----------
        num_opt_steps : int
            number of optimization steps
        num_print_steps : int
            number of steps between printing info
        lr : double
            learning rate
        balance : double
            balance factor for gpdm_loss
        Return
        ------
        losses: list of loss evaluated
        """
        print('\n### Optimize Gates Weights with Gradient Descent ###')
        # create observation matrix
        Y = self.get_Y()
        N = Y.shape[0]
        Y = torch.tensor(Y, dtype=self.dtype, device=self.device)
        # define optimizer
        self.set_training_mode()
        parameters = []
        parameters.append(   self.map_cluster_kernel_length  )
        parameters.append( self.latent_cluster_kernel_length )
        optimizer = torch.optim.LBFGS(  parameters, lr=lr, max_iter=20, history_size=7, line_search_fn='strong_wolfe'  )
        #optimizer = torch.optim.Adam(  parameters , lr=lr)#, max_iter=20, history_size=7, line_search_fn='strong_wolfe'  )
        losses = []
        t_start = time.time()
        for epoch in range( num_opt_steps ):
            '''
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                loss = self.gate_likelihood_loss(Y, N, balance)
                if loss.requires_grad:
                    loss.backward()
                return loss
            losses.append(closure().item())
            optimizer.step(closure)
            #optimizer_X.step(closure)
            '''
            optimizer.zero_grad()
            loss = self.gate_likelihood_loss(Y, N, balance)
            loss.backward()
            losses.append(loss.item())
            #print( self.map_cluster_kernel_length.grad )
            #print( self.latent_cluster_kernel_length.grad )

            with torch.no_grad():
                self.latent_cluster_kernel_length.copy_(self.latent_cluster_kernel_length - lr*self.latent_cluster_kernel_length.grad)
                self.map_cluster_kernel_length.copy_(self.map_cluster_kernel_length - lr*self.map_cluster_kernel_length.grad)

            if epoch % num_print_steps == 0:
                print('\n Opt. EPOCH:', epoch)
                print('Running loss:', "{:.4e}".format(losses[-1]))
                t_stop = time.time()
                print('Time elapsed:',t_stop-t_start)
                t_start = t_stop

        return losses


    def train_lbfgs(self, num_opt_steps, num_print_steps, lr=0.01, balance='default', opt_per_step = 3, update_gate = 'False'):
        """
        Optimize GPDM with L-BFGS
        Parameters
        ----------
        num_opt_steps : int
            number of optimization steps
        num_print_steps : int
            number of steps between printing info
        lr : double
            learning rate
        balance : double
            balance factor for gpdm_loss
        Return
        ------
        losses: list of loss evaluated
        """
        print('\n### Optimize GPDM (L-BFGS) ###')
        # create observation matrix
        Y = self.get_Y()
        N = Y.shape[0]
        Y = torch.tensor(Y, dtype=self.dtype, device=self.device)
        # define optimizer
        f_optim = lambda p : torch.optim.LBFGS(p, lr=lr, max_iter=20, history_size=7, line_search_fn='strong_wolfe')
        optimizer = f_optim(self.parameters())
        losses = []
        t_start = time.time()
        for epoch in range(num_opt_steps):
            t1 = time.time()
            for opt in range(int(opt_per_step)):
                def closure():
                    if torch.is_grad_enabled():
                        optimizer.zero_grad()
                    loss = self.gpdm_loss(Y, N, balance)
                    if loss.requires_grad:
                        loss.backward()
                    return loss
                losses.append(closure().item())
                optimizer.step(closure)
            t2 = time.time()
            print('Epoch '+str(epoch)+', GP elapsed time is',t2-t1)
            t1 = t2
            self.init_K_inv()
            self.Gibbs_update_clusters()
            t2 = time.time()
            print('Epoch '+str(epoch)+', Gibbs update elapsed time is',t2-t1)
            t1 = t2
            if(update_gate):
                for opt in range(int(opt_per_step)):
                    def closure():
                        if torch.is_grad_enabled():
                            optimizer.zero_grad()
                        loss = self.gate_likelihood_loss(Y, N, balance)
                        if loss.requires_grad:
                            loss.backward()
                        return loss
                    optimizer.step(closure)
                t2 = time.time()
                print('Epoch '+str(epoch)+', Gate optimization elapsed time is',t2-t1)

            if epoch % num_print_steps == 0:
                print('\nGPDM Opt. EPOCH:', epoch)
                print('Running loss:', "{:.4e}".format(losses[-1]))
                t_stop = time.time()
                print('Time elapsed:',t_stop-t_start)
                t_start = t_stop
        return losses

    def train_adam(self, num_opt_steps, num_print_steps, lr=0.01, balance='default'):
        """
        Optimize GPDM with Adam
        Parameters
        ----------
        num_opt_steps : int
            number of optimization step
        num_print_steps : int
            number of steps between printing info
        lr : double
            learning rate
        balance : double
            balance factor for gpdm_loss
        Return
        ------
        losses : list of loss evaluated
        """
        print('\n### Optimize GPDM (Adam) ###')
        # create observation matrix
        Y = self.get_Y()
        N = Y.shape[0]
        Y = torch.tensor(Y, dtype=self.dtype, device=self.device)
        self.set_training_mode()
        # define optimizer
        f_optim = lambda p : torch.optim.Adam(p, lr=lr)
        optimizer = f_optim(self.parameters())
        t_start = time.time()
        losses = []
        for epoch in range(num_opt_steps):
            optimizer.zero_grad()
            loss = self.gpdm_loss(Y, N, balance)
            loss.backward()
            if torch.isnan(loss):
                print('Loss is nan')
                break
            optimizer.step()
            losses.append(loss.item())
            if epoch % num_print_steps == 0:
                print('\nGPDM Opt. EPOCH:', epoch)
                print('Running loss:', "{:.4e}".format(loss.item()))
                t_stop = time.time()
                print('Time elapsed:',t_stop-t_start)
                t_start = t_stop
        return losses


    def save(self, folder='MODEL/'):
        """
        Save model

        Parameters
        ----------

        folder : string (optional)
            path of the desired save folder

        """

        print('\n### Save init data and model ###')
        torch.save(self.state_dict(), folder+'Mogpdm.pt')
        log_dict={}
        log_dict['observations_list'] = self.observations_list
        log_dict['controls_list'] = self.controls_list
        log_dict['D'] = self.D
        log_dict['d'] = self.d
        log_dict['u_dim'] = self.u_dim
        log_dict['N_max'] = self.N_max
        log_dict['sigma_n_num_X'] = self.sigma_n_num_X
        log_dict['sigma_n_num_Y'] = self.sigma_n_num_Y
        log_dict['keys_map_clusters'] = self.keys_map_clusters
        log_dict['map_cluster_idx'] = self.map_cluster_idx
        log_dict['keys_latent_clusters'] = self.keys_latent_clusters
        log_dict['latent_cluster_idx'] = self.latent_cluster_idx
        log_dict['map_cluster_kernel_length'] = self.map_cluster_kernel_length
        log_dict['latent_cluster_kernel_length'] = self.latent_cluster_kernel_length
        log_dict['PCA'] = self.pca
        pickle.dump(log_dict, open(folder+'log_dict.pt', 'wb'))



    def get_latent_sequences(self):
        """
        Return the latent trajectories associated to each observation sequence recorded

        Return
        ------

        X_list : list of latent states associated to each observation sequence
        """

        X_np = self.X.clone().detach().numpy()
        X_list = []
        x_start_index = 0
        for j in range(len(self.observations_list)):
            sequence_length = self.observations_list[j].shape[0]
            X_list.append(X_np[x_start_index:x_start_index+sequence_length,:])
            x_start_index = x_start_index+sequence_length

        return X_list

    def expert_map_x_to_y(self, Xstar, z, flg_noise=False):
        """
        Map Xstar to observation space: return mean and variance
        Parameters
        ----------
        Xstar : tensor(dtype)
            input latent state matrix
        flg_noise : boolean
            add noise to prediction variance
        Return
        ------
        mean_Y_pred : mean of Y prediction
        diag_var_Y_pred : variance of Y prediction
        """
        idx = np.where(self.map_cluster_idx == z)
        Y_obs = torch.tensor(self.Y[idx], dtype=self.dtype, device=self.device)
        Ky_inv = torch.tensor(self.Ky_inv[str(z)], dtype=self.dtype, device=self.device)
        Ky_star = self.get_y_kernel(self.X[idx],Xstar,z, False)
        mean_Y_pred = torch.chain_matmul(Y_obs.t()-self.meanY[str(z)].reshape(self.D,1),Ky_inv,Ky_star).t()
        diag_var_Y_pred_common = self.get_y_diag_kernel(Xstar, z, flg_noise) - torch.sum(torch.matmul(Ky_star.t(),Ky_inv)*Ky_star.t(),dim=1)
        y_log_lambdas = torch.exp(self.y_log_lambdas_dict[str(z)])**-2
        diag_var_Y_pred = diag_var_Y_pred_common.unsqueeze(1)*y_log_lambdas.unsqueeze(0)

        return mean_Y_pred + torch.tensor(self.meanY[str(z)], dtype=self.dtype, device=self.device), diag_var_Y_pred

    def map_x_to_y(self, X_hat, flg_noise = False):
        """
        Map X_hat to observation space: return mean and variance
        Parameters
        ----------
        X_hat : tensor(dtype)
            input latent state matrix
        flg_noise : boolean
            add noise to prediction variance
        Return
        ------
        mean_Y_pred : mean of Y prediction
        diag_var_Y_pred : variance of Y prediction
        """
        zs = list(self.keys_map_clusters)
        pz = torch.zeros(X_hat.shape[0],len(zs))
        K = self.map_cluster_kernel( X_hat.reshape(X_hat.shape[0],1,self.d), self.X.reshape(1, self.X.shape[0], self.d ) )
        for a in range(len(zs)):
            idx = np.where(self.map_cluster_idx == zs[a])
            pz[:,a] = torch.sum( K[:,idx[0]], axis = 1).squeeze()
        pz = pz/torch.sum(pz, axis = 1,keepdims = True).detach()
        mean_Y_pred = torch.zeros(X_hat.shape[0],self.D)
        gp_out_var = torch.zeros(X_hat.shape[0],self.D)
        for a in range(len(zs)):
            gp_mean, gp_var = self.expert_map_x_to_y(X_hat, zs[a])
            mean_Y_pred += pz[:,a].unsqueeze(-1)*gp_mean.squeeze()
            gp_out_var += pz[:,a].unsqueeze(-1)*gp_var.squeeze()

        return mean_Y_pred, gp_out_var


    def get_y_diag_kernel(self, X, z, flg_noise=False):
        """
        Compute only the diagonal of the latent mapping kernel GP Y
        Parameters
        ----------
        X : tensor(dtype)
            latent state matrix
        z : cluster indicator
        flg_noise : boolean
            add noise to prediction variance
        Return
        ------
        GP Y diag covariance matrix
        """
        n = X.shape[0]
        if flg_noise:
            return torch.ones(n, dtype=self.dtype, device=self.device) + torch.exp(self.y_log_sigma_n_dict[str(z)])**2 + self.sigma_n_num_Y**2
        else:
            return torch.ones(n, dtype=self.dtype, device=self.device)

    def expert_map_x_dynamics(self, Xstar, z, flg_noise=False):
        """
        Map Xstar to GP dynamics output
        Parameters
        ----------
        Xstar : tensor(dtype)
            input latent state matrix
        flg_noise : boolean
            add noise to kernel matrix
        Return
        ------
        mean_Xout_pred : mean of Xout prediction
        diag_var_Xout_pred : variance of Xout prediction
        """
        idx = np.where(self.latent_cluster_idx == z)
        Xin, Xout, _ = self.get_Xin_Xout_matrices()
        Kx_inv = torch.tensor(self.Kx_inv[str(z)], dtype=self.dtype, device=self.device)
        Kx_star = self.get_x_kernel(Xin[idx],Xstar,z, False)
        mean_Xout_pred = torch.chain_matmul(Xout[idx].t() - torch.tensor(self.meanX[str(z)].reshape(self.d, 1)),Kx_inv,Kx_star).t()
        diag_var_Xout_pred_common = self.get_x_diag_kernel(Xstar, z, flg_noise) - torch.sum(torch.matmul(Kx_star.t(),Kx_inv)*Kx_star.t(),dim=1)
        x_log_lambdas = torch.exp(self.x_log_lambdas_dict[str(z)])**-2
        diag_var_Xout_pred = diag_var_Xout_pred_common.unsqueeze(1)*x_log_lambdas.unsqueeze(0)
        return mean_Xout_pred + torch.tensor(self.meanX[str(z)], dtype=self.dtype, device=self.device), diag_var_Xout_pred

    def get_x_diag_kernel(self, X, z, flg_noise=False):
        """
        Compute only the diagonal of the dynamics mapping kernel GP Y
        Parameters
        ----------
        X : tensor(dtype)
            latent state matrix
        flg_noise : boolean
            add noise to prediction variance
        Return
        ------
        GP X diag covariance matrix
        """
        n = X.shape[0]
        Sigma = torch.diag(torch.exp(self.x_log_lin_coeff_dict[str(z)])**2)
        X = torch.cat([X,torch.ones(X.shape[0],1, dtype=self.dtype, device=self.device)],1)
        if flg_noise:
            return torch.ones(n, dtype=self.dtype, device=self.device) + torch.exp(self.x_log_sigma_n_dict[str(z)])**2 + self.sigma_n_num_X**2 +\
                   torch.sum(torch.matmul(X, Sigma)*(X), dim=1)
        else:
            return torch.ones(n, dtype=self.dtype, device=self.device) + \
                   torch.sum(torch.matmul(X, Sigma)*(X), dim=1)

    def get_next_x(self, Xold, Xin, flg_sample=False):
        """
        Predict GP X dynamics output to next latent state
        Parameters
        ----------
        gp_mean_out : tensor(dtype)
            mean of the GP X dynamics output
        gp_out_var : tensor(dtype)
            variance of the GP X dynamics output
        Xold : tensor(dtype)
            present latent state
        flg_noise : boolean
            add noise to prediction variance
        Return
        ------
        Predicted new latent state
        """
        XinT, __ , _ = self.get_Xin_Xout_matrices()
        zs = list(self.keys_latent_clusters)
        pz = np.zeros(len(zs))
        K = self.latent_cluster_kernel( Xin, XinT )
        for a in range(len(pz)):
            idx = np.where(self.latent_cluster_idx == zs[a])
            pz[a] = torch.sum( K[idx] )
        pz = pz/np.sum(pz)
        gp_mean_out = torch.zeros(self.d)
        gp_out_var = torch.zeros(self.d)
        for a in range(len(pz)):
            gp_mean, gp_var = self.expert_map_x_dynamics(Xin, zs[a])
            gp_mean_out = pz[a]*gp_mean
            gp_out_var = pz[a]*gp_var
        distribution = Normal(gp_mean_out, torch.sqrt(gp_out_var))
        if flg_sample:
            return Xold + distribution.rsample()
        else:
            return Xold + gp_mean_out

    def rollout(self, num_steps, control_actions, X0, X1=None, flg_sample_X=False, flg_sample_Y=False, flg_noise=False, sample = 'mean'):
        """
        Generate a rollout of length 'num_step'. Return latent and observation trajectories
        Parameters
        ----------
        num_steps : int
            rollout length
        control_actions : tensor(dtype)
            list of control inputs (dimension: N x u_dim)
        X0 : tensor(dtype)
            latent state at t=0
        X1 : tensor(dtype) (optionla)
            latent state at t=1
        flg_sample_X : boolean (optional)
            sample GP X output
        flg_sample_Y : boolean (optional)
            sample GP Y output
        flg_noise : boolean (optional)
            add noise to prediction variance
        Return
        ------
        X_hat : latent state rollout
        Y_hat : observation rollout
        time : vector of rollout times
        """

        if control_actions.shape[0]!=num_steps:
            raise ValueError('len(control_actions) must be equal to num_steps!')
        if not torch.is_tensor(control_actions):
            control_actions = torch.tensor(control_actions, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            X_hat = torch.zeros((num_steps, self.d), dtype=self.dtype, device=self.device)
            X_hat[0,:] = torch.tensor(X0, dtype=self.dtype, device=self.device)
            if X1 is None:
                X1 = X0
            X_hat[1,:] = torch.tensor(X1, dtype=self.dtype, device=self.device)
            t_start = 1
            # generate latent rollout
            XinT, Xout , _ = self.get_Xin_Xout_matrices()
            zs = list(self.keys_latent_clusters)
            idx = []
            for a in range(len(zs)):
                idx.append(np.where(self.latent_cluster_idx == zs[a]))
            time_tracking = np.zeros(num_steps-t_start)
            for t in range(t_start,num_steps):
                time_start = time.time()
                
                Xin = torch.cat((X_hat[t:t+1,:], X_hat[t-1:t,:], control_actions[t:t+1,:]),1)
                Xold = X_hat[t:t+1,:]
                pz = np.zeros(len(zs))
                K = self.latent_cluster_kernel( Xin, XinT )
                for a in range(len(pz)):
                    pz[a] = torch.sum( K[idx[a]] )
                pz = pz/np.sum(pz)
                if(np.any(np.isnan(pz))):
                    break
                
                if(sample == 'mean'):
                    gp_mean_out = torch.zeros(self.d)
                    gp_out_var = torch.zeros(self.d)
                    for a in range(len(pz)):
                        Kx_inv = torch.tensor(self.Kx_inv[str(zs[a])], dtype=self.dtype, device=self.device)
                        Kx_star = self.get_x_kernel(XinT[idx[a]],Xin,zs[a], False)
                        mean_Xout_pred = torch.chain_matmul(Xout[idx[a]].t(),Kx_inv,Kx_star).t() + torch.tensor(self.meanX[str(zs[a])], dtype=self.dtype, device=self.device)
                        diag_var_Xout_pred_common = self.get_x_diag_kernel(Xin, zs[a], False) - torch.sum(torch.matmul(Kx_star.t(),Kx_inv)*Kx_star.t(),dim=1)
                        x_log_lambdas = torch.exp(self.x_log_lambdas_dict[str(zs[a])])**-2
                        diag_var_Xout_pred = diag_var_Xout_pred_common.unsqueeze(1)*x_log_lambdas.unsqueeze(0)
                        gp_mean_out += pz[a]*mean_Xout_pred.squeeze()
                        gp_out_var += pz[a]*diag_var_Xout_pred.squeeze()
                    if flg_sample_X:
                        distribution = Normal(gp_mean_out, torch.sqrt(gp_out_var))
                        X_next = distribution.rsample()
                    else:
                        X_next = gp_mean_out
                elif(sample == 'mixture'):
                    a = np.random.choice(len(pz), p = pz)
                    Kx_inv = torch.tensor(self.Kx_inv[str(zs[a])], dtype=self.dtype, device=self.device)
                    Kx_star = self.get_x_kernel(XinT[idx[a]],Xin,zs[a], False)
                    mean_Xout_pred = torch.chain_matmul(Xout[idx[a]].t(),Kx_inv,Kx_star).t() + torch.tensor(self.meanX[str(zs[a])], dtype=self.dtype, device=self.device)
                    diag_var_Xout_pred_common = self.get_x_diag_kernel(Xin, zs[a], False) - torch.sum(torch.matmul(Kx_star.t(),Kx_inv)*Kx_star.t(),dim=1)
                    x_log_lambdas = torch.exp(self.x_log_lambdas_dict[str(zs[a])])**-2
                    diag_var_Xout_pred = diag_var_Xout_pred_common.unsqueeze(1)*x_log_lambdas.unsqueeze(0)
                    if flg_sample_X:
                        distribution = Normal(mean_Xout_pred, diag_var_Xout_pred)
                        X_next = distribution.rsample()
                    else:
                        X_next = mean_Xout_pred
                X_hat[t+1:t+2,:] = Xold + X_next

                time_end = time.time()
                time_tracking[t-t_start] = time_end-time_start


            # map to observation space

            mean_Y_pred, gp_out_var = self.map_x_to_y(X_hat)

            if flg_sample_Y:
                distribution = Normal(mean_Y_pred, torch.sqrt(gp_out_var))
                Y_hat = distribution.rsample()
            else:
                Y_hat = mean_Y_pred

            return X_hat, Y_hat, time_tracking


    def get_dynamics_map_performance(self, flg_noise=False):
        """
        Measure accuracy in latent dynamics prediction

        Parameters
        ----------

        flg_noise : boolean (optional)
            add noise to prediction variance

        Return
        ------

        mean_Xout_pred : mean of Xout prediction

        var_Xout_pred : variance of Xout prediction

        Xout : Xout matrix

        Xin : Xin matrix

        NMSE : Normalized Mean Square Error

        """

        with torch.no_grad():
            Xin, Xout, _ = self.get_Xin_Xout_matrices()
            mean_Xout_pred, var_Xout_pred = self.map_x_dynamics(Xin,flg_noise=flg_noise)

            mean_Xout_pred = mean_Xout_pred.clone().detach().numpy()
            var_Xout_pred = var_Xout_pred.clone().detach().numpy()
            Xout = Xout.clone().detach().numpy()
            Xin = Xin.clone().detach().numpy()

        return mean_Xout_pred, var_Xout_pred, Xout, Xin


    def get_latent_map_performance(self, flg_noise=False):
        """
        Measure accuracy of latent mapping

        Parameters
        ----------

        flg_noise : boolean (optional)
            add noise to prediction variance

        Return
        ------

        mean_Y_pred : mean of Y prediction

        var_Y_pred : variance of Y prediction

        Y : True observation matrix

        NMSE : Normalized Mean Square Error

        """

        with torch.no_grad():
            mean_Y_pred, var_Y_pred = self.map_x_to_y(self.X, flg_noise=flg_noise)

            mean_Y_pred = mean_Y_pred.clone().detach().numpy()
            var_Y_pred = var_Y_pred.clone().detach().numpy()

            Y = self.get_Y() + self.meanY

            return mean_Y_pred, var_Y_pred, Y




