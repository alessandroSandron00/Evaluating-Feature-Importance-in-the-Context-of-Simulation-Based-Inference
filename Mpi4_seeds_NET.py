from mpi4py import MPI

import random
import signal
import torch
import pickle
import numpy as np

from sklearn.isotonic import spearmanr
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn


import matplotlib.pyplot as plt
from sbi import analysis as analysis
from sbi import utils

import pandas as pd
import seaborn as sns
import os

import psutil

########## INDRODUCTION #######
#SBI allows us to evaluate a distribution as the model's output rather than a single value.
# In this analysis, we assessed the influence of each feature on the distribution of each parameter using different metrics, 
# providing a more comprehensive understanding of their impact.
#Given the dataset size, all computations were performed on a university server, 
# providing the necessary computational resources for an in-depth analysis.

#################
# function
def handler(signum, frame):
    raise TimeoutError("Il comando ha impiegato troppo tempo per eseguirsi")

# singal
signal.signal(signal.SIGALRM, handler)

# timeout to avoit error/block from the sistem ( there are some datum that could be difficult to simulatite or predict ( over time))
timeout_sec = 3
###########################


# bild matrix features parameter distribuzion 
########################################### strart function 
def matrix(X_test,y_test,N_distr,y_train,X_train,density_estimator_building_fun,prior,theta_para, param_limits):

    # on sample
    final_IQR=np.zeros((y_train.shape[1],X_train.shape[1]+1))
    final_CU=np.zeros((y_train.shape[1],X_train.shape[1]+1))
    final_KL=np.zeros((y_train.shape[1],X_train.shape[1]+1))
    final_corr_mat=np.zeros((int((len(theta_para)*(len(theta_para)-1))/2),X_train.shape[1]+1))
    final_Pre=np.zeros((y_train.shape[1],X_train.shape[1]+1))
    
    init_kl=np.zeros((N_distr,y_train.shape[1],X_test.shape[0]))
    ii=0
    process = psutil.Process(os.getpid())
    #base_memory_usage = process.memory_info().rss
    #print(base_memory_usage)
    problem=[]
    while ii<=X_train.shape[1]:
        print(f'features {ii}')
        #p=base_memory_usage-process.memory_info().rss

        IQR=np.zeros((y_train.shape[1],X_test.shape[0]))
        CU=np.zeros((y_train.shape[1],X_test.shape[0]))
        KL=np.zeros((y_train.shape[1],X_test.shape[0]))
        PRE=np.zeros((y_train.shape[1],X_test.shape[0]))
        corr_mat=np.zeros((int((len(theta_para)*(len(theta_para)-1))/2),X_test.shape[0]))
        if ii==0 : #all features
            
            #inference = SNPE(prior=prior)
            inference = SNPE(prior=prior,density_estimator=density_estimator_building_fun)
            # Aggiungi le features estratte come simulazioni per l'addestramento del modello
            inference.append_simulations(
                torch.from_numpy(np.float32(y_train)),
                torch.from_numpy(np.float32(X_train)))

            # train the neural density estimator
            density_estimator = inference.train(force_first_round_loss=True)
            posterior = inference.build_posterior(density_estimator)
            j=0
            #posterior_samples=np.numpy
            post_array=np.zeros((N_distr,y_test.shape[1]))

            rem=[]
            for idx in range(X_test.shape[0]):
                #a=base_memory_usage-process.memory_info().rss
                print(f'sample{idx}####')
                sample_ramdom= torch.from_numpy(np.array(X_test[idx,:], dtype=np.float32))
                try:
                    # Imposta un allarme che scatta dopo timeout_sec secondi
                    signal.alarm(timeout_sec)
                    post_array = (posterior.sample((N_distr,), sample_ramdom)).numpy()
                    #print(posterior_samples.type())
                    #post_array= posterior_samples.numpy()#(100x7)
                except TimeoutError:
                    print(F'sample{idx}:INTERROPT####################################')
                    problem.append(idx)
                    rem.append(j)
                
                finally:
                    # Disattiva l'allarme
                    signal.alarm(0)
                #print(posterior_samples.type())
                #post_array= posterior_samples.numpy()#(100x7)
                #b=base_memory_usage-process.memory_info().rss
                
                IQR[:,j]=calculate_iqr(post_array)
                CU[:,j]=calculate_kurtosis(post_array)
                PRE[:,j]=calculate_PRE(y_test[idx,:],post_array, param_limits)
                corr_mat[:,j],corr_name=corr_distrib(post_array,theta_para)
                init_kl[:,:,j]=post_array
                j=j+1

            IQR=np.delete(IQR, rem, axis=1)
            CU=np.delete(CU, rem, axis=1)
            PRE=np.delete(PRE, rem, axis=1)
            KL=np.delete(KL, rem, axis=1)
            corr_mat=np.delete(corr_mat, rem, axis=1)

            #PRE_error.append(calculate_PRE(theta_data, features, posterior, num_samples=N_distr))
        else:
            #without single features
            new_f=[col for col in range(X_train.shape[1]) if col != ii-1]
            #inference = SNPE(prior=prior)
            inference = SNPE(prior=prior,density_estimator=density_estimator_building_fun)
            #features singol features 
            inference.append_simulations(
            torch.from_numpy(np.float32(y_train)),
            torch.from_numpy(np.float32(X_train[:, new_f])))
            # train the neural density estimator
            density_estimator = inference.train(force_first_round_loss=True)
            posterior = inference.build_posterior(density_estimator)
            j=0
            rem=[] #rimozione bad sample 
            post_array=np.zeros((N_distr,y_test.shape[1]))

            for idx in range(X_test.shape[0]):
                #a=base_memory_usage-process.memory_info().rss
                print(f'sample{idx}####')
                sample_ramdom= torch.from_numpy(np.array(X_test[idx, new_f], dtype=np.float32))
                try:
                    
                    signal.alarm(timeout_sec)
                    post_array = (posterior.sample((N_distr,), sample_ramdom)).numpy()
                
                 #   b = base_memory_usage-process.memory_info().rss

                except TimeoutError:
                    print(F'sample{idx},:INTERROPT')
                    problem.append((idx,ii))
                    rem.append(j)

                
                finally:
                    # Disattiva l'allarme
                    signal.alarm(0)
                
                
                #b=base_memory_usage-process.memory_info().rss
                #print(f'sample{idx}:--{a-b}')
                IQR[:,j]=calculate_iqr(post_array)
                CU[:,j]=calculate_kurtosis(post_array)
                PRE[:,j]=calculate_PRE(y_test[idx,:],post_array, param_limits)
                KL[:,j]=kl_divergence(init_kl[:,:,j],post_array)
                corr_mat[:,j],corr_name=corr_distrib(post_array,theta_para)
                j=j+1
            
            IQR=np.delete(IQR, rem, axis=1)
            CU=np.delete(CU, rem, axis=1)
            PRE=np.delete(PRE, rem, axis=1)
            KL=np.delete(KL, rem, axis=1)
            corr_mat=np.delete(corr_mat, rem, axis=1)

        final_IQR[:,ii]=np.mean(IQR,axis=1)
        final_CU[:,ii]=np.mean(CU,axis=1)
        if ii>0:
            final_KL[:,ii]=np.mean(KL,axis=1)
            
        final_corr_mat[:,ii]=np.mean(corr_mat,axis=1) 
        final_Pre[:,ii]=np.mean(PRE,axis=1)
        #TOT_para_array=np.concatenate((TOT_para_array,post_array.reshape(N_distr, theta_data.shape[1], 1)),axis=2)
        

        #k0=base_memory_usage-process.memory_info().rss
        print(f'fine features {ii}################################################')
        ii=ii+1
    return final_IQR,final_CU,final_KL,final_corr_mat,final_Pre,corr_name, problem

######################################## end function ########################


######################### normalization #################
def normalize_data(data, param_limits):

    normalized_data = np.empty_like(data, dtype=np.float64)
    for par in param_limits:
        for i in range(data.shape[0]):
            normalized_data[i] = (data[i] - par[0]) / (par[1] - par[0])
    return normalized_data
######################### end #####################

######### PRE error #####################
def calculate_PRE(theta_data, distribution, param_limits):

    if param_limits is not None:
        theta_data = normalize_data(theta_data, param_limits)
        distribution = normalize_data(distribution, param_limits)

    # Calcolo delle differenze tra la distribuzione e i dati theta
    differences = distribution.T - theta_data.reshape((-1, 1))


    # Calcolo del PRE per ciascun parametro
    pre_params = np.mean(differences**2, axis=1) # vector dimension 7 
    
    return pre_params
######### END  PRE error #####################

######## IQR ###########################
def calculate_iqr(data):

    # Calcolo del primo e terzo quartile
    IQR=[]
    for i in range(data.shape[1]):
        Q1 = np.percentile(data[:,i], 25)
        Q3 = np.percentile(data[:,i], 75)
        # Calcolo dell'IQR
        iqr = Q3 - Q1
        IQR.append(iqr)
    return np.array(IQR)
######## END IQR #######################

###### CURTOSI ###############
def calculate_kurtosis(data): 

    CU=[]
    for i in range(data.shape[1]):
        mean = np.mean(data[:,i])
        std_dev = np.std(data)

        # Calcola la curtosi 
        kurt = np.mean((data - mean) ** 4) / (std_dev ** 4)
        CU.append(kurt)

    return np.array(CU)

##########   END CURTOSI  ################

############### KL ####################  quanto Q( approssimata)  apporssima P ( vera)
def kl_divergence(P_distr, Q_distr):
    KL=[]
    for i in range(P_distr.shape[1]):
        kl = np.sum(P_distr[:,i]* np.log(P_distr[:,i] / Q_distr[:,i]))
        KL.append(kl)
    return np.array(KL)

############## END KL #########################

########### START CORELATION ##################
def corr_distrib(distrib,theta_para):
    # correlation parameters and features
    corr_para=np.zeros((len(theta_para),len(theta_para)))
    for j in range(theta_data.shape[1]):
        for i in range(theta_data.shape[1]):
           corr, p_value = spearmanr(distrib[:,j], distrib[:,i])
           if p_value<0.05:
               corr_para[i,j]=corr
           else:
               corr_para[i,j]=0
    final=[]
    final_name=[]
    for row in range(len(corr_para)):
        for column in range(len(corr_para[row])):
            if column < row:  # check
                final.append(corr_para[row][column])
                stri=theta_para[row] + " " + theta_para[column]
                final_name.append(stri)
    return np.array(final), final_name

############ END CORRELATION ##################


#################START CODE #############
#START CODE ##############################################################################################################################################
sim=948819 # Pool dimension 
#fimrdion of test increase gradually
TT=3000


k_fold=10
rep=1

N_distr=2000
seed=1

### seed
random.seed(57)
torch.manual_seed(57)
np.random.seed(57)

##
with open('catch22/sim_theta', 'rb') as file: #catch22/
    th_data = pickle.load(file)

theta_para=th_data['parameters']
theta_para.pop(0)
theta_para.pop(0)
theta_para.pop(0)
theta_para.pop(0)
theta_para.insert(0,'E_I_net')

theta=th_data['data']

with open('catch22/sim_X', 'rb') as file:
    features = pickle.load(file)

# featrues catsh 22
#eatures=np.load('features_1.npy')
features1=features[:,:]


features_name=['mode_5','mode_10','outlier_timing_pos','outlier_timing_neg','acf_timescale','acf_first_min','low_freq_power','centroid_freq','forecast_error','whiten_timescale','high_fluctuation','stretch_high','stretch_decreasing','entropy_pairs','ami2','trev','ami_timescale','transition_variance','periodicity','embedding_dist','rs_range','dfa']
features_name=features_name[:]

theta_data=np.zeros((theta.shape[0],4))
A=(theta[:,0] / theta[:,2]).reshape((-1, 1))
B=(theta[:,1] / theta[:,3]).reshape((-1, 1))
print((A/B).shape)
theta_data[:, 0] = (A / B).ravel()
theta_data[:,1] = theta[:,4]
theta_data[:,2] = theta[:,5]
theta_data[:,3] = theta[:,6]



### ZSCORE NORMALIZZATION ## only features 

scaler = StandardScaler()
features = scaler.fit_transform(features1)

## PRIOR 
# Definisci i limiti per ciascun parametro
param_limits = [     
    (0.0025,400),   #NET                   
   #(0.5, 5),   # Parametro 1:  
  # (0.5, 5),    # Parametro 2: 
  # (-40, -1),   # Parametro 3: 
 #  (-40, -1),   # Parametro 4: 
    (0.5, 5),    # Parametro 5: 
    (0,5, 8),    # Parametro 6: 
    (10, 50)     # Parametro 7: 
]

# Crea la distribuzione a priori con limiti diversi per ciascun parametro
prior = utils.BoxUniform(low=torch.tensor([x[0] for x in param_limits]),
                          high=torch.tensor([x[1] for x in param_limits]))

### instantiate the SBI object
density_estimator_building_fun= posterior_nn(
    model="maf", hidden_features=100, num_transforms=1
    #embedding_net = embedding_net
)

####################### split
# Inizializzazione dell'ambiente MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Assicuriamoci di avere esattamente 5 nodi

# Pool sample
Pool = np.random.randint(low=0, high=features.shape[0], size=sim)

# Initialize result arrays
IQR_T = np.zeros((theta_data.shape[1], features.shape[1] + 1, k_fold * rep))
CU_T = np.zeros((theta_data.shape[1], features.shape[1] + 1, k_fold * rep))
KL_T = np.zeros((theta_data.shape[1], features.shape[1] + 1, k_fold * rep))
corr_mat_T = np.zeros((int((len(theta_para) * (len(theta_para) - 1)) / 2), features.shape[1] + 1, k_fold * rep))
Pre_T = np.zeros((theta_data.shape[1], features.shape[1] + 1, k_fold * rep))
problem_T = [set() for i in range(k_fold * rep)]

feat = features[Pool, :]
theta = theta_data[Pool, :]

rkf = RepeatedKFold(n_splits=k_fold, n_repeats=rep, random_state=1)

# Split data
splits = list(rkf.split(feat))
local_results = {
    'IQR_T': np.zeros_like(IQR_T),
    'CU_T': np.zeros_like(CU_T),
    'KL_T': np.zeros_like(KL_T),
    'corr_mat_T': np.zeros_like(corr_mat_T),
    'Pre_T': np.zeros_like(Pre_T),
    'problem_T':  problem_T #np.zeros_like(problem_T)
}

# Process a subset of folds in each node
for i in range(rank, len(splits), size):
    train_index, test_index = splits[i]
    X_train, X_test = feat[train_index], feat[test_index]
    y_train, y_test = theta[train_index], theta[test_index]

    # Build the matrix
    Pool2 = np.random.randint(low=0, high=X_test.shape[0], size=TT)

    Iqr, Cu, Kl, Corr, Pre, corr_name, problem = matrix(X_test[Pool2,:], y_test[Pool2,:], N_distr, y_train, X_train, density_estimator_building_fun, prior, theta_para, param_limits)

    local_results['IQR_T'][:, :, i] = Iqr
    local_results['CU_T'][:, :, i] = Cu
    local_results['KL_T'][:, :, i] = Kl
    local_results['corr_mat_T'][:, :, i] = Corr
    local_results['Pre_T'][:, :, i] = Pre
    for elemento in problem:
        local_results['problem_T'][i].add(elemento)

    comm.barrier()

# Gather results from all nodes at the root node
gathered_results = comm.gather(local_results, root=0)

# Average the results across all nodes
if rank == 0:
    for i in range(1, size):
        IQR_T += gathered_results[i]['IQR_T']
        CU_T += gathered_results[i]['CU_T']
        KL_T += gathered_results[i]['KL_T']
        corr_mat_T += gathered_results[i]['corr_mat_T']
        Pre_T += gathered_results[i]['Pre_T']
        problem_T += gathered_results[i]['problem_T']
    
    # Averaging
    IQR_T=np.mean(IQR_T, axis=2)
    CU_T=np.mean(CU_T, axis=2)
    KL_T=np.mean(KL_T, axis=2)
    corr_mat_T=np.mean(corr_mat_T, axis=2)

    KL_T = np.nan_to_num(KL_T, nan=0)
    ##
    #normalitation within total features
    norm_iqr=IQR_T/IQR_T[:,0].reshape(-1, 1)
    norm_cu=CU_T/CU_T[:,0].reshape(-1, 1)

    #dataframe
    features_name.insert(0, 'all features')
    labels=features_name
    iqr_df= pd.DataFrame(norm_iqr, index=theta_para, columns=labels)
    cu_df= pd.DataFrame(norm_cu, index=theta_para, columns=labels)
    kl_df= pd.DataFrame(np.ceil(KL_T).astype(int), index=theta_para, columns=labels)
    corr_df=pd.DataFrame(corr_mat_T, index=corr_name, columns=labels)
    pre_df=pd.DataFrame(Pre_T, index=theta_para, columns=labels)

    ##
    # save
    #destination_direct = "/home/ale0021/project/graphs_F_seed_nodes"
    destination_direct = "/home/TIC117/cmg/thesis/graphs_NET_TOT"

    # Verifica se la cartella di destinazione esiste, altrimenti creala
    if not os.path.exists(destination_direct):
        os.makedirs(destination_direct)
    ##

    # Percorso completo del file in cui salvare la lista
    percorso_file = os.path.join(destination_direct, 'problem.pkl')

    # Serializza e salva la lista nel file
    with open(percorso_file, 'wb') as f:
        pickle.dump(problem_T, f)


    #plot IQR
    plt.figure(figsize=(12, 8))
    sns.heatmap(iqr_df, annot=True, cmap='coolwarm', fmt=".2f")
    #plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.21)
    plt.title('Interquartile range (IQR)')
    plt.tight_layout()
    plt.xticks(rotation=45)
    # Salvataggio del grafico nella cartella di destinazione
    pwd = os.path.join(destination_direct, "IQR.png")
    plt.savefig(pwd)
    #plt.show()

    #plot PRE
    plt.figure(figsize=(12, 8))
    sns.heatmap(pre_df, annot=True, cmap='coolwarm', fmt=".2f")
    #plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.21)
    plt.title('Absolut Parameter recovery error (PRE)')
    plt.tight_layout()
    #plt.xticks(rotation=45)
    # Salvataggio del grafico nella cartella di destinazione
    pwd = os.path.join(destination_direct, "PRE.png")
    plt.savefig(pwd)
    #plt.show()


    # CURTOSI
    plt.figure(figsize=(12, 8))
    sns.heatmap(cu_df, annot=True, cmap='coolwarm', fmt=".2f")
    #plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.21)
    plt.title('Curtosi')
    plt.tight_layout()
    #plt.xticks(rotation=45)
    # Salvataggio del grafico nella cartella di destinazione
    pwd = os.path.join(destination_direct, "CU.png")
    plt.savefig(pwd)
    #plt.show()

    #KL
    plt.figure(figsize=(12, 8))
    sns.heatmap(kl_df, annot=True, cmap='coolwarm', fmt="d")
    #plt.subplots_adjust(left=0.1, right=1, top=0.9, bottom=0.21)
    plt.title('Kullback-Leibler divergence (KL)')
    plt.tight_layout()
    #plt.xticks(rotation=45)
    # Salvataggio del grafico nella cartella di destinazione
    pwd = os.path.join(destination_direct, "KL.png")
    plt.savefig(pwd)
    #plt.show()

    #plot corr
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f")
    #plt.subplots_adjust(left=0.17, right=1, top=0.9, bottom=0.21)
    plt.title('Correlation')
    plt.tight_layout()
    # Salvataggio del grafico nella cartella di destinazione
    pwd = os.path.join(destination_direct, "correlation.png")
    plt.savefig(pwd)
    #plt.show()
