import os
import pickle
import random
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

from sbi.inference.base import infer
from sbi.utils.get_nn_models import posterior_nn
from sbi.inference import SNPE, prepare_for_sbi
#from sbi import utils
from sbi import analysis as analysis
#import seaborn as sns
import matplotlib.pyplot as plt
import shap 
from shap._explanation import Explanation
import signal

####### INTRODUCTION ########
#We used the SHAP method to analyze the influence of each feature on individual parameters. 
#To ensure result consistency, we applied k-fold validation, reducing the risk of overfitting and improving the reliability of the estimates. 
#Given the dataset size, all computations were performed on a university server, providing the necessary computational resources for an in-depth analysis.
###############


#################
# function 
def handler(signum, frame):
    raise TimeoutError("Il comando ha impiegato troppo tempo per eseguirsi")

# management
signal.signal(signal.SIGALRM, handler)

#timeout
###########################
###
### remove features############
def remove_columns(matrix, columns_to_remove, column_names):
    # Trova gli indici delle colonne da eliminare
    indices_to_remove = [col for col in columns_to_remove]
    
    # Elimina le colonne dalla matrice
    reduced_matrix = np.delete(matrix, indices_to_remove, axis=1)
    
    # Crea una nuova lista di nomi di colonne senza i nomi corrispondenti
    reduced_column_names = [name for i, name in enumerate(column_names) if i not in indices_to_remove]
    
    return reduced_matrix, reduced_column_names
## end remove features ##############

##### START CODE #################################à
sim=948819  # Pool dimension 
#dim=0.1 #dimesione test
TT=100 #dimesione shap

N_distr=2000 # ( number of sample distribution )
k_fold=10
rep=2

seed=1  
#seed_list=[634, 234, 567, 79, 123, 789, 12, 456, 654, 321]

### seed
#random.seed(57)
#torch.manual_seed(57)
#np.random.seed(57)

# this will be performe for each parameter 
parameter= 0 #4 #5 #6

with open('catch22/sim_theta', 'rb') as file: #catch22/
    th_data = pickle.load(file)

#theta_para=th_data['parameters']

theta_para=['E_I_net']

theta=th_data['data']

with open('catch22/sim_X', 'rb') as file:
    features = pickle.load(file)

features_name=['mode_5','mode_10','outlier_timing_pos','outlier_timing_neg','acf_timescale','acf_first_min','low_freq_power','centroid_freq','forecast_error','whiten_timescale','high_fluctuation','stretch_high','stretch_decreasing','entropy_pairs','ami2','trev','ami_timescale','transition_variance','periodicity','embedding_dist','rs_range','dfa']
#theta_data = theta[:, parameter].reshape((-1, 1))

A=(theta[:,0] / theta[:,2]).reshape((-1, 1))
B=(theta[:,1] / theta[:,3]).reshape((-1, 1))
theta_data = A/B
print(theta_data.shape)

### remove the features detect on IQR with fewatures removal method(mat.py)

remove=[]
### remove featuers
features1,features_name=remove_columns(features,remove,features_name)

### ZSCORE## NORMALIZZATION#

scaler = StandardScaler()
features = scaler.fit_transform(features1)


# SAVE
destination_direct = "/home/ale0021/project/Shape_TF"
#destination_direct = "/home/TIC117/cmg/thesis/Shape"

# Verifica se la cartella di destinazione esiste, altrimenti creala
if not os.path.exists(destination_direct):
    os.makedirs(destination_direct)
##
# Definizione del modello SBI
density_estimator_building_fun= posterior_nn(
    model="maf", hidden_features=80, num_transforms=2
    #embedding_net = embedding_net
)

inference = SNPE(prior=None, density_estimator=density_estimator_building_fun)


# Function to detect important features for each parameter
def f(X, timeout_sec=10):     
    j = 0
    rem = []
    post_array = np.zeros(2000)
    m = np.zeros((X.shape[0], 1))
    
    for idx in range(X.shape[0]):
        try:
            # Set an alarm that triggers after timeout_sec seconds
            signal.alarm(timeout_sec)
            
            sample_ramdom = torch.from_numpy(np.array(X[idx], dtype=np.float32))
            post_array = (posterior.sample((2000,), sample_ramdom)).numpy()
            #print(post_array.dtype)
        except TimeoutError:
            rem.append(j)
        finally:
            # Disable the alarm
            signal.alarm(0)

        # Compute the distribution mean
        m[j, :] = np.mean(post_array, axis=0)
        j += 1
    

    return np.delete(m, rem, axis=0)


Pool=np.random.randint(low=0, high=features.shape[0], size=sim)
feat=features[Pool,:]
theta=theta_data[Pool,:]

rkf = RepeatedKFold(n_splits=k_fold, n_repeats=rep, random_state=1)


SV=np.zeros((TT,feat.shape[1],int(k_fold*rep)))
B=np.zeros((TT,k_fold*rep))
D=np.zeros((TT,feat.shape[1],k_fold*rep))

# Iterate through the splits
n=0
for train_index, test_index in rkf.split(feat):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = feat[train_index], feat[test_index]
    y_train, y_test = theta[train_index], theta[test_index]
    #print(X_train)
    #print(X_test)

    # train
    inference = inference.append_simulations(
            torch.from_numpy(np.float32(y_train)),
            torch.from_numpy(np.float32(X_train)))

    density_estimator = inference.train(force_first_round_loss=True)
    posterior = inference.build_posterior(density_estimator)

   #building matrix 
    Pool2=np.random.randint(low=0, high=X_test.shape[0], size=TT)
    exp=shap.Explainer(f,X_train,feature_names=features_name,seed=seed) #,feature_names=features_name)
    shap_values =  exp(X_test[Pool2,:]) #, nsamples=100) X_test[:500,:]
    

    SV[:,:,n]=shap_values.values
    B[:,n]=shap_values.base_values.flatten()
    D[:,:,n]=shap_values.data
    print(f'Fold: {n}')
    n=n+1
    ###### print 


SV_m=np.mean(SV,axis=2)
B_m=np.mean(B,axis=1)
D_m=np.mean(D,axis=2)

print(SV_m.shape)
print(B_m.shape)
print(D_m.shape)

explanation = Explanation(values=SV_m, base_values=B_m, data=D_m)
#print(shap_values.values)
#print(shap_values.base_values)
#print(shap_values.data)


# Wrap SHAP values in an Explanation object
explanation = Explanation(SV_m, base_values=B_m, data=D_m, feature_names=features_name)



# Plotting the SHAP values
fig, ax = plt.subplots()
shap.plots.beeswarm(explanation, max_display=22, show=False)#order=np.mean(np.abs(shap_values)
fig.subplots_adjust(left=0.3, right=1, top=0.9, bottom=0.21)
ax.set_title(f'Features importats for parameter: {theta_para[parameter]} features')
plt.savefig(os.path.join(destination_direct, f"swarm{theta_para[parameter]}.png"))

#plt.figure(figsize=(10, 12))
fig1, ax = plt.subplots()
shap.plots.bar(explanation, max_display=22, show=False)
ax.set_title(f'Features importants for parameter:{theta_para[parameter]} features')
plt.tight_layout()
#plt.subplots_adjust(left=0.30, right=1, top=0.9, bottom=0.21)
pwd = os.path.join(destination_direct, f"bar{theta_para[parameter]}.png")
#plt.show()
plt.savefig(pwd)


'''
# Assuming features, theta_data, features_name, and destination_direct are defined
# Assuming features, theta_data, features_name, and destination_direct are defined
Pool=np.random.randint(low=0, high=features.shape[0], size=sim)
#print(Pool)
X_train, X_test, y_train, y_test = train_test_split(features[Pool,:], theta_data[Pool,:], train_size=1-dim, test_size=dim, random_state=seed_list[seed])

print(X_train.shape)
print(X_test.shape)

# train
inference = inference.append_simulations(
            torch.from_numpy(np.float32(y_train)),
            torch.from_numpy(np.float32(X_train)))

density_estimator = inference.train()
posterior = inference.build_posterior(density_estimator)

Pool2=np.random.randint(low=0, high=X_test.shape[0], size=TT)


exp=shap.Explainer(f,X_train,feature_names=features_name,seed=seed) #,feature_names=features_name)
shap_values =  exp(X_test[Pool2,:]) #, nsamples=100) X_test[:500,:]

plt.figure(figsize=(10, 12))
fig, ax = plt.subplots()
shap.plots.beeswarm(shap_values,max_display=22,order=shap_values.abs.mean(0), show=False)
fig.subplots_adjust(left=0.3, right=1, top=0.9, bottom=0.21)
ax.set_title(f'Features importats for parameter: {theta_para[parameter]} features')
#plt.tight_layout()
pwd = os.path.join(destination_direct, f"swarm{theta_para[parameter]}:{seed}.png")
plt.savefig(pwd)


#plt.figure(figsize=(10, 12))
fig1, ax = plt.subplots()
shap.plots.bar(shap_values, max_display=22, show=False)
ax.set_title(f'Features importants for parameter:{theta_para[parameter]} features')
plt.tight_layout()
#plt.subplots_adjust(left=0.30, right=1, top=0.9, bottom=0.21)
pwd = os.path.join(destination_direct, f"bar{theta_para[parameter]}:{seed}.png")
#plt.show()
plt.savefig(pwd)

###########
'''