# run_EnPGF_IkeNet.py

#import matplotlib.pyplot as plt
from mod_EnPGF_IkeNet import h_list_to_dNall,EnPGF_exp_det_mp,EnPGF_exp_det_mp_final_ens
#from mod_EnPGF_IkeNet_numba import h_list_to_dNall,EnPGF_exp_det_mp,EnPGF_exp_det_mp_final_ens
import pandas as pd
import numpy as np
import pickle

###################
# load data
df_final=pd.read_csv('IkeNet_cleaned.csv',index_col=0,header=0)
tmp=np.empty([len(df_final)])
for i in range(len(df_final)):
    tmp[i]=int(df_final['sender'][i][3:5])
df_sender=pd.DataFrame(columns=['sender','time'])
df_sender.sender=tmp-1
df_sender.time=df_final.time

###################
# set up parameters and prepare data
tfin=8620
dt=0.1
Nsamp=100
fname='EnPGF_IkeNet_sender.pkl'

Nstep=tfin*dt
h_list=[[] for i in range(22)]
for i in range(len(df_sender)):
    h_list[int(df_sender['sender'][i])].append(df_sender['time'][i])
dNall=np.transpose(h_list_to_dNall(0,tfin,dt,h_list))

###################
# run EnPGF
###################
# to get mean estimates at each time step
alpha_mp1,mu_mp1,beta_mp1,lamb1 = EnPGF_exp_det_mp(dNall,dt,22,Nsamp,len(dNall),6)
# to get final EnPGF ensemble uncomment the following line
#alpha_mp1,mu_mp1,beta_mp1,lamb1 = EnPGF_exp_det_mp_final_ens(dNall,dt,22,Nsamp,len(dNall),6)

###################
# save results
output={ 
           "dNall" : dNall,
           "dt" : dt,
           "Nstep" : Nstep,
           "alpha" : alpha_mp1,
           "mu" : mu_mp1,
           "beta" : beta_mp1,
           "lamb" : lamb1
           }
outfile=open(fname,'wb')
pickle.dump(output,outfile)
outfile.close()

