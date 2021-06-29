import numpy as np
from numba import jit
import multiprocessing as mp

def h_list_to_dNall(t0,t1,dt,h_list):
    Tstep=np.arange(t0,t1,dt)
    dNall=np.zeros((len(h_list),Tstep.size))
    for j in range(0,len(h_list)):
        h1=np.array(h_list[j])
        t=t0
        for i in range(0,Tstep.size):
            h2=h1[(h1>=t) & (h1<t+dt)]
            dNall[j,i]=h2.size 
            t=t+dt
    return dNall

#@jit
def analysis_lamb(lb,dt,yo,nsamp):
    lbf=lb*dt
    lbdt=lb*dt
    lb_bar=lbdt.mean()
    A=lbdt-lb_bar
    #A *= 1.001
    Vf=np.matmul(A,A.T)/(nsamp-1)
    lb_bar_a=lb_bar+(yo-lb_bar)*Vf/(lb_bar+Vf)
    K1=A/lb_bar
    if yo==0:
        tmp_var=1
        K2=K1
    else:
        lb_tmp=np.random.gamma(yo,1,nsamp)
        lb_tmp_bar=lb_tmp.mean()
        K2=lb_tmp-lb_tmp_bar
        K2 /= lb_tmp_bar
        Vfr=Vf/lb_bar**2
        tmp_var=Vfr/(Vfr+1./yo)
    A_a = K1 + tmp_var*(K2-K1)
    A_a *= lb_bar_a
    lb_bar_a += A_a
    lamb = lb_bar_a/dt
    kk=(lb_bar_a-lbf)/Vf
    return lamb,kk,A

#@jit
def update_par(p,A,kk,nsamp):
    Ak = p-p.mean()
    Cf=np.matmul(A,Ak.T)/(nsamp-1)
    p +=Cf*kk     
    return p

#@jit(nopython=True)
def EnPGF_kernel(c,dNall,dt,Ncells,Nsamp,Nstep):
    print(c)
    tmp = np.random.gamma(1.5,1,[Nsamp])
    mu1_l = np.log(tmp.copy()) 
    lamb1 = tmp.copy()
    beta1_l = np.log(np.random.normal(3,0.5,[Nsamp]))
    alpha1_l=np.log(np.random.gamma(1,0.1,[Ncells,Nsamp]))
    mlamb=np.empty([Nstep])
    alpha_e=np.empty([Ncells,Nstep])
    mu_e=np.empty([Nstep])
    beta_e=np.empty([Nstep])
    for i in range(Nstep):
        # analysis 
        lamb1,kk,A=analysis_lamb(lamb1,dt,dNall[i,c],Nsamp)
        # update paramters
        mu1_l=update_par(mu1_l,A,kk,Nsamp)
        beta1_l=update_par(beta1_l,A,kk,Nsamp)    
        for cc in range(Ncells):
            alpha1_l[cc,:]=update_par(alpha1_l[cc,:],A,kk,Nsamp)
        lamb1=np.exp(mu1_l)+(lamb1-np.exp(mu1_l))*(1-np.exp(beta1_l)*dt)+np.matmul(dNall[i,:],np.exp(alpha1_l))
        mlamb[i]=lamb1.mean()
        alpha_e[:,i]=np.mean(np.exp(alpha1_l),1)
        beta_e[i]=np.mean(np.exp(beta1_l))
        mu_e[i]=np.mean(np.exp(mu1_l))
    return c,alpha_e,mu_e,beta_e,mlamb

def EnPGF_exp_det_mp(dNall,dt,Ncells,Nsamp,Nstep,Npool):
    lamb_e=np.empty([Ncells,Nstep])
    mu_e=np.empty([Ncells,Nstep])
    alpha_e=np.empty([Ncells,Ncells,Nstep])
    beta_e=np.empty([Ncells,Nstep])
    pool = mp.Pool(Npool)
    results = pool.starmap(EnPGF_kernel, [(c,dNall,dt,Ncells,Nsamp,Nstep) for c in range(Ncells)])
    pool.close()    
    for c in range(Ncells):
        alpha_e[results[c][0],:,:]=results[c][1]
        mu_e[results[c][0],:]=results[c][2]
        beta_e[results[c][0],:]=results[c][3]
        lamb_e[results[c][0],:]=results[c][4]
    return alpha_e,mu_e,beta_e,lamb_e

def EnPGF_exp_det_mp_final_ens(dNall,dt,Ncells,Nsamp,Nstep,Npool):
    lamb_ens=np.empty([Ncells,Nsamp])
    mu_ens=np.empty([Ncells,Nsamp])
    alpha_ens=np.empty([Ncells,Ncells,Nsamp])
    beta_ens=np.empty([Ncells,Nsamp])
    pool = mp.Pool(Npool)
    results = pool.starmap(EnPGF_kernel_final_ens, [(c,dNall,dt,Ncells,Nsamp,Nstep) for c in range(Ncells)])
    pool.close()    
    for c in range(Ncells):
        alpha_ens[results[c][0],:,:]=results[c][1]
        mu_ens[results[c][0],:]=results[c][2]
        beta_ens[results[c][0],:]=results[c][3]
        lamb_ens[results[c][0],:]=results[c][4]
    return alpha_ens,mu_ens,beta_ens,lamb_ens

def EnPGF_kernel_final_ens(c,dNall,dt,Ncells,Nsamp,Nstep):
    print(c)
    tmp = np.random.gamma(1.5,1,[Nsamp])
    mu1_l = np.log(tmp.copy()) 
    lamb1 = tmp.copy()
    beta1_l = np.log(np.random.normal(3,0.5,[Nsamp]))
    alpha1_l=np.log(np.random.gamma(1,0.1,[Ncells,Nsamp]))
    for i in range(Nstep):
        # analysis 
        lamb1,kk,A=analysis_lamb(lamb1,dt,dNall[i,c],Nsamp)
        # update paramters
        mu1_l=update_par(mu1_l,A,kk,Nsamp)
        beta1_l=update_par(beta1_l,A,kk,Nsamp)    
        for cc in range(Ncells):
            alpha1_l[cc,:]=update_par(alpha1_l[cc,:],A,kk,Nsamp)
        lamb1=np.exp(mu1_l)+(lamb1-np.exp(mu1_l))*(1-np.exp(beta1_l)*dt)+np.matmul(dNall[i,:],np.exp(alpha1_l))
    return c,alpha1_l,mu1_l,beta1_l,lamb1
