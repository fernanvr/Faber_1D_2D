import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from auxiliar_functions import *
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eig
from sympy import *
from time import time
import math


def matrix_H(dim,equ,param,k,beta):

    if dim==1:
        if equ=='scalar':
            H=np.array([[0,param*1j*k,-param],[1j*k,-beta,0],[0,beta*1j*k,-beta]])
        elif equ=='scalar_dx2':
            H=np.array([[0,1,0],[-param*k**2,-beta,param*1j*k],[-beta*1j*k,0,-beta]])
    elif dim==2:
        if equ=='scalar':
            H=np.array([[0,param*1j*k[0],param*1j*k[1],-param,-param],
                [1j*k[0],-beta[0],0,0,0],
                [1j*k[1],0,-beta[1],0,0],
                [0,beta[0]*1j*k[0],0,-beta[0],0],
                [0,0,beta[1]*1j*k[1],0,-beta[1]]])
        elif equ=='scalar_dx2':
            H=np.array([[0,1,0,0],
                [-beta[0]*beta[1]-param*(k[0]**2+k[1]**2),-(beta[0]+beta[1]),param*1j*k[0],param*1j*k[1]],
                [-(beta[0]-beta[1])*1j*k[0],0,-beta[0],0],
                [-(beta[1]-beta[0])*1j*k[1],0,0,-beta[1]]])
        elif equ=='elastic':
            H=np.array([[0,0,1,0,0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0,0,0,0],
                        [-beta[0]*beta[1],0,-(beta[0]+beta[1]),0,1/param[0]*1j*k[0],1/param[0]*1j*k[1],0,1/param[0]*1j*k[0],1/param[0]*1j*k[1],0,0],
                        [0,-beta[0]*beta[1],0,-(beta[0]+beta[1]),0,1/param[0]*1j*k[0],1/param[0]*1j*k[1],0,0,1/param[0]*1j*k[0],1/param[0]*1j*k[1]],
                        [0,0,(2*param[1]+param[2])*1j*k[0],param[2]*1j*k[1],0,0,0,0,0,0,0],
                        [0,0,param[1]*1j*k[1],param[1]*1j*k[0],0,0,0,0,0,0,0],
                        [0,0,param[2]*1j*k[0],(2*param[1]+param[2])*1j*k[1],0,0,0,0,0,0,0],
                        [(beta[1]-beta[0])*(2*param[1]+param[2])*1j*k[0],0,0,0,0,0,0,-beta[0],0,0,0],
                        [(beta[0]-beta[1])*param[1]*1j*k[1],0,0,0,0,0,0,0,-beta[1],0,0],
                        [0,(beta[1]-beta[0])*param[1]*1j*k[0],0,0,0,0,0,0,0,-beta[0],0],
                        [0,(beta[0]-beta[1])*(2*param[1]+param[2])*1j*k[1],0,0,0,0,0,0,0,0,-beta[1]]])
    aux=eigs(H)[0]
    return np.hstack((np.expand_dims(aux.real,axis=1),np.expand_dims(aux.imag,axis=1)))


def eigen_map_H(dim,equ,param,lim_beta,lim_k):

    if dim==1:
        beta=np.linspace(0,lim_beta,50)
        k=np.linspace(-lim_k,lim_k,100)
    elif dim==2:
        beta=np.zeros((88,2))
        beta[:10,0]=lim_beta
        beta[:10,1]=np.linspace(0,lim_beta,10)
        beta[10:20,1]=beta[:10,1]
        beta[20:30,0]=beta[:10,1]
        beta[20:30,1]=lim_beta
        beta[30:40,0]=beta[:10,1]

        beta[40:47,0]=lim_beta*2/3
        beta[40:47,1]=np.linspace(0,lim_beta*2/3,7)
        beta[47:54,1]=beta[40:47,1]
        beta[54:61,0]=beta[40:47,1]
        beta[54:61,1]=lim_beta*2/3
        beta[61:68,0]=beta[40:47,1]

        beta[68:73,0]=lim_beta*1/3
        beta[68:73,1]=np.linspace(0,lim_beta*1/3,5)
        beta[73:78,1]=beta[68:73,1]
        beta[78:83,0]=beta[68:73,1]
        beta[78:83,1]=lim_beta*1/3
        beta[83:88,0]=beta[68:73,1]

        k=np.zeros((141,2))
        k[:20,0]=lim_k
        k[:20,1]=np.linspace(-lim_k,lim_k,20)
        k[20:40,0]=-lim_k
        k[20:40,1]=k[:20,1]
        k[40:60,0]=k[:20,1]
        k[40:60,1]=lim_k
        k[60:80,0]=k[:20,1]
        k[60:80,1]=-lim_k

        k[80:90,0]=lim_k*2/3
        k[80:90,1]=np.linspace(-lim_k,lim_k,10)*2/3
        k[90:100,0]=-lim_k*2/3
        k[90:100,1]=k[80:90,1]
        k[100:110,0]=k[80:90,1]
        k[100:110,1]=lim_k*2/3
        k[110:120,0]=k[80:90,1]
        k[110:120,1]=-lim_k*2/3

        k[120:125,0]=lim_k*1/3
        k[120:125,1]=np.linspace(-lim_k,lim_k,5)*1/3
        k[125:130,0]=-lim_k*1/3
        k[125:130,1]=k[120:125,1]
        k[130:135,0]=k[120:125,1]
        k[130:135,1]=lim_k*1/3
        k[135:140,0]=k[120:125,1]
        k[135:140,1]=-lim_k*1/3

    eigen_values=np.zeros((0,2))
    for i in range(len(beta)):
        for j in range(len(k)):
            eigen_values=np.vstack((eigen_values,matrix_H(dim,equ,param,k[j],beta[i])))

    np.save('eigenvalues/eigen_continuum_dim'+str(dim)+'_equ_'+equ+'_lim_k_'+str(lim_k),eigen_values)
    np.load('eigenvalues/eigen_continuum_dim'+str(dim)+'_equ_'+equ+'_lim_k_'+str(lim_k)+'.npy')
    plt.scatter(eigen_values[:,0],eigen_values[:,1],alpha=0.7)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.savefig('eigenvalues/eigen_continuum_dim'+str(dim)+'_equ_'+equ+'_lim_k_'+str(lim_k)+'.pdf')
    plt.show()


eigen_map_H(dim=2,equ='elastic',param=np.array([1,1.5,8]),lim_beta=30,lim_k=20)
eigen_map_H(dim=2,equ='scalar',param=np.array([11]),lim_beta=30,lim_k=20)
eigen_map_H(dim=2,equ='scalar_dx2',param=np.array([11]),lim_beta=30,lim_k=20)
eigen_map_H(dim=1,equ='scalar',param=np.array([11]),lim_beta=30,lim_k=20)
eigen_map_H(dim=1,equ='scalar_dx2',param=np.array([11]),lim_beta=30,lim_k=20)
