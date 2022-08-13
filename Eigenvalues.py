import numpy as np
import matplotlib.pyplot as plt
import operators as op
import sys
import os
import Methods as meth
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from time import time


def limit_eigen(dx,equ,dim,delta,beta0,ord,T,Ndt,example):
    # computing 4 eigenvalues for different values of dx with: the smallest, the largest real part,
    # the largest imaginary part, and the largest magnitude

    # INPUT:
    # dx: spatial discretization step size (float)
    # equ: equation formulation (scalar, scalar_dx2, elastic)
    # dim: dimension of the problem (1, 2)
    # delta: thickness of the PML domain (float)
    # beta0: damping parameter of the PML layer (float)
    # ord: spatial discretization order ('4','8')
    # T: end time of the simulation (float)
    # Ndt: amount of time-step sizes used to compute the solutions (int)
    # example: selection of the wave equation parameters (string)

    # array of the four different eigenvalues to compute
    eigen_sr=np.zeros(len(dx))
    eigen_lr=np.zeros(len(dx))
    eigen_li=np.zeros(len(dx))
    eigen_lm=np.zeros(len(dx))

    # cycle to compute the eigenvalues for each dx
    for i in range(len(dx)):
        print('i: ',i)  # to know the iteration

        # Model parameters
        nx,ny,X,Y,param,f,param_ricker,Dt,NDt,points,source_type,var0=meth.domain_source(dx[i],T,Ndt,dim,equ,example,ord,delta)

        # constructing the linear operator
        def lin_op_H(var):
            var=np.expand_dims(var,axis=1)
            return op.op_H(var,equ,dim,delta,beta0,ord,dx[i],param,nx+1,ny+1)
        H=LinearOperator(shape=(len(var0),len(var0)),matvec=lin_op_H)

        start=time() # start counting time of computing the eigenvalues
        try:
            vals_sr,vecs_sr=eigs(H,1,ncv=np.min(np.array([1000,len(var0)-1])),which='SR')
            vals_lr,vecs_lr=eigs(H,1,ncv=np.min(np.array([1000,len(var0)-1])),which='LR')
            vals_li,vecs_li=eigs(H,1,ncv=np.min(np.array([1000,len(var0)-1])),which='LI')
            vals_lm,vecs_lm=eigs(H,1,ncv=np.min(np.array([1000,len(var0)-1])),which='LM')
        except:
            print('I knew it was not going to work!!!')
            break
        end=time() # end counting time of computing the eigenvalues
        print('time: ',end-start)

        eigen_sr[i]=vals_sr.real
        eigen_lr[i]=vals_lr.real
        eigen_li[i]=vals_li.imag
        eigen_lm[i]=vals_lm[0]

        if(end-start>40000):    # condition to stop if the time to compute the eigenvalues is too big
            break

    np.save('eigenvalues/eigen_sr_equ_'+equ+'_ord_'+str(ord)+'_example_'+example,eigen_sr)
    np.save('eigenvalues/eigen_lr_equ_'+equ+'_ord_'+str(ord)+'_example_'+example,eigen_lr)
    np.save('eigenvalues/eigen_li_equ_'+equ+'_ord_'+str(ord)+'_example_'+example,eigen_li)
    np.save('eigenvalues/eigen_lm_equ_'+equ+'_ord_'+str(ord)+'_example_'+example,eigen_lm)


def graph_limit_eigen(dx,equ,delta,beta0,ord,example,ind):
    # graphics of the eigenvalues limits in the real and imaginary parts

    # constructing the label for the different lines
    def label_fun(example,ord,equ):
        label_str=''
        if example=='1D_homogeneous_0':
            label_str=label_str+'Test_1_'
        elif example=='1D_heterogeneous_1a':
            label_str=label_str+'Test_2_'
        elif example=='1D_heterogeneous_2':
            label_str=label_str+'Test_3_'
        elif example=='2D_homogeneous_0a':
            label_str=label_str+'Test_4_'
        elif example=='2D_heterogeneous_3a':
            label_str=label_str+'Test_5_'
        elif example=='2D_heterogeneous_2':
            label_str=label_str+'Test_6_'
        elif example=='2D_heterogeneous_3':
            label_str=label_str+'Test_7_'
        label_str=label_str+'ord'+ord
        if equ=='scalar':
            label_str=label_str+'_1SD'
        elif equ=='scalar_dx2':
            label_str=label_str+'_2SD'

        return label_str

    # repetitive line codes for graphic parameters
    def graph_block():
        plt.legend(fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel('$1/\Delta x$',fontsize=22)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)

    # cycle for the eigenvalue with the smallest real part -----------------------------------------------------------
    for i in range(len(example)):
        eigen_sr=np.load('eigenvalues/eigen_sr_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy')
        index_sr=np.max(np.arange(len(eigen_sr))[np.abs(eigen_sr)>pow(10,-16)])

        plt.plot((1/dx)[:index_sr],eigen_sr[:index_sr],linewidth=2,label=label_fun(example[i],ord[i],equ[i]))

    plt.plot((1/dx)[:index_sr],-beta0*pow((delta-dx[:index_sr]/2)/delta,2),color='r',linestyle='--',linewidth=2)
    plt.ylabel('Real inferior limit',fontsize=22)
    graph_block()
    plt.savefig('eigenvalues/eigen_experiments_sr_'+ind+'.pdf')
    plt.show()

    # cycle for the eigenvalue with the largest real part ------------------------------------------------------------
    for i in range(len(example)):
        eigen_lr=np.load('eigenvalues/eigen_lr_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy')
        index_lr=np.max(np.arange(len(eigen_lr))[np.abs(eigen_lr)>pow(10,-16)])

        plt.plot((1/dx)[:index_lr],eigen_lr[:index_lr],linewidth=2,label=label_fun(example[i],ord[i],equ[i]))

    plt.ylabel('Real superior limit',fontsize=22)
    graph_block()
    plt.savefig('eigenvalues/eigen_experiments_lr_'+ind+'.pdf')
    plt.show()

    # cycle for the eigenvalue with the largest imaginary part -------------------------------------------------------
    for i in range(len(example)):
        eigen_li=np.load('eigenvalues/eigen_li_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy')
        index_li=np.max(np.arange(len(eigen_li))[eigen_li!=0])

        plt.plot((1/dx)[:index_li],eigen_li[:index_li],linewidth=2,label=label_fun(example[i],ord[i],equ[i]))

    plt.ylabel('Imaginary limit',fontsize=22)
    graph_block()
    plt.savefig('eigenvalues/eigen_experiments_li_'+ind+'.pdf')
    plt.show()

    # cycle for the eigenvalue with the slope of the largest imaginary part ------------------------------------------
    for i in range(len(example)):
        eigen_li=np.load('eigenvalues/eigen_li_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy')
        index_li=np.max(np.arange(len(eigen_li))[eigen_li!=0])

        plt.plot((1/dx)[:index_li-1],(eigen_li[1:index_li]-eigen_li[:index_li-1])/((1/dx)[1:index_li]-(1/dx)[:index_li-1]),linewidth=2,label=label_fun(example[i],ord[i],equ[i]))

    plt.ylabel('Slope',fontsize=22)
    graph_block()
    plt.savefig('eigenvalues/eigen_experiments_li_slope_'+ind+'.pdf')
    plt.show()

    # cycle for the real part of the eigenvalue with the largest magnitude -------------------------------------------
    for i in range(len(example)):
        eigen_lm=np.load('eigenvalues/eigen_lm_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy').real
        index_lm=np.max(np.arange(len(eigen_lm))[np.abs(eigen_lm)>pow(10,-16)])

        plt.plot((1/dx)[:index_lm],eigen_lm[:index_lm],linewidth=2,label=label_fun(example[i],ord[i],equ[i]))

    plt.ylabel('Real part',fontsize=22)
    graph_block()
    plt.savefig('eigenvalues/eigen_experiments_lm_real_'+ind+'.pdf')
    plt.show()

    # cycle for the imaginary part of the eigenvalue with the largest magnitude --------------------------------------
    for i in range(len(example)):
        eigen_lm=np.load('eigenvalues/eigen_lm_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy').imag
        index_lm=np.max(np.arange(len(eigen_lm))[np.abs(eigen_lm)>pow(10,-16)])

        plt.plot((1/dx)[:index_lm],eigen_lm[:index_lm],linewidth=2,label=label_fun(example[i],ord[i],equ[i]))

    plt.ylabel('Imaginary part',fontsize=22)
    graph_block()
    plt.savefig('eigenvalues/eigen_experiments_lm_imag_'+ind+'.pdf')
    plt.show()


def eigen_full(dx,equ,dim,delta,beta0,ord,T,Ndt,example):
    # computing full eigenspectrum of matrix H for different values of dx

    # cycle to compute the eigenvalues for each dx
    for i in range(len(dx)):
        print('i: ',i)  # to know the iteration

        # Model parameters
        nx,ny,X,Y,param,f,param_ricker,Dt,NDt,points,source_type,var0=meth.domain_source(dx[i],T,Ndt,dim,equ,example,ord,delta)

        # constructing the linear operator
        def lin_op_H(var):
            var=np.expand_dims(var,axis=1)
            return op.op_H(var,equ,dim,delta,beta0,ord,dx[i],param,nx+1,ny+1)
        H=LinearOperator(shape=(len(var0),len(var0)),matvec=lin_op_H)

        start=time() # start counting time of computing the eigenvalues

        vals,vecs=eigs(H,len(var0)-2)

        end=time() # end counting time of computing the eigenvalues
        print('time: ',end-start)

        np.save('eigenvalues/eigen_full_i_'+str(i)+'_equ_'+equ+'_ord_'+str(ord)+'_example_'+example,vals)

        if(end-start>20000):    # condition to stop if the time to compute the eigenvalues is too big
            break


def graph_eigen_full(dx,equ,dim,ord,example):
    # graphics of the full eigenspectrum of the operator H

    # cycle to plot the eigenspectrum for each dx
    for i in range(len(dx)):
        vals=np.load('eigenvalues/eigen_all_i_'+str(i)+'_equ_'+equ+'_ord_'+str(ord)+'_example_'+example+'.npy')

        # print some properties of 'vals'
        print('i: ',i)
        print('Dimensions: ',vals.shape)
        print('minimum real part: ',np.min(vals.real))
        print('maximum real part: ',np.max(vals.real))
        print('maximum imaginary part: ',np.max(vals.imag))

        # graphic parameters
        plt.scatter(vals.real,vals.imag,alpha=0.7)
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig('eigenvalues/eigenval_equ_'+equ+'_dim_'+str(dim)+'_ord_'+ord+'_i_'+str(i)+'.pdf')
        plt.show()


# cheking if there exist the paste to save the results, and creating one if there is not
if not os.path.isdir('eigenvalues/'):
    os.mkdir('eigenvalues')


# ------------------------------------------------------------------------------------
# Code for the computation of the spectrum of H and the eigenvalues limits, according to the results of the paper
# ------------------------------------------------------------------------------------

# param=np.array([['scalar',1,'4','1D_heterogeneous_2'],['scalar_dx2',1,'4','1D_heterogeneous_2'],['scalar',2,'4','2D_heterogeneous_3'],
#                 ['scalar_dx2',2,'4','2D_heterogeneous_3'],['elastic',2,'4','2D_heterogeneous_3'],['scalar',1,'8','1D_heterogeneous_2'],
#                 ['scalar_dx2',1,'8','1D_heterogeneous_2'],['scalar',2,'8','2D_heterogeneous_3'],['scalar_dx2',2,'8','2D_heterogeneous_3'],
#                 ['elastic',2,'8','2D_heterogeneous_3']])
#
# for i in range(len(param)):
#
#     # condition to adjust dx for the different dimensions
#     if int(param[i,1])==1:
#         dx=10.5/np.array([100,500,1000,5000])  # for the all spectrum
#         dx=10.5/(10*np.arange(3,500))          # for the limit eigenvalues
#     else:
#         dx=8/np.array([10,50,100,500])         # for the all spectrum
#         dx=8/(10*np.arange(3,500))             # for the limit eigenvalues
#
#     # computing the spectrum and eigenvalue limits
#     eigen_full(dx=dx,equ=param[i,0],dim=int(param[i,1]),delta=1.5,beta0=30,ord=param[i,2],T=2,Ndt=1,example=param[i,3])
#     limit_eigen(dx=dx,equ=param[i,0],dim=int(param[i,1]),delta=1.5,beta0=30,ord=param[i,2],T=2,Ndt=1,example=param[i,3])
#
#     # plotting the spectrum
#     graph_eigen_full(dx=dx,equ=param[i,0],dim=int(param[i,1]),ord=param[i,2],example=param[i,3])

# code for using a bash script to compute the full spectrum and the eigenvalues limits
equ=sys.argv[1]
dim=int(sys.argv[2])
ord=sys.argv[3]
example=sys.argv[4]

if int(dim)==1:
    # dx=10.5/np.array([100,500,1000,5000])  # for the all spectrum
    dx=10.5/(10*np.arange(3,500))          # for the limit eigenvalues
else:
    # dx=8/np.array([10,50,100,500])         # for the all spectrum
    dx=8/(10*np.arange(3,500))             # for the limit eigenvalues

# computing the spectrum and the eigenvalues limits
# eigen_full(dx=dx,equ=equ,dim=int(dim),delta=1.5,beta0=30,ord=ord,T=2,Ndt=1,example=example)
limit_eigen(dx=dx,equ=equ,dim=int(dim),delta=1.5,beta0=30,ord=ord,T=2,Ndt=1,example=example)


# Figures for the eigenvalues limits, according to the paper experiments

# equ=np.array(['scalar','scalar_dx2','scalar','scalar_dx2'])
# ord=np.array(['4','4','8','8'])
# example=np.array(['1D_heterogeneous_2','1D_heterogeneous_2','1D_heterogeneous_2','1D_heterogeneous_2'])
# graph_limit_eigen(dx=10.5/(10*np.arange(3,500)),equ=equ,delta=1.5,beta0=30,ord=ord,example=example,ind='Test3')
#
# equ=np.array(['scalar','scalar_dx2','scalar','scalar_dx2'])
# ord=np.array(['4','4','8','8'])
# example=np.array(['2D_heterogeneous_3','2D_heterogeneous_3','2D_heterogeneous_3','2D_heterogeneous_3'])
# graph_limit_eigen(dx=8/(10*np.arange(3,500)),equ=equ,delta=1.5,beta0=30,ord=ord,example=example,ind='Test5')
#
# equ=np.array(['elastic','elastic'])
# ord=np.array(['4','8'])
# example=np.array(['2D_heterogeneous_3','2D_heterogeneous_3'])
# graph_limit_eigen(dx=8/(10*np.arange(3,500)),equ=equ,delta=1.5,beta0=30,ord=ord,example=example,ind='Test7')