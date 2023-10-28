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

    start_i=0
    if os.path.isfile('eigenvalues/eigen_sr_equ_'+equ+'_ord_'+ord+'_example_'+example+'.npy'):
        eigen_sr=np.load('eigenvalues/eigen_sr_equ_'+equ+'_ord_'+ord+'_example_'+example+'.npy')
        eigen_lr=np.load('eigenvalues/eigen_lr_equ_'+equ+'_ord_'+ord+'_example_'+example+'.npy')
        eigen_li=np.load('eigenvalues/eigen_li_equ_'+equ+'_ord_'+ord+'_example_'+example+'.npy')
        eigen_lm=np.load('eigenvalues/eigen_lm_equ_'+equ+'_ord_'+ord+'_example_'+example+'.npy')
        start_i=np.max(np.arange(len(eigen_sr))[np.abs(eigen_sr)>pow(10,-16)])
    else:
        eigen_sr=np.zeros(len(dx))
        eigen_lr=np.zeros(len(dx))
        eigen_li=np.zeros(len(dx))
        eigen_lm=np.zeros(len(dx),dtype=np.complex_)

    # cycle to compute the eigenvalues for each dx
    for i in range(start_i,len(dx)):
        print('example: ',example,' equ: ',equ,' ord: ',ord,' i: ',i,)  # to know the iteration

        # Model parameters
        nx,ny,X,Y,param,f,param_ricker,Dt,NDt,points,source_type,var0=meth.domain_source(dx[i],T,Ndt,dim,equ,example,ord,delta)

        # constructing the linear operator
        def lin_op_H(var):
            var=np.expand_dims(var,axis=1)
            return op.op_H(var,equ,dim,delta,beta0,ord,dx[i],param,nx+1,ny+1)
        H=LinearOperator(shape=(len(var0),len(var0)),matvec=lin_op_H)

        start=time() # start counting time of computing the eigenvalues
        try:
            # ncv=np.min(np.array([np.max(np.array([1000,int((len(var0)-1))])),len(var0)-1]))
            ncv=len(var0)-1
            vals_sr,vecs_sr=eigs(H,1,ncv=ncv,which='SR')
            vals_lr,vecs_lr=eigs(H,1,ncv=ncv,which='LR')
            vals_li,vecs_li=eigs(H,1,ncv=ncv,which='LI')
            # vals_lm,vecs_lm=eigs(H,1,ncv=ncv,which='LM')
        except:
            print('I knew it was not going to work!!!')
            break
        end=time() # end counting time of computing the eigenvalues
        print('time: ',end-start)

        eigen_sr[i]=vals_sr.real
        eigen_lr[i]=vals_lr.real
        eigen_li[i]=vals_li.imag
        # eigen_lm[i]=vals_lm[0]

        # if(end-start>100000):    # condition to stop if the time to compute the eigenvalues is too big
        #     break
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
            label_str=label_str+'TC#1_'
        elif example=='1D_heterogeneous_1a':
            label_str=label_str+'TC#2_'
        elif example=='1D_heterogeneous_2':
            label_str=label_str+'TC#3_'
        elif example=='2D_homogeneous_0a':
            label_str=label_str+'TC#4_'
        elif example=='2D_heterogeneous_3a':
            label_str=label_str+'TC#5_'
        elif example=='2D_heterogeneous_2':
            label_str=label_str+'TC#6_'
        elif example=='2D_heterogeneous_3':
            label_str=label_str+'TC#7_'
        label_str=label_str+'ord'+ord
        if equ=='scalar':
            label_str=label_str+'_1SD'
        elif equ=='scalar_dx2':
            label_str=label_str+'_2SD'

        return label_str

    # repetitive line codes for graphic parameters
    def graph_block():
        plt.legend(fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('$1/\Delta x$',fontsize=24)
        plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.9)

    # lines types
    graph_dashes=np.array([[12,6],[6,1,6,6],[12,6,2,6],[3,2],[2,1,2,1,2,6]],dtype=np.ndarray)

    # cycle for the eigenvalue with the smallest real part -----------------------------------------------------------
    for i in range(len(example)):
        eigen_sr=np.load('eigenvalues/eigen_sr_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy')
        index_sr=np.max(np.arange(len(eigen_sr))[np.abs(eigen_sr)>pow(10,-16)])

        plt.plot((1/dx)[:index_sr],eigen_sr[:index_sr],linewidth=3,label=label_fun(example[i],ord[i],equ[i]),dashes=graph_dashes[i],alpha=0.8)

    plt.plot((1/dx)[:index_sr],-beta0*pow((delta-dx[:index_sr])/delta,2),color='k',linestyle='--',linewidth=3,label='left_bound')
    plt.ylabel('Real limit',fontsize=24)
    graph_block()
    plt.savefig('eigenvalues_images/eigen_experiments_sr_'+ind+'.pdf')
    plt.show()

    # cycle for the eigenvalue with the largest real part ------------------------------------------------------------
    for i in range(len(example)):
        eigen_lr=np.load('eigenvalues/eigen_lr_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy')
        index_lr=np.max(np.arange(len(eigen_lr))[np.abs(eigen_lr)>pow(10,-16)])

        plt.plot((1/dx)[:index_lr],eigen_lr[:index_lr],linewidth=3,label=label_fun(example[i],ord[i],equ[i]),dashes=graph_dashes[i],alpha=0.8)

    plt.ylabel('Real superior limit',fontsize=24)
    graph_block()
    plt.savefig('eigenvalues_images/eigen_experiments_lr_'+ind+'.pdf')
    plt.show()

    # cycle for the eigenvalue with the largest imaginary part -------------------------------------------------------
    for i in range(len(example)):
        eigen_li=np.load('eigenvalues/eigen_li_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy')
        index_li=np.max(np.arange(len(eigen_li))[eigen_li!=0])

        plt.plot((1/dx)[:index_li],-eigen_li[:index_li],linewidth=3,label=label_fun(example[i],ord[i],equ[i]),dashes=graph_dashes[i],alpha=0.8)

    plt.ylabel('Imaginary limit',fontsize=24)
    graph_block()
    plt.savefig('eigenvalues_images/eigen_experiments_li_'+ind+'.pdf')
    plt.show()

    # cycle for the eigenvalue with the slope of the largest imaginary part ------------------------------------------
    for i in range(len(example)):
        eigen_li=np.load('eigenvalues/eigen_li_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy')
        index_li=np.max(np.arange(len(eigen_li))[eigen_li!=0])

        plt.plot((1/dx)[:index_li-1],-(eigen_li[1:index_li]-eigen_li[:index_li-1])/((1/dx)[1:index_li]-(1/dx)[:index_li-1]),linewidth=3,label=label_fun(example[i],ord[i],equ[i]),dashes=graph_dashes[i],alpha=0.8)

    plt.ylabel('Slope',fontsize=24)
    graph_block()
    plt.savefig('eigenvalues_images/eigen_experiments_li_slope_'+ind+'.pdf')
    plt.show()

    # # cycle for the real part of the eigenvalue with the largest magnitude -------------------------------------------
    # for i in range(len(example)):
    #     eigen_lm=np.load('eigenvalues/eigen_lm_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy').real
    #     index_lm=np.max(np.arange(len(eigen_lm))[np.abs(eigen_lm)>pow(10,-16)])
    #
    #     plt.plot((1/dx)[:index_lm],eigen_lm[:index_lm],linewidth=3,label=label_fun(example[i],ord[i],equ[i]),dashes=graph_dashes[i],alpha=0.8)
    #
    # plt.ylabel('Real part',fontsize=24)
    # graph_block()
    # plt.savefig('eigenvalues/eigen_experiments_lm_real_'+ind+'.pdf')
    # plt.show()
    #
    # # cycle for the imaginary part of the eigenvalue with the largest magnitude --------------------------------------
    # for i in range(len(example)):
    #     eigen_lm=np.load('eigenvalues/eigen_lm_equ_'+equ[i]+'_ord_'+str(ord[i])+'_example_'+example[i]+'.npy').imag
    #     index_lm=np.max(np.arange(len(eigen_lm))[np.abs(eigen_lm)>pow(10,-16)])
    #
    #     plt.plot((1/dx)[:index_lm],eigen_lm[:index_lm],linewidth=3,label=label_fun(example[i],ord[i],equ[i]),dashes=graph_dashes[i],alpha=0.8)
    #
    # plt.ylabel('Imaginary part',fontsize=24)
    # graph_block()
    # plt.savefig('eigenvalues/eigen_experiments_lm_imag_'+ind+'.pdf')
    # plt.show()


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

        vals,vecs=eigs(A=H,k=len(var0)-2)

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
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
        plt.savefig('eigenvalues_images/eigenval_equ_'+equ+'_dim_'+str(dim)+'_ord_'+ord+'_i_'+str(i)+'.pdf')
        plt.show()


# cheking if there exist the paste to save the results, and creating one if there is not
if not os.path.isdir('eigenvalues/'):
    os.mkdir('eigenvalues')


# code for using a bash script to compute the full spectrum and the eigenvalues limits ------------------------------
# equ=sys.argv[1]
# dim=int(sys.argv[2])
# ord=sys.argv[3]
# example=sys.argv[4]
#
# if int(dim)==1:
#     # dx=10.5/np.array([100,500,1000,5000])  # for the all spectrum
#     dx=10.5/(10*np.arange(3,135))          # for the limit eigenvalues
# else:
#     # dx=8/np.array([10,50,100,500])         # for the all spectrum
#     dx=8/(5*np.arange(3,80))             # for the limit eigenvalues
#
# # computing the spectrum and the eigenvalues limits
# # eigen_full(dx=dx,equ=equ,dim=int(dim),delta=1.5,beta0=30,ord=ord,T=2,Ndt=1,example=example)
# limit_eigen(dx=dx,equ=equ,dim=int(dim),delta=1.5,beta0=30,ord=ord,T=2,Ndt=1,example=example)



#
#
# code for eigenvalues limits with variations of the PML conditions --------------------------------------------------

def graph_limit_eigen_PML(dx,equ,ord):
    #
    # delta=float(sys.argv[1])
    # beta0=float(sys.argv[2])
    # example=sys.argv[3]
    #
    # limit_eigen(dx=dx,equ=equ,dim=dim,delta=delta,beta0=beta0,ord=ord,T=2,Ndt=1,example=example)

    # limit_eigen(dx=dx,equ=equ,dim=dim,delta=1.5,beta0=30,ord=ord,T=2,Ndt=1,example='1D_heterogeneous_1a')
    # limit_eigen(dx=dx,equ=equ,dim=dim,delta=0.1,beta0=0.1,ord=ord,T=2,Ndt=1,example='1D_heterogeneous_1b')
    # limit_eigen(dx=dx,equ=equ,dim=dim,delta=3.2,beta0=200,ord=ord,T=2,Ndt=1,example='1D_heterogeneous_1c')


    def graph_block():
        plt.legend(fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('$1/\Delta x$',fontsize=26)
        plt.subplots_adjust(left=0.22, bottom=0.15, right=0.9, top=0.9)

    eigen_lia=np.load('eigenvalues/eigen_li_equ_'+equ+'_ord_'+str(ord)+'_example_1D_heterogeneous_1a.npy')
    eigen_lib=np.load('eigenvalues/eigen_li_equ_'+equ+'_ord_'+str(ord)+'_example_1D_heterogeneous_1b.npy')
    eigen_lic=np.load('eigenvalues/eigen_li_equ_'+equ+'_ord_'+str(ord)+'_example_1D_heterogeneous_1c.npy')

    index_lia=np.max(np.arange(len(eigen_lia))[eigen_lia!=0])
    index_lib=np.max(np.arange(len(eigen_lib))[eigen_lib!=0])
    index_lic=np.max(np.arange(len(eigen_lic))[eigen_lic!=0])

    graph_dashes=np.array([[12,6],[6,1,6,6],[12,6,2,6],[3,2],[2,1,2,1,2,6]],dtype=np.ndarray)
    # maximum imaginary part
    plt.plot((1/dx)[:index_lia],-eigen_lia[:index_lia],linewidth=3.2,label='$\delta=1.5$, '+r'$\beta_0=30$',alpha=0.8,dashes=graph_dashes[0])
    plt.plot((1/dx)[:index_lib],-eigen_lib[:index_lib],linewidth=2.9,label='$\delta=0.1$, '+r'$\beta_0=0.1$',alpha=0.8,dashes=graph_dashes[1])
    plt.plot((1/dx)[:index_lic],-eigen_lic[:index_lic],linewidth=3,label='$\delta=3.2$, '+r'$\beta_0=200$',alpha=0.8,dashes=graph_dashes[3])

    plt.ylabel('Imaginary limit',fontsize=26)
    graph_block()
    plt.savefig('eigenvalues/eigen_experiments_li_PML.pdf')
    plt.show()

    #  slope of the maximum imaginary part
    plt.plot((1/dx)[:index_lia-1],-(eigen_lia[1:index_lia]-eigen_lia[:index_lia-1])/((1/dx)[1:index_lia]-(1/dx)[:index_lia-1]),linewidth=3,label='$\delta=1.5$, '+r'$\beta_0=30$',marker='D',markersize=3)
    plt.plot((1/dx)[:index_lib-1],-(eigen_lib[1:index_lib]-eigen_lib[:index_lib-1])/((1/dx)[1:index_lib]-(1/dx)[:index_lib-1]),linewidth=3,label='$\delta=0.1$, '+r'$\beta_0=0.1$',marker='o',markersize=3)
    plt.plot((1/dx)[:index_lic-1],-(eigen_lic[1:index_lic]-eigen_lic[:index_lic-1])/((1/dx)[1:index_lic]-(1/dx)[:index_lic-1]),linewidth=3,label='$\delta=3.2$, '+r'$\beta_0=200$',marker='v',markersize=3)

    plt.ylabel('Slope',fontsize=26)
    graph_block()
    plt.savefig('eigenvalues_images/eigen_experiments_li_slope_PML.pdf')
    plt.show()
