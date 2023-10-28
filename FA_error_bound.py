import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
import auxiliary_functions as aux_fun
import auxiliar_functions as aux0_fun
from scipy.stats import ortho_group
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eig
from matrices import *
from operators import *
import Methods as meth
import pickle
from scipy.optimize import minimize
from sympy import *
from time import time
import math
import sys # to run with a Bash script
from wezlz import *

import graph_q_ellip as graph_ell


def do_matrix(lamb00,lamb01,lamb1): # constructing the matrix with real eiganvalues lamb0 and imaginary eigenvalues lamb1
    L=np.zeros((2*len(lamb00)+len(lamb1),2*len(lamb00)+len(lamb1)))
    for i in range(len(lamb00)):
        L[2*i,2*i]=lamb00[i]
        L[2*i,2*i+1]=np.sqrt(np.abs(lamb01[i]))
        L[2*i+1,2*i]=-np.sqrt(np.abs(lamb01[i]))
        L[2*i+1,2*i+1]=lamb00[i]
    for i in range(len(lamb1)):
        L[2*len(lamb00)+i,2*len(lamb00)+i]=lamb1[i]
    return L


def limit_m_f(m,gamma,c,d):
    if m<=2*gamma:
        aux=4*gamma/(4*gamma-m)
        return 8*gamma/m*mp.exp(gamma*(aux+d/gamma+c**2/(4*gamma**2)/aux)-m**2/(4*gamma))
    else:
        return 4*mp.exp(d+c**2/(4*m))*pow(mp.exp(1)*gamma/m,m)


def ellipse_properties(w,scale,type,line_type=[1,0]): # plotting the minimum ellipse and returning its parameters

    e=pow(10,-12)
    max_it=20

    Q=np.zeros((len(w),2))
    Q[:,0]=w.real
    Q[:,1]=w.imag

    R=np.zeros((0,2))

    if np.max(np.abs(Q[:,0]))<pow(10,-5):
        aux0=np.max(Q[:,1])
        aux1=np.min([10,aux0/10])
        a12=0
        a22=pow(aux0/aux1,2)
        b1=0
        b2=0
        c=-aux0**2
    else:
        R,a12,a22,b1,b2,c=wezlz(Q,R,e,max_it)

    if type=='circles1':
        gamma,c_a,d,a=graph_ell.graph_q_circles1(Q,a12,a22,b1,b2,c,300,line_type,scale)
    elif type=='ellip':
        gamma,c_a,d,a=graph_ell.graph_q_ellip(Q,a12,a22,b1,b2,c,300,line_type,scale)
    elif type=='ellip_0':
        gamma,c_a,d,a=graph_ell.graph_q_ellip(Q,a12,a22,b1,b2,c,300,line_type,scale,'r')
    elif type=='circles2':
        gamma,c_a,d,a=graph_ell.graph_q_circles2(Q,a12,a22,b1,b2,c,300,line_type,scale)

    print('gamma: ',gamma,', c: ',c_a,', d: ',d,', a: ',a)
    # plt.show()
    if isinstance(gamma,float):
        return gamma*scale,c_a*scale,d[0]*scale,a*scale
        # return gamma,c,d[0],a
    return gamma[0]*scale,c_a[0]*scale,d[0][0],a*scale
    # return gamma[0],c[0],d[0][0],a


def examples_error_normal_matrix(n):

    # for i in range(n):
    #     print(i)
    #     aux=0
    #     n_imag=int(np.random.rand(1)[0]*50)
    #     if n_imag==0:
    #         n_imag=1
    #     elif n_imag==50:
    #         n_imag=49
    #     n_real=50-n_imag
    #
    #     while aux!=2:
    #         aux=0
    #         lim_imag=np.random.rand(1)[0]*100
    #         lim_real=np.random.rand(1)[0]*100
    #         while aux==0:
    #             aux=error_normal_matrix(n_imag,n_real,lim_imag,lim_real,i)
    from matplotlib.colors import LogNorm
    result=np.zeros((n,1000))*np.nan
    error_bound_full=result+0
    error_full=result+0
    max_degree=1
    for i in range(n):
        error=np.load('Faber_error_bound/error_exact_ind_'+str(i)+'.npy')
        error_bound=np.load('Faber_error_bound/error_estimate_ind_'+str(i)+'.npy')
        max_degree=np.maximum(max_degree,len(error))
        error_full[i,:len(error)]=error
        error_bound_full[i,:len(error)]=error_bound
        result[i,:len(error)]=error_bound-error
    result=np.abs(result[:,:max_degree])
    error_full=np.abs(error_full[:,:max_degree])
    error_bound_full=np.abs(error_bound_full[:,:max_degree])
    plt.contourf(result,norm = LogNorm(),levels=1000,cmap='winter')
    plt.colorbar()
    plt.xlabel('Polynomial degree', fontsize=18)
    plt.ylabel('Matrix idicator', fontsize=18)
    # plt.title('Diference between error bound and real error',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    np.save('Faber_error_bound/error_bound_data_n_'+str(n),error_bound_full)
    np.save('Faber_error_bound/real_error_data_n_'+str(n),error_full)
    plt.savefig('Faber_error_bound/error_bound_n_'+str(n)+'.pdf')
    plt.show()

    # import matplotlib as mpl
    # from mpl_toolkits.mplot3d import Axes3D
    # import numpy as np
    # import matplotlib.ticker as mticker
    # mpl.rcParams['legend.fontsize'] = 10
    #
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # n=10
    # error=np.load('Faber_error_bound/error_exact_ind_'+str(0)+'.npy')
    # error_bound=np.load('Faber_error_bound/error_estimate_ind_'+str(0)+'.npy')
    # x=np.arange(len(error))
    # y=np.ones(len(error))
    # ax.plot(x,y,np.log10(error),color='r',label='real error')
    # ax.plot(x,y,np.log10(error_bound),color='b',label='error bound')
    # for i in range(1,n):
    #     error=np.load('Faber_error_bound/error_exact_ind_'+str(i)+'.npy')
    #     error_bound=np.load('Faber_error_bound/error_estimate_ind_'+str(i)+'.npy')
    #     x=np.arange(len(error))
    #     y=(i+1)*np.ones(len(error))
    #     ax.plot(x,y,np.log10(error),color='r')
    #     ax.plot(x,y,np.log10(error_bound),color='b')
    #
    # # ax.plot_wireframe(X, Y, np.log10(Z), rstride=10, cstride=10)
    # def log_tick_formatter(val, pos=None):
    #     return f"$10^{{{int(val)}}}$"
    # ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    # ax.zaxis.set_major_locator(mticker.MaxNLocator(integer=True))


    # plt.savefig('Faber_error_bound/error_bound_n_'+str(n)+'.pdf')
    # plt.show()


def error_normal_matrix(n_imag,n_real,lim_imag,lim_real,ind):
    # Approximation of some matrices exponential using matrix-matrix multiplication

    mp.mp.dps=30

    if ind==1: # negative eigenvalues
        lamb00=(np.random.rand(n_imag)-0.8)*lim_real
        lamb1=(np.random.rand(n_real)-0.8)*lim_real
    elif ind==0: # positive eigenvalues
        lamb00=(np.random.rand(n_imag)+0.3)*lim_real
        lamb1=(np.random.rand(n_real)+0.3)*lim_real
    lamb01=np.random.rand(n_imag)*lim_imag
    lamb00[:int(n_imag/4)]=0  # creating some pure imaginary eigenvalues

    Q=ortho_group.rvs(dim=2*len(lamb00)+len(lamb1))
    H=Q.dot(do_matrix(lamb00,lamb01,lamb1).dot(Q.T))
    T=1
    H=T*H
    w,v=np.linalg.eig(np.array(H.tolist(),dtype=complex))

    gamma,c,d,a=ellipse_properties(w=w,scale=1,type='ellip_0')
    plt.title(r'Ellipse containing $\sigma(H)$',fontsize=20)
    if ind==1:
        plt.xlim(([-21,6]))
    elif ind==0:
        plt.ylim([-5,5])
    plt.savefig('Faber_error_bound_images/ellipse_ind_'+str(ind)+'.pdf')
    plt.show()

    if np.isnan(gamma):
        return 0

    if limit_m_f(1,gamma,c,d)>1.e10:
        print('not this!')
        return 1

    eps=pow(10,-16)

    m_max=aux0_fun.limit_m(eps/2,gamma,c,d,100)
    print('m_max: ',m_max)
    m_max=100
    coefficients_faber=aux_fun.Faber_approx_coeff(m_max+1,gamma,c,d)

    error_bound=aux0_fun.err_estimate(m_max,gamma,c,d,a,coefficients_faber)+eps/2

    print('error_bound')

    error=aux0_fun.err_exact(lamb00,lamb01,lamb1,Q,H,m_max,gamma,c,d,coefficients_faber)

    np.save('Faber_error_bound/error_exact_ind_'+str(ind),error)
    np.save('Faber_error_bound/error_estimate_ind_'+str(ind),error_bound)

    error=np.load('Faber_error_bound/error_exact_ind_'+str(ind)+'.npy')
    error_bound=np.load('Faber_error_bound/error_estimate_ind_'+str(ind)+'.npy')

    plt.plot((error_bound),'b',label='Our estimation',linestyle='--',linewidth=2)
    plt.plot((error),'r',label='Real error',linewidth=2)
    plt.yscale('log')
    plt.legend(fontsize=18)
    plt.xlabel('Polynomial degree', fontsize=24)
    plt.xticks(fontsize=18)
    plt.ylabel('Error', fontsize=24)
    plt.yticks(fontsize=18)
    plt.title('Error estimation',fontsize=25)
    plt.gca().set_ylim(bottom=pow(10,-17))
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('Faber_error_bound_images/error_bound_ind_'+str(ind)+'.pdf')
    plt.show()


def five_ellipses(n_imag,n_real,lim_imag,lim_real):
    # Approximation of some matrices exponential using matrix-matrix multiplication

    mp.mp.dps=30

    lamb00=(np.random.rand(n_imag)-0.9)*lim_real
    lamb01=np.random.rand(n_imag)*lim_imag
    lamb00[:int(n_imag/4)]=0  # creating some pure imaginary eigenvalues
    lamb1=(np.random.rand(n_real)-0.9)*lim_real

    Q=ortho_group.rvs(dim=2*len(lamb00)+len(lamb1))
    H=Q.dot(do_matrix(lamb00,lamb01,lamb1).dot(Q.T))
    T=1
    H=T*H
    w,v=np.linalg.eig(np.array(H.tolist(),dtype=complex))

    R=np.zeros((len(w),2))
    R[:,0]=w.real
    R[:,1]=w.imag
    plt.plot(R[:,0],R[:,1],'ob',alpha=0.8)

    ellipse_percent=np.array([1.3,1.1,1,0.9,0.7])

    graph_dashes=np.array([[12,0],[6,1,6,6],[8,4,2,4],[3,2],[1,1,1,1,1,6]],dtype=np.ndarray)
    for i in range(5):
        gamma,c,d,a=ellipse_properties(w=w,scale=ellipse_percent[i],type='ellip',line_type=graph_dashes[i])
        if np.isnan(gamma):
            print('fuck this')
            return 0

        # if limit_m_f(1,gamma,c,d)>1.e12:
        #     print('not this!')
        #     return 1

        eps=pow(10,-16)
        m_max=aux0_fun.limit_m(eps/2,gamma,c,d)
        print('m_max: ',m_max)
        m_max=200
        coefficients_faber=aux0_fun.Faber_approx_coeff(m_max+1,gamma,c,d)
        error=aux0_fun.err_exact(lamb00,lamb01,lamb1,Q,H,m_max,gamma,c,d,coefficients_faber)
        np.save('Faber_error_bound/error_exact_five_ellipses_ind_'+str(i),error)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.savefig('Faber_error_bound_images/five_ellipses.pdf')
    plt.show()

    for i in range(5):
        error=np.load('Faber_error_bound/error_exact_five_ellipses_ind_'+str(i)+'.npy')
        plt.plot(error,label=str(int(ellipse_percent[i]*100))+'%',linewidth=2,dashes=graph_dashes[i])
    plt.yscale('log')
    plt.legend(fontsize=18)
    plt.xlabel('Polynomial degree', fontsize=24)
    plt.xticks(fontsize=18)
    plt.ylabel('Error', fontsize=24)
    plt.yticks(fontsize=18)
    # plt.title('Error estimation',fontsize=20)
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('Faber_error_bound_images/error_five_ellipses.pdf')
    plt.show()


def five_circles_1(n_imag,n_real,lim_imag,lim_real,test='b'):
    # Approximation of some matrices exponential using matrix-matrix multiplication

    mp.mp.dps=30

    lamb00=(np.random.rand(n_imag)-1)*lim_real
    lamb01=np.random.rand(n_imag)*lim_imag
    lamb00[:int(n_imag/4)]=0  # creating some pure imaginary eigenvalues
    lamb1=(np.random.rand(n_real)-1)*lim_real

    Q=ortho_group.rvs(dim=2*len(lamb00)+len(lamb1))
    H=Q.dot(do_matrix(lamb00,lamb01,lamb1).dot(Q.T))
    T=1
    H=T*H
    w,v=np.linalg.eig(np.array(H.tolist(),dtype=complex))

    if test=='c':
        ellipse_percent=np.array([1,0])
    else:
        ellipse_percent=np.array([1,0.75,0.5,0.25,0])

    cant_ell=len(ellipse_percent)
    graph_dashes=np.array([[12,0],[6,1,6,6],[8,4,2,4],[3,2],[1,1,1,1,1,6]],dtype=np.ndarray)

    for i in range(cant_ell):
        gamma,c,d,a=ellipse_properties(w=w,scale=ellipse_percent[i],type='circles1',line_type=graph_dashes[i])
        if np.isnan(gamma):
            print('not this')
            return 0

        if limit_m_f(1,gamma,c,d)>1.e12:
            print('neither this!')
            return 1

        # eps=pow(10,-16)
        # m_max=aux0_fun.limit_m(eps/2,gamma,c,d)
        # print('m_max: ',m_max)
        # m_max=100
        # coefficients_faber=aux0_fun.Faber_approx_coeff(m_max+1,gamma,c,d)
        # error=aux0_fun.err_exact(lamb00,lamb01,lamb1,Q,H,m_max,gamma,c,d,coefficients_faber)
        # np.save('Faber_error_bound/error_exact_five_circles_1'+test+'_ind_'+str(i),error)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.xticks(fontsize=18)
    plt.xticks([-60,-45,-30,-15,0,10])
    plt.yticks(fontsize=18)
    plt.savefig('Faber_error_bound_images/five_circles_1'+test+'.pdf')
    plt.show()

    label=np.array(['circle','ellipse'])
    for i in range(cant_ell):
        error=np.load('Faber_error_bound/error_exact_five_circles_1'+test+'_ind_'+str(i)+'.npy')
        if test=='c':
           plt.plot(error,label=label[i],linewidth=2,dashes=graph_dashes[i])
        else:
            plt.plot(error,label=str(int(ellipse_percent[i]*100))+'%',linewidth=2,dashes=graph_dashes[i])
    plt.yscale('log')
    plt.legend(fontsize=18)
    plt.xlabel('Polynomial degree', fontsize=24)
    plt.xticks(fontsize=18)
    plt.ylabel('Error', fontsize=24)
    plt.yticks(fontsize=18)
    # plt.title('Error estimation',fontsize=20)
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('Faber_error_bound_images/error_five_circles_1'+test+'.pdf')
    plt.show()


def five_circles_2(n_imag,n_real,lim_imag,lim_real):
    # Approximation of some matrices exponential using matrix-matrix multiplication

    mp.mp.dps=30

    lamb00=(np.random.rand(n_imag)-0.9)*lim_real
    lamb01=np.random.rand(n_imag)*lim_imag
    lamb00[:int(n_imag/4)]=0  # creating some pure imaginary eigenvalues
    lamb1=(np.random.rand(n_real)-0.9)*lim_real

    Q=ortho_group.rvs(dim=2*len(lamb00)+len(lamb1))
    H=Q.dot(do_matrix(lamb00,lamb01,lamb1).dot(Q.T))
    T=1
    H=T*H
    w,v=np.linalg.eig(np.array(H.tolist(),dtype=complex))

    R=np.zeros((len(w),2))
    R[:,0]=w.real
    R[:,1]=w.imag
    # plt.plot(R[:,0],R[:,1],'ob',alpha=0.8)

    ellipse_percent=np.array([5,2,1,0.5,0.1])

    # coeff_graph=np.zeros((101,5))
    # Faber_graph=np.zeros((5,100,2*len(lamb00)+len(lamb1),2*len(lamb00)+len(lamb1)))

    graph_dashes=np.array([[12,6],[6,1,6,6],[12,6,2,6],[3,2],[2,1,2,1,2,6]],dtype=np.ndarray)
    for i in range(5):
        gamma,c,d,a=ellipse_properties(w=w,scale=ellipse_percent[i],type='circles2',line_type=graph_dashes[i])
        # if np.isnan(gamma):
        #     print('Error with gamma')
        #     return 0
        #
        # if limit_m_f(1,gamma,c,d)>1.e12:
        #     print('Is not suitable for approximation! Too large error')
        #     return 1

        # eps=pow(10,-16)
        # m_max=limit_m(eps/2,gamma,c,d)
        # print('m_max: ',m_max)
        m_max=100
        # coefficients_faber=Faber_approx_coeff(m_max+1,gamma,c,d)
        # error=err_exact(lamb00,lamb01,lamb1,Q,H,m_max,gamma,c,d,coefficients_faber)
        # np.save('Faber_error_bound/error_exact_five_circles_2b_ind_'+str(i),error)

        # coeff_graph[:,i]=coefficients_faber
        # Faber_graph[i,:,:,:]=Faber_approx_seq(H,m_max,gamma,c,d,coefficients_faber)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.savefig('Faber_error_bound_images/five_circles_2b.pdf')
    plt.show()

    for i in range(5):
        error=np.load('Faber_error_bound/error_exact_five_circles_2b_ind_'+str(i)+'.npy')
        plt.plot(error,label=str(int(ellipse_percent[i]*100))+'%',linewidth=2,dashes=graph_dashes[i])
        # plt.plot(coeff_graph[:,i],label=str(int(ellipse_percent[i]*100))+'%',linewidth=2)

    # error_Faber_graph=np.zeros((4,100))
    # for i in range(4):
    #     for j in range(100):
    #         error_Faber_graph[i,j]=np.max(np.abs(Faber_graph[0,j,:,:]-Faber_graph[1+i,j,:,:]))
    # for i in range(4):
    #     plt.plot(error_Faber_graph[i,:],label=str(int(ellipse_percent[i+1]*100))+'%',linewidth=2)
    plt.yscale('log')
    plt.legend(fontsize=15)
    plt.xlabel('Polynomial degree', fontsize=18)
    plt.xticks(fontsize=12)
    plt.ylabel('Error', fontsize=18)
    plt.yticks(fontsize=12)
    # plt.title('Error estimation',fontsize=20)
    plt.ylim([-16, 2])
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('Faber_error_bound_images/error_five_circles_2b.pdf')
    plt.show()


def five_circles_3(n_imag,n_real,lim_imag,lim_real):
    # Approximation of some matrices exponential using matrix-matrix multiplication

    mp.mp.dps=30

    lamb00=(np.random.rand(n_imag)-0.9)*lim_real
    lamb01=np.random.rand(n_imag)*lim_imag
    lamb00[:int(n_imag/4)]=0  # creating some pure imaginary eigenvalues
    lamb1=(np.random.rand(n_real)-0.9)*lim_real

    Q=ortho_group.rvs(dim=2*len(lamb00)+len(lamb1))
    H=Q.dot(do_matrix(lamb00,lamb01,lamb1).dot(Q.T))
    T=1
    H=T*H
    w,v=np.linalg.eig(np.array(H.tolist(),dtype=complex))

    R=np.zeros((len(w),2))
    R[:,0]=w.real
    R[:,1]=w.imag
    # plt.plot(R[:,0],R[:,1],'ob',alpha=0.8)

    ellipse_percent=np.array([1,0.7,0.4,0.1,0.01])

    coeff_graph=np.zeros((101,5))
    Faber_graph=np.zeros((5,100,2*len(lamb00)+len(lamb1),2*len(lamb00)+len(lamb1)))

    graph_dashes=np.array([[12,6],[6,1,6,6],[12,6,2,6],[3,2],[2,1,2,1,2,6]],dtype=np.ndarray)
    for i in range(5):
        gamma,c,d,a=ellipse_properties(w=w,scale=ellipse_percent[i],type=graph_dashes[i])
        if np.isnan(gamma):
            print('fuck this')
            return 0

        if limit_m_f(1,gamma,c,d)>1.e12:
            print('not this!')
            return 1

        m_max=100
        coefficients_faber=aux0_fun.Faber_approx_coeff(m_max+1,gamma,c,d)
        coeff_graph[:,i]=coefficients_faber
        Faber_graph[i,:,:,:]=aux0_fun.Faber_approx_seq(H,m_max,gamma,c,d,coefficients_faber)

    np.save('Faber_error_bound/Faber_graph',Faber_graph)
    np.save('Faber_error_bound/coeff_graph',coeff_graph)
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.draw()
    plt.savefig('Faber_error_bound/five_circles_2b.pdf')
    plt.show()

    Faber_graph=np.load('Faber_error_bound/Faber_graph.npy')
    # coeff_graph=np.load('Faber_error_bound/coeff_graph.npy')
    # for i in range(5):
    #     plt.plot(coeff_graph[:,i],label=str(int(ellipse_percent[i]*100))+'%',linewidth=2)

    error_Faber_graph=np.zeros((4,100))
    for i in range(4):
        for j in range(100):
            error_Faber_graph[i,j]=np.max(np.abs(Faber_graph[0,j,:,:]-Faber_graph[1+i,j,:,:]))
    for i in range(4):
        plt.plot(error_Faber_graph[i,:],label=str(int(ellipse_percent[i+1]*100))+'%',linewidth=2,dashes=graph_dashes[i])
    plt.yscale('log')
    plt.legend(fontsize=15)
    plt.xlabel('Polynomial degree', fontsize=18)
    plt.xticks(fontsize=12)
    plt.ylabel('Error', fontsize=18)
    plt.yticks(fontsize=12)
    # plt.title('Error estimation',fontsize=20)
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('Faber_error_bound/error_five_circles_2b.pdf')
    plt.show()


def sol_full_RK_7(var0,NDt,Dt,equ,dim,abc,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,example):

    RK_ref=np.zeros((len(var0),1))
    for i in range(len(NDt)):
        print('i',i)
        start=time()
        var=var0
        for j in range(NDt[i]):
            # var=RK_7_source(var=var,dt=Dt[0],equ=equ,dim=dim,abc=abc,delta=delta,beta0=beta0,ord=ord,dx=dx,c2=c2,nx=nx+1,ny=0,f=f,t0=t0+np.floor(Dt[0]*j/T0)*T0,f0=f0,i=j,source_type=source_type,f1=f1)
            var=RK_7_source(var=var,dt=Dt[i],equ=equ,dim=dim,abc=abc,delta=delta,beta0=beta0,ord=ord,dx=dx,param=param,nx=nx+1,ny=ny+1,f=f,param_ricker=param_ricker,i=j,source_type=source_type)
        print('time ',time()-start)
        RK_ref[:,i]=var[:,0]
    np.save(str(example)+'/RK_ref_full_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx),RK_ref)

    return RK_ref


def sol_full_faber(var0,NDt,Dt,equ,dim,abc,delta,beta0,ord,dx,param,nx,ny,f,source_type,example,degree,ind_source):

    vals=spectral_dist(var0,equ,dim,abc,delta,beta0,ord,dx,param,nx,ny,f,3,source_type,Dt[0],f)
    gamma,c,d,a_e=ellipse_properties(vals,1)

    sol_faber=np.zeros((len(degree),len(var0)))
    for i in range(len(NDt)):
        print(i)
        coefficients_faber=np.array(Faber_approx_coeff(degree[-1]+1,gamma*Dt[i],c*Dt[i],d*Dt[i]).tolist(),dtype=np.float_)
        if coefficients_faber[-1]==0:  # if the coeffcients are too large then there is no point to use larger time-steps
            print('break')
            break
        for j in range(len(degree)):
            u_k=0
            var=var0*1
            for l in range(NDt[i]):
                var=Faber_approx(var,degree[j]+1,gamma,c,d,equ,dim,abc,delta,beta0,ord,dx,param,nx+1,ny+1,coefficients_faber,ind_source,u_k)
            sol_faber[j,:]=var[:,0]
        np.save(str(example)+'/sol_faber_full_equ_'+str(equ)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(i)+'_dx_'+str(dx),sol_faber)
    return sol_faber,coefficients_faber,gamma,c,d,a_e


def acoustic_error_bound(dx,equ,dim,abc,delta,beta0,ord,T,Ndt,degree,example,ind_source):
    # solution of 1d acoustic equation with font term and constructed solutions with PML to validate accuracy
    # of the methods: RK7, Faber, RK-High order, RK2, Devito

    const=np.zeros(len(dx))
    for i in range(len(dx)):
        print('i:                ',i)
        # Model parameters
        a,b,nx,ny,X,param,x0,f,param_ricker,Dt,NDt,points,source_type,var0=domain_source(dx[i],T,Ndt,dim,equ,example,abc,ord)
        var0=var0*0+1

        # 7th order Runge-Kutta
        RK7=sol_full_RK_7(var0,np.array([int(T/np.minimum(Dt[0],0.005))]),np.array([np.minimum(Dt[0],0.005)]),equ,dim,abc,delta,beta0,ord,dx[i],param,nx,ny,f,param_ricker,source_type,example)

        # Faber polynomial approximation scalar
        FA,coefficients_faber,gamma,c,d,a_e=sol_full_faber(var0,np.array([1]),np.array([T]),equ,dim,abc,delta,beta0,ord,dx[i],param,nx,ny,f,source_type,example,degree,ind_source)

        error_bound=(err_estimate(degree[-1],gamma,c,d,a_e,coefficients_faber)+1.e-16/2)*np.linalg.norm(var0,2)
        error_real=np.zeros(degree[-1])
        for j in range(degree[-1]):
            error_real[j]=np.linalg.norm(RK7[:,0]-FA[j,:],2)

        np.save('Faber_error_bound/error_bound_equ_'+equ+'_ord_'+ord+'_example_'+str(example)+'_dx_'+str(dx[i]),error_bound)
        np.save('Faber_error_bound/error_real_equ_'+equ+'_ord_'+ord+'_example_'+str(example)+'_dx_'+str(dx[i]),error_real)
        error_bound=np.load('Faber_error_bound/error_bound_equ_'+equ+'_ord_'+ord+'_example_'+str(example)+'_dx_'+str(dx[i])+'.npy')
        error_real=np.load('Faber_error_bound/error_real_equ_'+equ+'_ord_'+ord+'_example_'+str(example)+'_dx_'+str(dx[i])+'.npy')

        # plt.plot(np.abs(error_real[1:]-error_real[:-1])>1.e-16)
        # plt.show()
        lim_compare=np.max(np.arange(len(error_real)-1)[np.abs(error_real[1:]-error_real[:-1])>np.min(error_real)])
        # plt.plot(aux)
        # plt.show()
        # asdf
        # lim_compare=10
        # if i>15:
        #     lim_compare=20
        # if i>40:
        #     lim_compare=30
        # if i>50:
        #     lim_compare=44
        # if i>60:
        #     lim_compare=50
        # if i>100:
        #     lim_compare=50
        const[i]=np.max(error_real[:lim_compare]/(error_bound[:lim_compare]))

        if (i+2)%10==0 or i==0 or i==len(dx)-1:
            plt.plot(error_real)
            plt.plot(error_bound*const[i])
            plt.yscale('log')
            plt.savefig('Faber_error_bound/error_bound_equ_'+equ+'_ord_'+ord+'_example_'+example+'_i_'+str(i+2)+'.pdf')
            plt.close()

    plt.plot(1/dx,const,'+-')
    plt.yscale('log')
    plt.savefig('Faber_error_bound/cond_2P_equ_'+equ+'_ord_'+ord+'_example_'+example+'.pdf')
    plt.close()




# five_circles_1(25,15,500,20)

# five_circles_2(25,15,0.5,1)
# five_circles_3(25,15,50,70)
# mp.mp.dps=50

# acoustic_error_bound(dx=10.5/(100*np.arange(2,151)),equ='scalar',dim=1,abc=1,delta=1.5,beta0=30,ord='4',T=0.005,Ndt=1,degree=np.arange(1,100),example='1D_homogeneous_0',ind_source='zero')
# acoustic_error_bound(dx=10.5/(100*np.arange(2,151)),equ='scalar',dim=1,abc=1,delta=1.5,beta0=30,ord='8',T=0.005,Ndt=1,degree=np.arange(1,100),example='1D_homogeneous_0',ind_source='zero')
# acoustic_error_bound(dx=10.5/(100*np.arange(2,80)),equ='scalar',dim=1,abc=1,delta=1.5,beta0=30,ord='4',T=0.005,Ndt=1,degree=np.arange(1,100),example='1D_heterogeneous_1a',ind_source='zero')

# acoustic_error_bound(dx=10.5/(100*np.arange(2,151)),equ='scalar_dx2',dim=1,abc=1,delta=1.5,beta0=30,ord='4',T=0.005,Ndt=1,degree=np.arange(1,100),example='1D_homogeneous_0',ind_source='zero')
# acoustic_error_bound(dx=10.5/(100*np.arange(2,151)),equ='scalar_dx2',dim=1,abc=1,delta=1.5,beta0=30,ord='8',T=0.005,Ndt=1,degree=np.arange(1,100),example='1D_homogeneous_0',ind_source='zero')
# acoustic_error_bound(dx=10.5/(100*np.arange(2,80)),equ='scalar_dx2',dim=1,abc=1,delta=1.5,beta0=30,ord='4',T=0.005,Ndt=1,degree=np.arange(1,100),example='1D_heterogeneous_1a',ind_source='zero')
