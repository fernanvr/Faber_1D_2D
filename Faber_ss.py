import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from auxiliar_functions import *
from scipy.stats import ortho_group
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eig
from matrices import *
from operators import *
import pickle
from scipy.optimize import minimize
from sympy import *
from time import time
import math


def example_1():
    # Approximation of some matrices exponential using matrix-matrix multiplication

    # lamb0=np.array([20,9])
    # lamb1=np.array([-50,-5,1])
    # lamb0=np.random.rand(20)*50
    # lamb1=(np.random.rand(30)-0.9)*100
    lamb0=np.random.rand(10)
    lamb1=(np.random.rand(5)-0.8)*5

    Q=ortho_group.rvs(dim=2*len(lamb0)+len(lamb1))
    H=Q.dot(do_matrix(lamb0,lamb0,lamb1).dot(Q.T))
    T=1
    H=T*H
    w,v=np.linalg.eig(np.array(H.tolist(),dtype=complex))

    gamma,c,d,a=ellipse_properties(w=w,scale=1)

    eps=pow(10,-16)

    m_max=limit_m(eps/2,gamma,c,d)
    print('m_max: ',m_max)
    coefficients_faber=Faber_approx_coeff(m_max+1,gamma,c,d)

    error_bound=err_estimate(m_max,gamma,c,d,a,coefficients_faber)+eps/2
    plt.plot(np.log10(error_bound),'g',label='Our estimation')

    error=err_exact(lamb0,lamb0,lamb1,Q,H,m_max,gamma,c,d,coefficients_faber)
    plt.plot(np.log10(error),'r',label='Real error')
    plt.legend(loc='upper right')
    plt.xlabel('Polynomial degree', fontsize=18)
    plt.ylabel('Error ($log_{10}$)', fontsize=18)
    plt.title('Error estimation',fontsize=20)
    plt.savefig('example_1/experiment_2.pdf')
    plt.show()


def example_2(a,b,dx,equ,dim,abc,delta,beta0,ord):
    # Eigenvalue distribution of the discretized operator (using a matrix) of the wave equation with PML

    nx=round(a/dx)
    ny=round(b/dx)
    print(nx)
    print(ny)

    # c=np.zeros((ny,nx))+3
    # c[:int(ny/2),:]=10

    # c=np.zeros((ny,nx))+15
    # c[:int(ny/2),:]=6
    # c[:int(ny/3),:]=0.2

    c=np.zeros((ny,nx))+0.04
    c[:int(ny/2),:]=1
    c[:int(ny/3),:]=12
    c[:,int(nx/2):]=20
    c=np.matrix.flatten(c,'F')

    H1=matrix_H(equ,dim,abc,delta,beta0,ord,dx,pow(c,2),nx+1,ny+1)

    w,vr=eig(H1)
    # print(np.max(np.abs(w.real)))
    # print(np.max(np.max(w.imag)))
    plt.plot(w.real,w.imag,'bo')
    print(np.max(np.abs(w.real)))
    print(np.max(np.abs(w.imag)))
    plt.axhline(0,color='red')
    plt.axvline(0,color='red')
    plt.savefig('example_2/eigenvalues_'+str(equ)+'_'+str(dim)+'_'+str(abc)+'_'+str(ord)+'_'+str(dx)+'.pdf')
    plt.show()


def example_3(a,b,dx,n_dx,equ,dim,abc,delta,beta0,ord):
    # Calculating the inferior limit of cond_2(P) in the discretized operator of the wave equation

    cond_number=np.zeros(n_dx)

    for i in range(n_dx):
        print('Estamos por el paso i=',i)
        nx=round(a/dx)
        ny=round(b/dx)
        print(nx)
        print(ny)

        c2=np.zeros((ny,nx))+9
        c2[:int(ny/2),:]=100

        # c2=np.zeros((ny,nx))+225
        # c2[:int(ny/2),:]=36
        # c2[:int(ny/3),:]=0.04

        # c2=np.zeros((ny,nx))+121
        # c2[:int(ny/2),:]=36
        # c2[:int(ny/3),:]=0.04
        #
        # c2=np.zeros((ny,nx))+0.04
        # c2[:int(ny/2),:]=1
        # c2[:int(ny/3),:]=100
        # c2[:,int(nx/2):]=64

        c2=np.matrix.flatten(c2,'F')

        if abc==1:
            var=np.ones((nx*ny*5,1))/np.sqrt(nx*ny*5)
        else:
            var=np.ones((nx*ny*3,1))/np.sqrt(nx*ny*3)

        if len(var)>pow(10,4):
            def lin_op_H(var):
                var=np.expand_dims(var,axis=1)
                return op_H(var,equ,dim,abc,delta,beta0,ord,dx,np.expand_dims(c2,axis=1),nx+1,ny+1)

            H=LinearOperator(shape=(len(var),len(var)),matvec=lin_op_H)
            vals,vecs=eigs(H,100,which='LM')
            vals=np.concatenate((vals,np.conj(vals),np.array([-beta0*pow((delta-dx/2)/delta,2)]),vals-beta0*pow((delta-dx/2)/delta,2)/2,np.conjugate(vals)-beta0*pow((delta-dx/2)/delta,2)/2))
            np.save('example_6/vals_'+str(len(var)),vals)
            vals=np.load('example_6/vals_'+str(len(var))+'.npy')
        else:
            H=matrix_H(equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1)
            vals,vr=eig(H)
            
        gamma,c,d,a_e=ellipse_properties(w=vals,scale=1)


        m_max=np.max([100,limit_m(pow(10,-10)/2,gamma,c,d)])
        print('m_max: ',m_max)


        coefficients_faber=Faber_approx_coeff(m_max+1,gamma,c,d)

        error_bound=err_estimate(m_max,gamma,c,d,a_e,coefficients_faber)+limit_m_f(m_max,gamma,c,d)

        cfl=dx/np.max(c2)
        error=err_exact_general(var,m_max,gamma,c,d,cfl,equ,dim,abc,delta,beta0,ord,dx,np.expand_dims(c2,1),nx+1,ny+1,coefficients_faber)

        aux=error*np.reciprocal(error_bound)
        constant=pow(10,-5)
        if i==1:
            constant=pow(10,0)
        cond_number[i]=np.max(aux[error>=constant])
        print('condition number: ',cond_number[i])

        # plt.close()
        plt.plot(np.log10(np.array((cond_number[i]*error_bound).tolist(),dtype=np.float_)),'g',label='Our estimation')
        plt.plot(np.log10(error),'r',label='Real error')
        plt.legend(loc='upper right')
        plt.xlabel('Polynomial degree', fontsize=18)
        plt.ylabel('Error ($log_{10}$)', fontsize=18)
        plt.title('Error estimation',fontsize=20)
        plt.savefig('example_3/error_bound_'+str(equ)+'_'+str(dim)+'_'+str(abc)+'_'+str(ord)+'_dx_'+str(dx)+'_'+str(i)+'.pdf')
        plt.show()
        plt.close()
        np.savetxt('example_3/variables_'+str(equ)+'_'+str(dim)+'_'+str(abc)+'_'+str(ord)+'_dx_'+str(dx)+'.txt', np.expand_dims(cond_number[i],1))
        dx=dx/2

    print('cond_number_sharp',cond_number)
    plt.plot(cond_number,'b',label='Real error')
    plt.xlabel('$dx$ index', fontsize=18)
    plt.ylabel('Estimation of $cond_2(P)$', fontsize=18)
    plt.savefig('example_3/cond_2(P)_'+str(equ)+'_'+str(dim)+'_'+str(abc)+'_'+str(ord)+'.pdf')
    plt.show()
    np.savetxt('example_3/variables_'+str(equ)+'_'+str(dim)+'_'+str(abc)+'_'+str(ord)+'.txt', cond_number)


def example_4(a,b,dx,n_dx,equ,dim,abc,delta,beta0,ord,cond_P):
    # roundoff error of Faber polynomial evaluation

    for i in range(n_dx):
        print('Estamos por el paso i=',i)
        nx=round(a/dx)
        ny=round(b/dx)
        print(nx)
        print(ny)

        c2=np.zeros((ny,nx))+9
        c2[:int(ny/2),:]=100

        # c2=np.zeros((ny,nx))+225
        # c2[:int(ny/2),:]=36
        # c2[:int(ny/3),:]=0.04

        # c2=np.zeros((ny,nx))+121
        # c2[:int(ny/2),:]=36
        # c2[:int(ny/3),:]=0.04
        #
        # c2=np.zeros((ny,nx))+0.04
        # c2[:int(ny/2),:]=1
        # c2[:int(ny/3),:]=100
        # c2[:,int(nx/2):]=64

        c2=np.matrix.flatten(c2,'F')

        H=matrix_H(equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1)

        w,vr=eig(H)
        gamma,c,d,a_e=ellipse_properties(w,1)

        m_max=100

        coefficients_faber=Faber_approx_coeff(m_max,gamma,c,d)
        error_bound=cond_P[i]*err_estimate_poly(m_max,gamma,c,d,a_e,coefficients_faber)#/np.sqrt(nx*ny*5)

        if abc==1:
            var=np.ones((nx*ny*5,1))/np.sqrt(nx*ny*5)
        else:
            var=np.ones((nx*ny*3,1))/np.sqrt(nx*ny*3)
        error=err_exact_poly(var,m_max,gamma,c,d,equ,dim,abc,delta,beta0,ord,dx,np.expand_dims(c2,1),nx+1,ny+1,coefficients_faber)

        plt.plot(np.log10(np.array(error_bound[1:].tolist(),dtype=np.float_)),'g',label='Our estimation')
        plt.plot(np.log10(error[1:]),'r',label='Real error')
        plt.legend(loc='upper right')
        plt.xlabel('Polynomial degree', fontsize=18)
        plt.ylabel('Error ($log_{10}$)', fontsize=18)
        plt.title('Error estimation',fontsize=20)
        plt.savefig('example_4/error_bound_poly_'+str(equ)+'_'+str(dim)+'_'+str(abc)+'_'+str(ord)+'+_dx_'+str(dx)+'.pdf')
        plt.show()
        plt.close()
        dx=dx/2


def example_5():
    # Example of optimal choice of scaling parameter and polynomial degree with a matrix from first_example
    # and its comparation with high order RK methods
    lamb0=np.random.rand(20)*50
    lamb1=(np.random.rand(30)-0.9)*100

    Q=ortho_group.rvs(dim=2*len(lamb0)+len(lamb1))
    H=Q.dot(do_matrix(lamb0,lamb1).dot(Q.T))
    H=H
    w,v=np.linalg.eig(np.array(H.tolist(),dtype=complex))

    gamma,c,d,a=ellipse_properties(w,scale=1)

    tol=pow(10,-10)

    # m_max=limit_m(eps/2,gamma,c,d)
    # print(m_max)
    m_max=100

    v=np.ones((2*len(lamb0)+len(lamb1),1))/np.sqrt(2*len(lamb0)+len(lamb1))

    n_s=100
    s=np.linspace(1,n_s,n_s)
    sol_exact=exact_exp(lamb0,lamb1,Q).dot(v)


    # error=np.zeros((m_max,n_s))
    #
    # coefficients_faber=np.zeros((m_max+1,n_s))
    # for i in range(n_s):
    #     print('i ', i)
    #     coefficients_faber[:,i]=Faber_approx_coeff(m_max+1,gamma/s[i],c/s[i],d/s[i])
    #     sol_faber=Faber_approx_seq(H,m_max,gamma,c,d,coefficients_faber[:,i])
    #     sol_approx=sol_faber.dot(v)
    #     for l in range(int(s[i]-1)):
    #         for m in range(m_max):
    #             sol_approx[m,:]=sol_faber[m,:].dot(sol_approx[m,:])
    #
    #     for j in range(m_max):
    #         error[j,i]=np.sqrt(np.sum(pow(sol_exact-sol_approx[j,:],2)))
    #
    # np.savetxt('example_5/error_novo',error)

    error=np.loadtxt('example_5/error_novo')
    error1=np.log10(error)
    error1[error1>1]=1
    error1[error1<-16]=-16
    plt.contourf(error1,levels=100)
    plt.title('Exact error ($log_{10}$)',fontsize=20)
    plt.xlabel('Scaling parameter (s)', fontsize=18)
    plt.ylabel('Polynomial degree (m)', fontsize=18)
    plt.colorbar()
    # plt.savefig('example_5/error.pdf')
    plt.show()

    operations=np.expand_dims(np.linspace(0,m_max-1,m_max),1).dot(np.expand_dims(s,0))

    print('operations ',np.min(operations[error<tol]))
    aux=np.argmin(operations[error<tol])
    degree=np.expand_dims(np.linspace(0,m_max-1,m_max),1).dot(np.ones((1,n_s)))
    print('# steps',np.min(operations[error<tol])/degree[error<tol][aux])
    print('degree',degree[error<tol][aux])
    print(error[int(degree[error<tol][aux]),int(np.min(operations[error<tol])/degree[error<tol][aux]-1)])
    # sdfasfsdf
    # error_estimate=np.zeros((m_max,n_s))
    # for i in range(n_s):
    #     print('i ', i)
    #     error_estimate[:,i]=err_estimate(m_max,gamma,c,d,a,coefficients_faber[:,i])+limit_m_f(m_max,gamma/s[i],c/s[i],d/s[i])
    #
    # np.savetxt('example_5/error_estimate',error_estimate)

    # error_estimate=np.loadtxt('example_5/error_estimate')
    # print(error_estimate)
    #
    # error1=np.log10(error_estimate)
    # plt.contourf(error1,levels=100)
    # plt.title('Estimated error ($log_{10}$)', fontsize=20)
    # plt.xlabel('Scaling parameter (s)', fontsize=18)
    # plt.ylabel('Polynomial degree (m)', fontsize=18)
    # plt.colorbar()
    # plt.savefig('example_5/error_estimate.pdf')
    # plt.show()

    # aux=np.argmin(operations[error_estimate<tol])
    # print('# steps',np.min(operations[error_estimate<tol])/degree[error_estimate<tol][aux])
    # print('degree',degree[error_estimate<tol][aux])
    # print(error_estimate[int(degree[error_estimate<tol][aux]),int(np.min(operations[error_estimate<tol])/degree[error_estimate<tol][aux]-1)])

    # error_rk=np.zeros((m_max-2,n_s))

    # for i in range(n_s):
    #     print('i ', i)
    #     sol_rk=RK_matrix_seq(H,1,1/s[i],m_max)
    #     sol_approx=sol_rk.dot(v)
    #     for l in range(int(s[i]-1)):
    #         for m in range(m_max-2):
    #             sol_approx[m,:]=sol_rk[m,:].dot(sol_approx[m,:])
    #
    #     for j in range(m_max-2):
    #         error_rk[j,i]=np.sqrt(np.sum(pow(sol_exact-sol_approx[j,:],2)))

    # np.savetxt('example_5/error_rk_novo',error_rk)

    error_rk=np.loadtxt('example_5/error_rk')
    print(error_rk)

    error1=np.log10(error_rk)
    error1[error1>1]=1
    error1[error1<-16]=-16
    plt.contourf(error1,levels=100)
    plt.title('Exact error ($log_{10}$)', fontsize=20)
    plt.xlabel('Time step (s)', fontsize=18)
    plt.ylabel('Polynomial degree (m)', fontsize=18)
    plt.colorbar()
    plt.savefig('example_5/error_rk.pdf')
    plt.show()

    operations=np.expand_dims(np.linspace(2,99,98),1).dot(np.expand_dims(s,0))

    print('operations ',np.min(operations[error_rk<tol]))
    degree=np.expand_dims(np.linspace(2,99,98),1).dot(np.ones((1,n_s)))
    aux=np.argmin(operations[error_rk<tol])
    print('# steps',np.min(operations[error_rk<tol])/degree[error_rk<tol][aux])
    print('degree',degree[error_rk<tol][aux])
    print(error_rk[int(degree[error_rk<tol][aux]-2),int(np.min(operations[error_rk<tol])/degree[error_rk<tol][aux]-1)])


def example_6(a,b,dx,equ,dim,abc,delta,beta0,ord,T):
    # 2D acoustic wave equation in heterogeneous medium using Faber series, error estimate, and assuming RK7 as the real solution

    nx=round(a/dx)
    ny=round(b/dx)
    print(nx)
    print(ny)

    X,Y=np.meshgrid(np.linspace(dx,a,nx),np.linspace(dx,b,ny))
    np.save('sixth_example_X',X)
    np.save('sixth_example_Y',Y)

    c2=np.zeros((ny,nx))+100
    c2[Y>=b/2]=36
    c2[(Y>=3/4*b)*(X>=2/3*a)]=0.4
    c2=np.expand_dims(np.matrix.flatten(c2,'F'),axis=1)

    f=source_x(x0=4.96,y0=2,rad=dx/2,X=np.matrix.flatten(X,'F'),Y=np.matrix.flatten(Y,'F'),nx=nx,ny=ny,abc=abc)
    f0=24
    t0=1.2/f0
    T0=0.14

    if abc==1:
        var=np.zeros((nx*ny*5,1))
    else:
        var=np.zeros((nx*ny*3,1))

    dt=dx/np.max(np.sqrt(c2))/2
    Nt0=int(T0/dt)+1
    dt0=T0/Nt0
    print('Nt0 ',Nt0)
    for i in range(Nt0):
        var=RK_7_source(var,dt0,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1,f,t0,f0,i)
    var0=var


    print('Pronto 1')
    def lin_op_H(var):
        var=np.expand_dims(var,axis=1)
        return (T-T0)*op_H(var,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1)

    H=LinearOperator(shape=(len(var),len(var)),matvec=lin_op_H)


    # vals,vecs=eigs(H,100,which='LM')
    # vals=np.concatenate((vals,np.conj(vals),np.array([-beta0*pow((delta-dx/2)/delta,2)]),vals-beta0*pow((delta-dx/2)/delta,2)/3,np.conjugate(vals)-beta0*pow((delta-dx/2)/delta,2)/3))
    # np.save('example_6/vals',vals)
    vals=np.load('example_6/vals.npy')
    gamma,c,d,a_e=ellipse_properties(vals,1)

    dx0=0.4
    # cond_P=estimation_cond_P(a,b,dx0,equ,dim,abc,delta,beta0,ord,dt)
    # np.savetxt('example_6/cond_P',cond_P)
    cond_P=np.loadtxt('example_6/cond_P')
    print('example_6/cond_2(P)_'+str(equ)+'_'+str(dim)+'_'+str(abc)+'_'+str(ord)+'.pdf',cond_P)

    constant_err=cond_P*np.sqrt(sum(var0*var0))

    tol=pow(10,-8)
    m_max=100
    s_max=11
    estimate_error=np.zeros((m_max,s_max))
    real_error=np.zeros((m_max,s_max))
    coefficients_faber=np.zeros((m_max+1,s_max))*mp.exp(0)

    for i in range(s_max):
        i=10
        s0=(T-T0)/(i+1)
        coefficients_faber[:,i]=Faber_approx_coeff(m_max+1,gamma*s0,c*s0,d*s0)
        estimate_error[:,i]=(err_estimate(m_max,gamma,c,d,a_e,coefficients_faber[:,i])+limit_m_f(m_max,gamma*s0,c*s0,d*s0))*constant_err
        Nt2=int(s0/dt)+1
        print(Nt2)
        dt2=s0/Nt2
        print('dt2 ',dt2)
        var=var0
        for j in range(Nt2):
            var=RK_7(var,dt2,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1)

        sol_faber=Faber_approx_op_seq(var0,m_max,gamma,c,d,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1,coefficients_faber[:,i])
        for j in range(m_max):
            real_error[j,i]=np.sqrt(sum(pow(sol_faber[:,j]-var[:,0],2)))
        plt.plot(real_error[:,i])
        plt.plot(estimate_error[:, i])
        f_list=np.zeros(m_max)
        for i in range(m_max):
            f_list[i]=limit_m_f(i+1,gamma*s0,c*s0,d*s0)
        plt.plot(f_list*constant_err)
        plt.plot(f_list)
        print(constant_err)
        plt.show()
        print(real_error[:,i])
        dsfasdgaf


    np.save('example_6/estimate_error',estimate_error)
    np.save('example_6/real_error', real_error)
    np.save('example_6/coeff_faber', coeff_faber)
    estimate_error=np.load('example_6/estimate_error.npy')
    real_error=np.load('example_6/real_error.npy')
    coeff_faber=np.load('example_6/coeff_faber.npy')
    print(estimate_error)
    print(np.min(real_error))
    print(coeff_faber[:,99])
    print(real_error[:,99])
    hggfhjfhg
    plt.contourf(np.log10(estimate_error),levels=100)
    plt.xlabel('Scaling parameter (s)', fontsize=18)
    plt.ylabel('Polynomial degree (m)', fontsize=18)
    plt.title('Error estimate',fontsize=22)
    plt.colorbar()
    plt.savefig('example_6/fig_estimate_error')
    plt.show()

    plt.contourf(np.log10(real_error),levels=100)
    plt.xlabel('Scaling parameter (s)', fontsize=18)
    plt.ylabel('Polynomial degree (m)', fontsize=18)
    plt.title('Real error',fontsize=22)
    plt.colorbar()
    plt.savefig('example_6/fig_real_error')
    plt.show()

    plt.contourf(np.log10(coeff_faber),levels=100)
    plt.xlabel('Scaling parameter (s)', fontsize=18)
    plt.ylabel('Polynomial degree (m)', fontsize=18)
    plt.title('Faber coefficients',fontsize=22)
    plt.colorbar()
    plt.savefig('example_6/coeff_faber')
    plt.show()


def example_7(a,b,dx,equ,dim,abc,delta,beta0,ord,T):
    # 2D acoustic wave equation in heterogeneous medium using Faber series, assuming RK7 as the real solution, and
    # using Faber estimate to estimate the optimum parameters to calculate the solution
    nx=round(a/dx)
    ny=round(b/dx)
    print(nx)
    print(ny)

    X,Y=np.meshgrid(np.linspace(dx,a,nx),np.linspace(dx,b,ny))
    np.save('sixth_example_X',X)
    np.save('sixth_example_Y',Y)

    c2=np.zeros((ny,nx))+100
    c2[Y>=b/2]=36
    c2[(Y>=3/4*b)*(X>=2/3*a)]=0.4
    c2=np.expand_dims(np.matrix.flatten(c2,'F'),axis=1)

    f=source_x(x0=4.96,y0=2,rad=dx/2,X=np.matrix.flatten(X,'F'),Y=np.matrix.flatten(Y,'F'),nx=nx,ny=ny,abc=abc)
    f0=24
    t0=1.2/f0
    T0=0.14

    dt=dx/np.max(np.sqrt(c2))/2
    Nt0=int(T0/dt)+1
    dt0=T0/Nt0
    Nt1=int((T-T0)/dt)+1
    dt1=(T-T0)/Nt1

    print('Nt0 ',Nt0)
    print('Nt1 ',Nt1)

    jump=1

    RK_ref=np.zeros((int((Nt0+Nt1)/jump+1),ny,nx))
    if abc==1:
        var=np.zeros((nx*ny*5,1))
    else:
        var=np.zeros((nx*ny*3,1))

    for i in range(Nt0):
        var=RK_7_0(var,dt0,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1,f,t0,f0,i)
        if i%jump==0:
            RK_ref[int(i/jump),:]=var[:nx*ny].reshape((ny,nx),order='F')
    var0=var
    for i in range(Nt1):
        var=RK_7(var,dt1,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1)
        if (Nt0+i)%jump==0:
            RK_ref[int((Nt0+i)/jump),:]=var[:nx*ny].reshape((ny,nx),order='F')
    np.save('sixth_example',RK_ref)
    np.save('sixth_example_var0', var0)

    print('Pronto 1')
    def lin_op_H(var):
        var=np.expand_dims(var,axis=1)
        return (T-T0)*op_H(var,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1)

    if abc==1:
        H=LinearOperator(shape=(nx*ny*5,nx*ny*5),matvec=lin_op_H)
    else:
        H=LinearOperator(shape=(nx*ny*3,nx*ny*3),matvec=lin_op_H)

    # vals,vecs=eigs(H,100,which='LM')
    # vals=np.concatenate((vals,np.conj(vals),np.array([-beta0*pow((delta-dx/2)/delta,2)]),vals-beta0*pow((delta-dx/2)/delta,2)/2,np.conjugate(vals)-beta0*pow((delta-dx/2)/delta,2)/2))
    # vals=np.concatenate((vals,np.conj(vals),np.array([-beta0*pow((delta-dx/2)/delta,2)])))
    # np.save('sixth_example_vals',vals)
    vals=np.load('sixth_example_vals.npy')
    gamma,c,d,a_e=ellipse_properties(vals,1)

    mp.mp.dps=300
    dx0=0.4
    cond_P=estimation_cond_P(a,b,dx0,equ,dim,abc,delta,beta0,ord,dt)/(dx0/dx*4)
    np.savetxt('sixth_example_cond_P',cond_P)
    cond_P=np.loadtxt('sixth_example_cond_P')
    print('cond_P ',cond_P)

    constant_err=cond_P*np.sqrt(sum(var0*var0))

    tol=pow(10,-8)
    # def fun_error(s,m):
    #     aux=err_estimate(m,gamma*(T-T0)/s,c*(T-T0)/s,d*(T-T0)/s,a_e*(T-T0)/s)[-1]*constant_err
    #     print(aux)
    #     if aux>tol:
    #         return 10*(Nt0+Nt1)
    #     else:
    #         return s*m
    #
    # def opt_error(m_max):
    #     s0=1
    #     s=np.arange(1,int(2*gamma/m_max*(T-T0)))
    #     s1=s[np.argmin(8*gamma/(s/(T-T0)*m_max)*np.exp(4*gamma/np.reciprocal(4*gamma*s/(T-T0)-m_max*s*s/((T-T0)*(T-T0)))+d*np.reciprocal(s/(T-T0))-m_max**2*s/(T-T0)/(4*gamma)+c**2*(4*gamma*np.reciprocal(s/(T-T0))-m_max)/(16*gamma**2)))]
    #     aux=8*gamma/(s1/(T-T0)*m_max)*np.exp(4*gamma/(4*gamma*s1/(T-T0)-m_max*s1*s1/((T-T0)*(T-T0)))+d/(s1/(T-T0))-m_max**2*s1/(T-T0)/(4*gamma)+c**2*(4*gamma/(s1/(T-T0))-m_max)/(16*gamma**2))*constant_err
    #     if aux>tol:
    #         s1=int(np.max(np.array([np.exp(1)*gamma/(m_max*pow(tol/(4*constant_err*np.max([1,np.exp(d*m_max/(2*gamma)+c**2*m_max/(16*gamma**2))])),1/m_max)),2*gamma/m_max])*(T-T0)))+1
    #     s1=np.max([s1,int((T-T0)*(gamma+c*c/(4*gamma)+d)/np.log(pow(10,12)*tol))+1])
    #     print('s1 ',s1)
    #     print('fun_error',fun_error(s1, m_max))
    #     while s1>s0:
    #         aux=fun_error(int((s0+s1)/2),m_max)
    #         aux1=Faber_approx_value_coeff(2,gamma*(T-T0)/int((s0+s1)/2),c*(T-T0)/int((s0+s1)/2),d*(T-T0)/int((s0+s1)/2))[-1]
    #         if aux==10*(Nt0+Nt1) or aux1>pow(10,12)*tol:
    #             s0=int((s0+s1)/2)+1
    #             print('s0 ',s0)
    #         else:
    #             s1=int((s0+s1)/2)
    #             print('s1 ',s1)
    #     m0=1
    #     m1=m_max
    #     aux=(err_estimate(m_max,gamma*(T-T0)/s0,c*(T-T0)/s0,d*(T-T0)/s0,a_e*(T-T0)/s0)+limit_m_f(m_max,gamma*(T-T0)/s0,c*(T-T0)/s0,d*(T-T0)/s0))*constant_err
    #     while m1>m0:
    #         if aux[int((m0+m1)/2)]>tol:
    #             m0=int((m0+m1)/2)+1
    #             print('m0 ',m0)
    #         else:
    #             m1=int((m0+m1)/2)
    #             print('m1 ',m1)
    #     print('aux[m0] ',aux[m0-1])
    #     return s0,m0
    #
    # print(gamma,d,c)
    # # asdfasg
    # m_max=100
    # # s,m=opt_error(m_max)
    # s=5
    # m=96
    # print(s,m)

    Nt2=int((T-T0)/s/dt)+1
    print(Nt2)
    dt2=(T-T0)/s/Nt2
    print('dt2 ',dt2)
    var=var0
    for i in range(Nt2):
        var=RK_7(var,dt2,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1)

    sol_faber=var0
    # theta=np.linspace(0,1,1000)
    # plt.plot(theta,np.exp((gamma*(T-T0)/s+c*c/(4*gamma)*(T-T0)/s)*np.cos(2*np.pi*theta)+d+1j*(gamma*(T-T0)/s-c*c/(4*gamma)*(T-T0)/s)*np.sin(2*np.pi*theta)).real)
    # plt.show()
    coeff_faber=Faber_approx_value_coeff(m,gamma*(T-T0)/s,c*(T-T0)/s,d*(T-T0)/s)
    np.savetxt('sixth_example_coeff',coeff_faber)
    # plt.xlabel('index $j$', fontsize=18)
    # plt.ylabel('$log_{10}a_j$', fontsize=18)
    # plt.plot(coeff_faber)
    # plt.show()

    sol_faber=Faber_approx_value(sol_faber*mp.exp(0),m,gamma*(T-T0)/s,c*(T-T0)/s,d*(T-T0)/s,nx*ny*5,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1,(T-T0)/s,coeff_faber)
    error=(sol_faber-var)
    # print(error)
    # print(np.sqrt(sum(error*error)))
    dsfgsdg

    for i in range(s):
        start=time()
        sol_faber=Faber_approx_value(sol_faber,m,gamma*(T-T0)/s,c*(T-T0)/s,d*(T-T0)/s,nx*ny*5,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,ny+1,(T-T0)/s,coeff_faber)
        end=time()
        print('time for step ',i,': ',end-start)

    error=sol_faber-var
    print(error)
    print(np.sqrt(sum(error*error)))
    #'0.0057780740702915012196447592424807074137548317696679297'
    #'0.0035847196051179233345025259738386990364027454026674332'
    #'1.7445169843899117324314853619713167692604416369357622e+138' 50
    # 243300571202292449130007552.33750121145195313452005358


def example_8(a,dx,equ,dim,abc,delta,beta0,ord,T):
    # first order 1D homogeneous scalar wave equation without source term and PML

    nx=round(a/dx)
    print(nx)
    
    X=np.expand_dims(np.linspace(dx,a,nx),1)

    np.save('example_8/X',X)

    c2=np.zeros((nx,1))+50

    dt=dx/np.max(np.sqrt(c2))/2
    Nt=int(T/dt)+1
    dt=T/Nt
    print('dt ',dt)
    print('Nt ',Nt)

    if abc==1:
        var0=np.zeros((nx*3,1))
    else:
        var0=np.zeros((nx*2,1))
    var0[:nx]=example_8_ic(X,a)

    jump=1
    RK_ref=np.zeros((nx,int(Nt/jump+1)))
    var=var0
    for i in range(Nt):
        var=RK_7(var,dt,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0)
        if i%jump==0:
            RK_ref[:,int(i/jump)]=var[:nx,0]

    np.save('example_8/RK_ref',RK_ref)

    solution=np.zeros((nx,int(Nt/jump+1)))
    for i in range(Nt):
        solution[:,int(i/jump)]=example_8_solution(X,a,i*dt,np.sqrt(c2[0]))[:,0]
    np.save('example_8/solution',solution)

    def lin_op_H(var):
        var=np.expand_dims(var,axis=1)
        return op_H(var,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0)

    H=LinearOperator(shape=(len(var0),len(var0)),matvec=lin_op_H)

    # vals,vecs=eigs(H,100,which='LM')
    # vals=np.concatenate((vals,np.conj(vals),np.array([-beta0*pow((delta-dx/2)/delta,2)]),vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2,-vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2))
    # np.save('example_8/vals',vals)
    vals=np.load('example_8/vals.npy')
    gamma,c,d,a_e=ellipse_properties(vals,1)

    coefficients_faber=Faber_approx_coeff(3,gamma*dt,c*dt,d*dt)
    sol_faber=np.zeros((nx,int(Nt/jump+1)))
    var=var0
    for i in range(Nt):
        var=Faber_approx(var,3,gamma,c,d,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,coefficients_faber)
        if i%jump==0:
            sol_faber[:,int(i/jump)]=var[:nx,0]
    np.save('example_8/sol_faber',sol_faber)

    print('1')
    sol_rk=np.zeros((nx,int(Nt/jump+1)))
    var=var0
    for i in range(Nt):
        var=RK_op(var,1,dt,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,4)
        if i%jump==0:
            sol_rk[:,int(i/jump)]=var[:nx,0]
    np.save('example_8/sol_rk',sol_rk)

    print('2')
    sol_rk2=np.zeros((nx,int(Nt/jump+1)))
    var=var0
    for i in range(Nt):
        var=RK_2(var,dt,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0)
        if i%jump==0:
            sol_rk2[:,int(i/jump)]=var[:nx,0]
    np.save('example_8/sol_rk2',sol_rk2)

    print('3')
    sol_devito=np.zeros((nx,int(Nt/jump+1)))
    var0=var0[:nx]
    sol_devito[:,0]=var0[:,0]
    var=var0+dt**2*dp2x(c2*var0,dx)
    for i in range(1,Nt):
        var1=2*var-var0+dt**2*dp2x(c2*var,dx)
        var0=var
        var=var1
        if i%jump==0:
            sol_devito[:,int(i/jump)]=var[:,0]
    np.save('example_8/sol_devito',sol_devito)


def example_9(a,dx,equ,dim,abc,delta,beta0,ord,T,Ndt,m_max):
    # first order 1D homogeneous scalar wave equation without source term and PML

    nx=round(a/dx)
    print(nx)

    X=np.expand_dims(np.linspace(dx,a,nx),1)

    np.save('example_9/X',X)

    c2=np.zeros((nx,1))+15

    dt=dx/np.max(np.sqrt(c2))/10
    print(dt)
    Dt=np.linspace(dt,Ndt*dt,Ndt)
    NDt=np.ceil(T/Dt).astype(int)
    Dt=T/NDt

    source_type="8"
    var0=ini_var0(abc,nx,X,source_type,a)
    # plt.plot(X,var0[:nx])
    # plt.title('Initial condition',fontsize=20)
    # plt.xlabel('x',fontsize=18)
    # plt.ylabel('u',fontsize=18)
    # plt.show()
    # asdfsdaf

    # # solution=example_8_solution(X,a,T,np.sqrt(c2[0]))[:,0]
    # # np.save('example_9/solution',solution)
    # # solution=np.load('example_9/solution.npy')
    # #
    # # RK_ref=np.zeros((1,Ndt))
    # # for i in range(Ndt):
    # #     var=var0
    # #     for j in range(NDt[i]):
    # #         var=RK_7(var,Dt[i],equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0)
    # #     RK_ref[0,i]=np.log10(np.max(np.abs(var[:nx,0]-solution)))
    # # np.save('example_9/RK_ref',RK_ref)
    # RK_ref=np.load('example_9/RK_ref.npy')
    # RK_ref[RK_ref>1]=1
    # RK_ref[RK_ref!=RK_ref]=1
    # print('RK_ref',RK_ref)
    # # plt.plot(Dt,RK_ref[0,:],label='RK7')
    # # plt.xlabel('$\Delta$t',fontsize=18)
    # # plt.ylabel('$log_{10}$(error)',fontsize=18)
    # # # plt.show()
    # #
    # # sol_rk2=np.zeros((1,Ndt))
    # # for i in range(Ndt):
    # #     var=var0
    # #     for j in range(NDt[i]):
    # #         var=RK_2(var,Dt[i],equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,0,0,0,0,0,0)
    # #     sol_rk2[0,i]=np.log10(np.max(np.abs(var[:nx,0]-solution)))
    # # np.save('example_9/sol_rk2',sol_rk2)
    # sol_rk2=np.load('example_9/sol_rk2.npy')
    # sol_rk2[sol_rk2>1]=1
    # sol_rk2[sol_rk2!=sol_rk2]=1
    # print('sol_rk2',sol_rk2)
    # # plt.plot(Dt,sol_rk2[0,:],label='RK2')
    # # # plt.show()
    # #
    # # sol_devito=np.zeros((1,Ndt))
    # # for i in range(Ndt):
    # #     var=devito_ord2_t(var0[:nx],NDt[i],Dt[i],dx,c2,source_type,0,0)
    # #     sol_devito[0,i]=np.log10(np.max(np.abs(var-np.expand_dims(solution,axis=1))))
    # # sol_devito[sol_devito>1]=1
    # # np.save('example_9/sol_devito',sol_devito)
    # sol_devito=np.load('example_9/sol_devito.npy')
    # sol_devito[sol_devito!=sol_devito]=1
    # print('sol_devito',sol_devito)
    # # plt.plot(Dt,sol_devito[0,:],label='2nd in time')
    # # plt.legend()
    # # plt.savefig('example_9/lower_order.pdf')
    # # plt.show()
    # #
    # #
    # #
    # # sdfasdf

    # vals=spectral_dist(var0,equ,dim,abc,delta,beta0,ord,dx,c2,nx,0,var0*0,0,source_type,0,0)
    # vals=np.concatenate((vals,np.conj(vals),np.array([-beta0*pow((delta-dx/2)/delta,2)]),vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2,-vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2))
    # np.save('example_9/vals',vals)
    # vals=np.load('example_9/vals.npy')
    # gamma,c,d,a_e=ellipse_properties(vals,1)
    #
    # sol_faber=np.zeros((m_max-2,Ndt))
    # for i in range(Ndt):
    #     print(i)
    #     coefficients_faber=np.array(Faber_approx_coeff(m_max,gamma*Dt[i],c*Dt[i],d*Dt[i]).tolist(),dtype=np.float_)
    #     for j in range(m_max-2):
    #         var=var0
    #         for l in range(NDt[i]):
    #             var=Faber_approx(var,j+3,gamma,c,d,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,coefficients_faber,0,0)
    #         sol_faber[j,i]=np.log10(np.max(np.abs(np.array(var[:nx,0].tolist(),dtype=complex)-solution)))
    # np.save('example_9/sol_faber',sol_faber)
    sol_faber=np.load('example_9/sol_faber.npy')
    sol_faber[sol_faber>1]=1
    sol_faber[sol_faber!=sol_faber]=1
    print('sol_faber',sol_faber)
    x_axis, Grau = np.meshgrid(Dt, np.linspace(2, m_max, m_max-2))
    plt.contourf(x_axis,Grau,sol_faber,levels=100)
    plt.colorbar()
    plt.xlabel('$\Delta$t',fontsize=18)
    plt.ylabel('Polynomial degree',fontsize=18)
    plt.title('Faber approximation error',fontsize=20)
    plt.savefig('example_9/faber.pdf')
    plt.show()

    plt.plot(sol_faber[:,10])
    plt.show()

    error=sol_faber
    # print(np.expand_dims(np.linspace(2,m_max,m_max-1),1))
    # print(np.expand_dims(np.arange(50,1),0))
    # operations=np.expand_dims(np.linspace(2,49,49-1),1).dot(np.expand_dims(np.arange(1,50),0))
    # aux=operations*0
    # for i in range(0,48):
    #     aux[i,:]=operations[-(i+1),:]
    # aux[47,:]=operations[0,:]
    # operations=aux
    # print(operations)
    # tol=-5
    # print('operations ',np.min(operations[error<tol]))
    # aux=np.argmin(operations[error<tol])
    # degree=np.expand_dims(np.linspace(0,m_max-1,m_max),1).dot(np.ones((1,n_s)))
    # print('# steps',np.min(operations[error<tol])/degree[error<tol][aux])
    # print('degree',degree[error<tol][aux])


    # sol_rk=np.zeros((m_max-2,Ndt))
    # for i in range(Ndt):
    #     print(i)
    #     for j in range(m_max-2):
    #         var=var0
    #         for l in range(NDt[i]):
    #             var=RK_op(var,1,Dt[i],equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,j+2,0,0)
    #         sol_rk[j,i]=np.log10(np.max(np.abs(var[:nx,0]-solution)))
    # np.save('example_9/sol_rk',sol_rk)
    sol_rk=np.load('example_9/sol_rk.npy')
    sol_rk[sol_rk>1]=1
    sol_rk[sol_rk!=sol_rk]=1
    print('sol_rk',sol_rk)
    # x_axis, Grau = np.meshgrid(x_axis, Grau,Dt, np.linspace(2, m_max, m_max - 2))
    plt.contourf(x_axis,Grau,sol_rk,levels=100)
    plt.colorbar()
    plt.xlabel('$\Delta$t',fontsize=18)
    plt.ylabel('Polynomial degree',fontsize=18)
    plt.title('RK approximation error',fontsize=20)
    plt.savefig('example_9/rk.pdf')
    plt.show()

    plt.plot(sol_rk[:,10])
    plt.show()


def example_10(a,dx,equ,dim,abc,delta,beta0,ord,T,Ndt,m_max):
    # first order 1D homogeneous scalar wave equation without source term and PML

    nx=round(a/dx)
    print(nx)

    X=np.expand_dims(np.linspace(dx,a,nx),1)

    np.save('example_10/X',X)

    c2=np.zeros((nx,1))+15

    x0=X[int(len(X)/2)]
    f=source_x_1D(x0=x0,rad=dx/2,X=np.matrix.flatten(X,'F'),nx=nx,abc=abc)
    f0=24
    t0=1.2/f0
    # T0=0.14
    T0=0.2

    dt=dx/np.max(np.sqrt(c2))/8
    print(dt)
    Dt=np.linspace(dt,Ndt*dt,Ndt)
    NDt=np.ceil(T/Dt).astype(int)
    Dt=T/NDt

    if abc==1:
        var0=np.zeros((nx*3,1))
    else:
        var0=np.zeros((nx*2,1))

    RK_ref=np.zeros((len(X),1))
    var=var0
    for i in range(NDt[0]):
        var=RK_7_source(var,Dt[0],equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,f,t0+np.floor(Dt[0]*i/T0)*T0,f0,i)
    RK_ref[:,0]=var[:nx,0]
    solution=RK_ref

    sol_rk=np.zeros((m_max-2,Ndt))
    for i in range(Ndt):
        print(i)
        for j in range(m_max-2):
            var=var0
            for l in range(NDt[i]):
                var=RK_op(var,1,Dt[i],equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,j+2)
            sol_rk[j,i]=np.log10(np.max(np.abs(var[:nx,0]-solution)))
    np.save('example_10/sol_rk',sol_rk)
    plt.contourf(sol_rk,levels=100)
    print(sol_rk)
    plt.colorbar()
    plt.show()

    sol_rk2=np.zeros((1,Ndt))
    for i in range(Ndt):
        var=var0
        for j in range(NDt[i]):
            var=RK_2(var,Dt[i],equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0)
        sol_rk2[0,i]=np.log10(np.max(np.abs(var[:nx,0]-solution)))
    np.save('example_10/sol_rk2',sol_rk2)
    plt.plot(Dt,sol_rk2[0,:])
    plt.show()

    sol_devito=np.zeros((1,Ndt))
    for i in range(Ndt):
        var1=var0[:nx]
        var=var1+Dt[i]**2*dp2x(c2*var1,dx)
        for j in range(1,NDt[i]):
            var2=2*var-var1+Dt[i]**2*dp2x(c2*var,dx)
            var1=var
            var=var2
        sol_devito[0,i]=np.log10(np.max(np.abs(var-solution)))
    np.save('example_10/sol_devito',sol_devito)
    plt.plot(Dt,sol_devito[0,:])
    plt.show()


    plt.plot(solution)
    plt.show()
    sdfasdf

    if sum(f)<pow(10,-15):
        scale=0
    else:
        scale=Dt[0]
    u_k=g_approx(f,3,f0,t0,0)

    var0=np.vstack((var0,np.array([[0],[0],[1]])))

    def lin_op_H(var):
        var=np.expand_dims(var,axis=1)
        if scale==0:
            return op_H(var,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0)
        else:
            return op_H_extended(var,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,scale,u_k)

    H=LinearOperator(shape=(len(var0),len(var0)),matvec=lin_op_H)

    # vals,vecs=eigs(H,100,which='LM')
    # vals=np.concatenate((vals,np.conj(vals),np.array([-beta0*pow((delta-dx/2)/delta,2)]),vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2,-vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2))
    # np.save('example_10/vals',vals)
    vals=np.load('example_10/vals.npy')
    gamma,c,d,a_e=ellipse_properties(vals,1)

    sol_faber=np.zeros((m_max-2,Ndt))
    for i in range(Ndt):
        print(i)
        coefficients_faber=np.array(Faber_approx_coeff(m_max,gamma*Dt[i],c*Dt[i],d*Dt[i]).tolist(),dtype=np.float_)
        for j in range(m_max-2):
            var=var0
            for l in range(NDt[i]):
                u_k=g_approx(f,3,f0,t0+np.floor(Dt[i]*l/T0)*T0,Dt[i]*l)
                var=Faber_approx(var,j+3,gamma,c,d,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,coefficients_faber,Dt[i],u_k)
            sol_faber[j,i]=np.log10(np.max(np.abs(np.array(var[:nx,0].tolist(),dtype=complex)-solution)))
        # print(sol_faber[:,0])
        # asfasgs
    np.save('example_10/sol_faber',sol_faber)
    sol_faber=np.load('example_10/sol_faber.npy')
    sol_faber[sol_faber>10]=10
    print(sol_faber)
    plt.contourf(sol_faber,levels=100)
    plt.colorbar()
    plt.show()
    # dfgdsfgsdf


def example_11(a,dx,equ,dim,abc,delta,beta0,ord,T,Ndt,m_max):
    # solution of 1d acoustic equation with font term and constructed solutions without PML to validate accuracy
    # of the methods: RK7, Faber, RK-High order, RK2, Devito

    nx=round(a/dx)
    print(nx)

    X=np.expand_dims(np.linspace(dx,a,nx),1)

    np.save('example_11/X',X)

    c2=np.zeros((nx,1))+15
    c2[int(nx/2):]=3

    # f,f1=example_11_source_x(points=np.array([-dx,-dx/2,0,a+dx/2,a+dx,a+dx*3/2,a+2*dx]),x=X,c2=c2)
    source_type='11'

    # example_11_test_4ord_dx(points=np.array([-dx,-dx/2,0,a+dx/2,a+dx,a+dx*3/2]),x=X)

    dt=dx/np.max(np.sqrt(c2))/8
    print(dt)
    Dt=np.linspace(dt,Ndt*dt,Ndt)
    NDt=np.ceil(T/Dt).astype(int)
    Dt=T/NDt
    print(NDt[0])

    var0=ini_var0(abc,nx,0,0,0)

    # solution=example_11_solution(points=np.array([-dx,-dx/2,0,a+dx/2,a+dx,a+dx*3/2,a+2*dx]),x=X,t=T)
    # np.save('example_11/solution',solution)
    solution=np.load('example_11/solution.npy')
    plt.plot(solution)
    plt.show()

    # RK_ref=np.zeros((1,Ndt))
    # for i in range(Ndt):
    #     var=var0
    #     for j in range(NDt[i]):
    #         var=RK_7_source(var=var,dt=Dt[i],equ=equ,dim=dim,abc=abc,delta=delta,beta0=beta0,ord=ord,dx=dx,c2=c2,nx=nx+1,ny=0,f=f,t0=0,f0=0,i=j,source_type=source_type,f1=f1)
    #     RK_ref[0,i]=np.log10(np.max(np.abs(var[:nx,0]-solution)))
    # np.save('example_11/RK_ref',RK_ref)
    RK_ref=np.load('example_11/RK_ref.npy')
    RK_ref[RK_ref>1]=1
    print('RK_ref',RK_ref)
    plt.plot(Dt,RK_ref[0,:])
    plt.show()

    # sol_rk2=np.zeros((1,Ndt))
    # for i in range(Ndt):
    #     var=var0
    #     for j in range(NDt[i]):
    #         var=RK_2(var,Dt[i],equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,j,f,0,0,source_type,f1)
    #     sol_rk2[0,i]=np.log10(np.max(np.abs(var[:nx,0]-solution)))
    # np.save('example_11/sol_rk2',sol_rk2)
    sol_rk2=np.load('example_11/sol_rk2.npy')
    sol_rk2[sol_rk2>1]=1
    print('sol_rk2',sol_rk2)
    plt.plot(Dt,sol_rk2[0,:])
    plt.show()

    # sol_devito=np.zeros((1,Ndt))
    # for i in range(Ndt):
    #     var=devito_ord2_t(var0[:nx],NDt[i],Dt[i],dx,c2,source_type,f[:nx],f1[:nx])
    #     sol_devito[0,i]=np.log10(np.max(np.abs(var-np.expand_dims(solution,axis=1))))
    # np.save('example_11/sol_devito',sol_devito)
    sol_devito=np.load('example_11/sol_devito.npy')
    sol_devito[sol_devito>1]=1
    print('sol_devito',sol_devito)
    plt.plot(Dt,sol_devito[0,:])
    plt.show()

    # vals=spectral_dist(var0,equ,dim,abc,delta,beta0,ord,dx,c2,nx,0,f,3,source_type,Dt[0],f1)
    # vals=np.concatenate((vals,np.conj(vals),np.array([-beta0*pow((delta-dx/2)/delta,2)]),vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2,-vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2))
    # np.save('example_11/vals',vals)
    # vals=np.load('example_11/vals.npy')
    # gamma,c,d,a_e=ellipse_properties(vals,1)
    #
    # sol_faber=np.zeros((m_max-2,Ndt))
    # for i in range(Ndt):
    #     print(i)
    #     coefficients_faber=np.array(Faber_approx_coeff(m_max,gamma*Dt[i],c*Dt[i],d*Dt[i]).tolist(),dtype=np.float_)
    #     for j in range(m_max-2):
    #         ext=np.zeros((j+3,1))
    #         ext[j+2,0]=1
    #         var=np.vstack((var0,ext))
    #         for l in range(NDt[i]):
    #             u_k=g_approx(f=f,p=j+3,f0=0,t0=0,t=l*Dt[i],source_type=source_type,f1=f1)
    #             eta=np.max(np.sum(np.abs(u_k),axis=0))
    #             u_k=pow(2,-np.log2(eta))*u_k
    #             var[-1]=pow(2,np.log2(eta))
    #             var=Faber_approx(var,j+3,gamma,c,d,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,coefficients_faber,Dt[i],u_k)
    #             var[-(j+3):]=0
    #         sol_faber[j,i]=np.log10(np.max(np.abs(np.array(var[:nx,0].tolist(),dtype=complex)-solution)))
    # np.save('example_11/sol_faber',sol_faber)
    sol_faber=np.load('example_11/sol_faber.npy')
    sol_faber[sol_faber>1]=1
    print('sol_faber',sol_faber)
    plt.contourf(sol_faber,levels=100)
    plt.colorbar()
    plt.show()

    # sol_rk=np.zeros((m_max-2,Ndt))
    # for i in range(Ndt):
    #     print(i)
    #     for j in range(m_max-2):
    #         ext=np.zeros((j+3,1))
    #         ext[j+2,0]=1
    #         var=np.vstack((var0,ext))
    #         for l in range(NDt[i]):
    #             u_k=g_approx(f=f,p=j+3,f0=0,t0=0,t=l*Dt[i],source_type=source_type,f1=f1)
    #             eta=np.max(np.sum(np.abs(u_k),axis=0))
    #             u_k=pow(2,-np.log2(eta))*u_k
    #             var[-1]=pow(2,np.log2(eta))
    #             var=RK_op(var,1,Dt[i],equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,j+2,Dt[i],u_k)
    #             var[-(j+3):]=0
    #         sol_rk[j,i]=np.log10(np.max(np.abs(var[:nx,0]-solution)))
    # np.save('example_11/sol_rk',sol_rk)
    sol_rk=np.load('example_11/sol_rk.npy')
    sol_rk[sol_rk>1]=1
    plt.contourf(sol_rk,levels=100)
    print('sol_rk',sol_rk)
    plt.colorbar()
    plt.show()


def example_12(a,dx,equ,dim,abc,delta,beta0,ord,T,Ndt,m_max):
    # solution of 1d acoustic equation with font term and constructed solutions with PML to validate accuracy
    # of the methods: RK7, Faber, RK-High order, RK2, Devito

    nx=round(a/dx)
    print(nx)

    X=np.expand_dims(np.linspace(dx,a,nx),1)

    np.save('example_11/X',X)

    c2=np.zeros((nx,1))+15
    c2[int(nx/2):]=3

    # f,f1=example_11_source_x(points=np.array([-dx,-dx/2,0,a+dx/2,a+dx,a+dx*3/2,a+2*dx]),x=X,c2=c2)
    # source_type='11'
    x0=X[int(len(X)/2)]
    f=source_x_1D(x0=x0,rad=dx/2,X=np.matrix.flatten(X,'F'),nx=nx,abc=abc)
    f0=24
    t0=1.2/f0
    T0=0.2
    f1=f*0
    source_type=0

    # example_11_test_4ord_dx(points=np.array([-dx,-dx/2,0,a+dx/2,a+dx,a+dx*3/2]),x=X)

    dt=dx/np.max(np.sqrt(c2))/8
    print(dt)
    Dt=np.linspace(dt,Ndt*dt,Ndt)
    NDt=np.ceil(T/Dt).astype(int)
    Dt=T/NDt
    print(NDt[0])

    var0=ini_var0(abc,nx,0,0,0)

    var=var0
    for j in range(NDt[0]):
        var=RK_7_source(var=var,dt=Dt[0],equ=equ,dim=dim,abc=abc,delta=delta,beta0=beta0,ord=ord,dx=dx,c2=c2,nx=nx+1,ny=0,f=f,t0=t0+np.floor(Dt[0]*j/T0)*T0,f0=f0,i=j,source_type=source_type,f1=f1)
        # var=RK_7_source(var=var,dt=Dt[0],equ=equ,dim=dim,abc=abc,delta=delta,beta0=beta0,ord=ord,dx=dx,c2=c2,nx=nx+1,ny=0,f=f,t0=t0,f0=f0,i=j,source_type=source_type,f1=f1)
    RK_ref=var[:nx]
    np.save('example_12/RK_ref',RK_ref)
    RK_ref=np.load('example_12/RK_ref.npy')

    plt.plot(RK_ref[:,0])
    plt.show()
    fdsgdsgf
    sol_rk2=np.zeros((1,Ndt))
    for i in range(Ndt):
        var=var0
        for j in range(NDt[i]):
            var=RK_2(var=var,dt=Dt[i],equ=equ,dim=dim,abc=abc,delta=delta,beta0=beta0,ord=ord,dx=dx,c2=c2,nx=nx+1,ny=0,i=j,f=f,f0=f0,t0=t0+np.floor(Dt[0]*j/T0)*T0,source_type=source_type,f1=f1)
        sol_rk2[0,i]=np.log10(np.max(np.abs(var[:nx,0]-RK_ref)))
        plt.plot(var[:nx,0])
        plt.plot(RK_ref)
        plt.show()
        asdfsa
    np.save('example_12/sol_rk2',sol_rk2)
    sol_rk2=np.load('example_12/sol_rk2.npy')
    print('sol_rk2',sol_rk2)
    plt.plot(Dt,sol_rk2[0,:])
    plt.show()

    # for PML not used
    # sol_devito=np.zeros((1,Ndt))
    # for i in range(Ndt):
    #     var=devito_ord2_t(var0[:nx],NDt[i],Dt[i],dx,c2,source_type,f[:nx],f1[:nx])
    #     sol_devito[0,i]=np.log10(np.max(np.abs(var-np.expand_dims(solution,axis=1))))
    # np.save('example_11/sol_devito',sol_devito)
    # sol_devito=np.load('example_11/sol_devito.npy')
    # print('sol_devito',sol_devito)
    # plt.plot(Dt,sol_devito[0,:])
    # plt.show()

    # vals=spectral_dist(var0,equ,dim,abc,delta,beta0,ord,dx,c2,nx,0,f,3,source_type,Dt[0],f1)
    # vals=np.concatenate((vals,np.conj(vals),np.array([-beta0*pow((delta-dx/2)/delta,2)]),vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2,-vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2))
    # np.save('example_12/vals',vals)
    # vals=np.load('example_12/vals.npy')
    # gamma,c,d,a_e=ellipse_properties(vals,1)
    #
    var0=np.vstack((var0,np.array([[0],[0],[1]])))
    #
    # sol_faber=np.zeros((m_max-2,Ndt))
    # for i in range(Ndt):
    #     print(i)
    #     coefficients_faber=np.array(Faber_approx_coeff(m_max,gamma*Dt[i],c*Dt[i],d*Dt[i]).tolist(),dtype=np.float_)
    #     for j in range(m_max-2):
    #         var=var0
    #         for l in range(NDt[i]):
    #             u_k=g_approx(f=f,p=3,f0=0,t0=0,t=l*Dt[i],source_type=source_type,f1=f1)
    #             var=Faber_approx(var,j+3,gamma,c,d,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,coefficients_faber,Dt[i],u_k)
    #             var[-3]=0
    #             var[-2]=0
    #             var[-1]=1
    #         sol_faber[j,i]=np.log10(np.max(np.abs(np.array(var[:nx,0].tolist(),dtype=complex)-RK_ref)))
    # np.save('example_12/sol_faber',sol_faber)
    # sol_faber=np.load('example_12/sol_faber.npy')
    # sol_faber[sol_faber>10]=10
    # print('sol_faber',sol_faber)
    # plt.contourf(sol_faber,levels=100)
    # plt.colorbar()
    # plt.show()

    sol_rk=np.zeros((m_max-2,Ndt))
    for i in range(Ndt):
        print(i)
        for j in range(m_max-2):
            var=var0
            for l in range(NDt[i]):
                u_k=g_approx(f=f,p=3,f0=0,t0=0,t=l*Dt[i],source_type=source_type,f1=f1)
                var=RK_op(var,1,Dt[i],equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,j+2,Dt[i],u_k)
                var[-3]=0
                var[-2]=0
                var[-1]=1
            sol_rk[j,i]=np.log10(np.max(np.abs(var[:nx,0]-RK_ref)))
            if j==m_max-3:
                plt.plot(var[:nx,0])
                plt.plot(RK_ref)
                plt.show()
    np.save('example_12/sol_rk',sol_rk)
    sol_rk=np.load('example_12/sol_rk.npy')
    sol_rk[sol_rk>10]=10
    plt.contourf(sol_rk,levels=100)
    print('sol_rk',sol_rk)
    plt.colorbar()
    plt.show()


def example_13(a,dx,equ,dim,abc,delta,beta0,ord,T,Ndt,m_max):
    # error estimation of 1D example of faber approximation, compared with the real error

    nx=round(a/dx)
    print(nx)

    X=np.expand_dims(np.linspace(dx,a,nx),1)

    np.save('example_13/X',X)

    c2=np.zeros((nx,1))+15

    dt=dx/np.max(np.sqrt(c2))/10
    print(dt)
    Dt=np.linspace(dt,Ndt*dt,Ndt)
    NDt=np.ceil(T/Dt).astype(int)
    Dt=T/NDt

    source_type="8"
    var0=ini_var0(abc,nx,X,source_type,a)

    # solution=example_8_solution(X,a,T,np.sqrt(c2[0]))[:,0]
    # np.save('example_13/solution',solution)
    solution=np.load('example_13/solution.npy')

    # vals=spectral_dist(var0,equ,dim,abc,delta,beta0,ord,dx,c2,nx,0,var0*0,0,source_type,0,0)
    # vals=np.concatenate((vals,np.conj(vals),np.array([-beta0*pow((delta-dx/2)/delta,2)]),vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2,-vals.imag*1j-beta0*pow((delta-dx/2)/delta,2)/2))
    # np.save('example_13/vals',vals)
    # vals=np.load('example_13/vals.npy')
    # gamma,c,d,a_e=ellipse_properties(vals,1)
    #
    # sol_faber=np.zeros((m_max-2,Ndt))
    # error_bound=np.zeros((m_max-2,Ndt))
    # for i in range(Ndt):
    #     print(i)
    #     coefficients_faber=np.array(Faber_approx_coeff(m_max,gamma*Dt[i],c*Dt[i],d*Dt[i]).tolist(),dtype=np.float_)
    #     for j in range(m_max-2):
    #         var=var0
    #         for l in range(NDt[i]):
    #             var=Faber_approx(var,j+3,gamma,c,d,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,coefficients_faber,0,0)
    #         sol_faber[j,i]=np.log10(np.max(np.abs(np.array(var[:nx,0].tolist(),dtype=complex)-solution)))
    #     error_bound[:,i]=np.log10(err_estimate(m_max,gamma,c,d,a_e,coefficients_faber))[2:]
    # np.save('example_13/sol_faber',sol_faber)
    sol_faber=np.load('example_13/sol_faber.npy')
    sol_faber[sol_faber>1]=1
    sol_faber[sol_faber<-16]=-16
    print('sol_faber',sol_faber)
    plt.contourf(sol_faber,levels=100)
    plt.colorbar()
    plt.show()

    # np.save('example_13/error_bound',error_bound)
    error_bound=np.load('example_13/error_bound.npy')
    error_bound[error_bound>1]=1
    error_bound[error_bound<-3.90312684]=-3.90312684
    print('error_bound',error_bound)
    plt.contourf(error_bound,levels=100)
    plt.colorbar()
    plt.show()

    plt.contourf(error_bound-sol_faber/np.max(np.abs(solution)),levels=100)
    plt.colorbar()
    plt.show()


def example_14(a,dx,equ,dim,abc,delta,beta0,ord):
    # examples os spectral distribution and its limit pentagon

    nx=round(a/dx)
    print(nx)

    X=np.expand_dims(np.linspace(dx,a,nx),1)

    np.save('example_14/X',X)

    c2=np.zeros((nx,1))+15  # example a
    # c2=np.zeros((nx,1))+15 # example b
    # c2[int(nx/2):]=0.5     # example b

    source_type="8"
    var0=ini_var0(abc,nx,X,source_type,a)

    vals=spectral_dist(var0,equ,dim,abc,delta,beta0,ord,dx,c2,nx,0,var0*0,0,source_type,0,0)
    np.save('example_14/vals_dx_'+str(dx)+'_a',vals)
    vals=np.load('example_14/vals_dx_'+str(dx)+'_a.npy')

    print(np.max(np.abs(vals.imag)))
    points=np.array([-beta0*pow((delta-dx/2)/delta,2),-beta0*pow((delta-dx/2)/delta,2)/2+1j*np.sqrt(np.max(c2))/(10*dx)*24,-beta0*pow((delta-dx/2)/delta,2)/2-1j*np.sqrt(np.max(c2))/(10*dx)*24,-1j*np.sqrt(np.max(c2))/(10*dx)*24,1j*np.sqrt(np.max(c2))/(10*dx)*24])

    ellipse_properties(points,1)
    plt.plot(vals.real,vals.imag,'og')


    plt.xlabel('real axis', fontsize=18)
    plt.ylabel('imaginari axis', fontsize=18)
    plt.title('Eigenvalues minimum ellipse',fontsize=20)
    # plt.legend(loc='lower left')
    plt.show()


def example_14_0(a,dx,equ,dim,abc,delta,beta0,ord):
    # function to find the optimal gamma and c to minimize the error

    nx=round(a/dx)
    print(nx)

    c2=np.zeros((nx,1))+15  # example a
    # c2=np.zeros((nx,1))+15
    # c2[int(nx/2):]=0.5

    vals=np.array([-beta0*pow((delta-dx/2)/delta,2),-beta0*pow((delta-dx/2)/delta,2)/2+1j*np.sqrt(np.max(c2))/(10*dx)*24,-beta0*pow((delta-dx/2)/delta,2)/2-1j*np.sqrt(np.max(c2))/(10*dx)*24,-1j*np.sqrt(np.max(c2))/(10*dx)*24,1j*np.sqrt(np.max(c2))/(10*dx)*24])
    gamma,c,d,a_e=ellipse_properties(vals/100,1)
    plt.show()
    print(a_e)
    b_e=gamma-a_e

    if a_e>b_e:
        b_e=np.linspace(b_e,a_e,10000)
        c=np.sqrt(a_e**2-b_e**2)
    else:
        a_e=np.linspace(a_e,b_e,10000)
        c=np.sqrt(b_e**2-a_e**2)
    gamma=(a_e+b_e)/2

    estimative=np.zeros((100,10000))
    optimum=np.zeros((100))
    for m in range(100):
        for i in range(10000):

            estimative[m,i]=math.log10(limit_m_f(m+1,gamma[i],c[i],d))
        optimum[m]=np.argmin(estimative[m,:])
    print(optimum)

    np.save('example_14_0/estimative',estimative)
    estimative=np.load('example_14_0/estimative.npy')
    plt.contourf(estimative,levels=100)
    plt.colorbar()

    plt.xlabel('range of c',fontsize=18)
    plt.ylabel('Polynomial degree',fontsize=18)
    plt.title('Faber approximation error bound',fontsize=20)
    plt.savefig('example_14_0/bound.pdf')
    plt.show()

    # conclusion: the optimal strategy minimizing the theoretical bound of the Faber approximation is to choose the
    # minimum gamma no matter how be c can be


def example_15(a,dx,equ,dim,abc,delta,beta0,ord,T,m_max,example,source_type):
    # error estimation of 1D example of faber approximation, compared with the real error in order to obtain cond_P

    nx=round(a/dx)
    print(nx)

    X=np.expand_dims(np.linspace(dx,a,nx),1)

    # np.save('example_15/X',X)

    if example=='a':
        c2=np.zeros((nx,1))+15
    else:
        c2=np.zeros((nx,1))+0.5

    var0=ini_var0(abc,nx,X,source_type,a)

    # plt.plot(example_8_solution(X,a,0.3,np.sqrt(c2[0]))[:,0])
    # plt.show()
    # sagsdfgs

    # vals=spectral_dist(var0,equ,dim,abc,delta,beta0,ord,dx,c2,nx,0,var0*0,0,source_type,0,0)
    # np.save('example_15/vals_dx_'+str(dx)+'_a',vals)
    # vals=np.load('example_15/vals_dx_'+str(dx)+'_a.npy')
    # vals=np.array([-beta0*pow((delta-dx/2)/delta,2),-beta0*pow((delta-dx/2)/delta,2)/2+1j*np.sqrt(np.max(c2))/(10*dx)*24,-beta0*pow((delta-dx/2)/delta,2)/2-1j*np.sqrt(np.max(c2))/(10*dx)*24,-1j*np.sqrt(np.max(c2))/(10*dx)*24,1j*np.sqrt(np.max(c2))/(10*dx)*24])
    vals1=np.array([-beta0*pow((delta-dx/2)/delta,2),-beta0*pow((delta-dx/2)/delta,2)+1j*np.sqrt(np.max(c2))/(10*dx)*24,-beta0*pow((delta-dx/2)/delta,2)-1j*np.sqrt(np.max(c2))/(10*dx)*24,-1j*np.sqrt(np.max(c2))/(10*dx)*24,1j*np.sqrt(np.max(c2))/(10*dx)*24])

    gamma,c,d,a_e=ellipse_properties(vals1,1)
    np.save('example_15/gamma_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example),gamma)
    np.save('example_15/c_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example),c)
    np.save('example_15/d_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example),d)
    np.save('example_15/a_e_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example),a_e)

    gamma=np.load('example_15/gamma_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example)+'.npy')
    c=np.load('example_15/c_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example)+'.npy')
    d=np.load('example_15/d_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example)+'.npy')
    a_e=np.load('example_15/a_e_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example)+'.npy')

    # plt.plot(vals.real,vals.imag,'ob')
    # plt.plot(vals1.real,vals1.imag,'og')

    sol_faber=np.zeros((m_max-2))

    coefficients_faber=np.array(Faber_approx_coeff(m_max+1,gamma*T,c*T,d*T).tolist(),dtype=np.float_)
    np.save('example_15/coeff_faber_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example),coefficients_faber)
    coefficients_faber=np.load('example_15/coeff_faber_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example)+'.npy')
    # plt.plot(np.log10(np.array(coefficients_faber.tolist(),dtype=complex)))
    plt.close()
    # plt.show()
    # safasdfsd
    for i in range(m_max-2):
        print(i)
        var=Faber_approx(var0,i+3,gamma,c,d,equ,dim,abc,delta,beta0,ord,dx,c2,nx+1,0,coefficients_faber,0,0)
        sol_faber[i]=np.log10(np.max(np.abs(np.array(var[:nx,0].tolist(),dtype=complex)-example_8_solution(X,a,T,np.sqrt(c2[0]))[:,0])))
    error_bound=np.log10(err_estimate(m_max,gamma,c,d,a_e,coefficients_faber)*np.sqrt(np.sum(pow(np.abs(var0),2))*dx))[2:]

    np.save('example_15/sol_faber_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example),sol_faber)
    np.save('example_15/error_bound_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example),error_bound)

    sol_faber=np.load('example_15/sol_faber_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example)+'.npy')
    error_bound=np.load('example_15/error_bound_dx_'+str(dx)+'_ic_'+str(source_type)+'_'+str(example)+'.npy')

    # sol_faber[sol_faber>1]=1

    # error_bound[error_bound>1]=1
    # error_bound[error_bound<-3.90312684]=-3.90312684

    f_list=np.zeros((m_max-2))
    for i in range(m_max-2):
        f_list[i]=np.log10(np.array(np.array([limit_m_f(i+2,gamma*T,c*T,d*T)]).tolist(),dtype=complex)*np.sqrt(np.sum(pow(np.abs(var0),2))*dx))[0]
    print('sol_faber',sol_faber)
    # plt.plot(example_8_solution(X,a,T,np.sqrt(c2[0]))[:,0])
    # print('error_bound',error_bound)

    aux=f_list-sol_faber
    interval=5
    print(np.min(aux[:interval]))
    aux1=error_bound-sol_faber
    print(np.min(aux1[:interval]))

    error_bound=error_bound-np.min(aux1[:interval])
    f_list=f_list-np.min(aux[:interval])

    sol_faber[sol_faber<-16]=-16
    error_bound[error_bound<-16]=-16
    f_list[f_list<-16]=-16

    plt.plot(sol_faber,'r',label='Real error')
    plt.plot(error_bound,'g',label='Our estimation')
    plt.plot(f_list,'b',label='Theoretical estimation')
    plt.xlabel('Polynomial degree', fontsize=18)
    plt.ylabel('Error ($log_{10}$)', fontsize=18)
    plt.title('Error estimation',fontsize=20)
    plt.legend(loc='lower left')
    plt.savefig('example_15/errors_'+str(dx)+'_a.pdf')
    plt.show()



mp.mp.dps=200


example_1()

# example_2(a=16,b=10,dx=1.0,equ='scalar',dim=2,abc=1,delta=2.5,beta0=30,ord=2)
# example_3(a=10,b=8,dx=0.0625,n_dx=1,equ='scalar',dim=2,abc=1,delta=2.5,beta0=30,ord=2)
# cond_P0=np.array([1.171014018112446625e-01,2.643611346841615896e-03,1.755659207625759430e-04])
# example_4(a=8,b=10,dx=2,n_dx=3,equ='scalar',dim=2,abc=1,delta=2.5,beta0=30,ord=2,cond_P=cond_P0)
# example_5()
# example_6(a=10,b=8,dx=0.08,equ='scalar',dim=2,abc=1,delta=1,beta0=30,ord=2,T=1)
# example_8(a=20,dx=0.02,equ='scalar',dim=1,abc=0,delta=5,beta0=30,ord=4,T=1.9)
# example_9(a=20,dx=0.01,equ='scalar',dim=1,abc=1,delta=1,beta0=30,ord=4,T=1.9,Ndt=50,m_max=50)
# example_10(a=20,dx=0.02,equ='scalar',dim=1,abc=0,delta=5,beta0=30,ord=4,T=1.9,Ndt=10,m_max=10)
# example_11(a=20,dx=0.01,equ='scalar',dim=1,abc=0,delta=5,beta0=30,ord=4,T=1.9,Ndt=30,m_max=30)
# example_12(a=20,dx=0.005,equ='scalar',dim=1,abc=1,delta=5,beta0=30,ord=4,T=1.9,Ndt=10,m_max=10)
# example_13(a=20,dx=0.02,equ='scalar',dim=1,abc=1,delta=1,beta0=30,ord=4,T=1.9,Ndt=1,m_max=10)
# example_14(a=20,dx=0.025,equ='scalar',dim=1,abc=1,delta=1,beta0=30,ord=4)
# example_14_0(a=20,dx=0.001,equ='scalar',dim=1,abc=1,delta=1,beta0=30,ord=4)
# example_15(a=10,dx=0.08,equ='scalar',dim=1,abc=1,delta=1,beta0=30,ord=4,T=0.01,m_max=70,example='a',source_type="8")

# example 15, set example='a', a=10, T=0.01
# source_type='8', at 0.0025 Faber coefficients reaches 10^15 of magnitude the error and is not longer reliable
# dx= [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
# [1.3209075812237097, 1.5058869335899059, 1.5735221280908802, 1.8824510732342388, 2.1125091834837537,2.5644865035899613]
# [0.9866781784240223, 1.3138613812345783, 1.720783027142268, 2.5665505356250464, 4.290629548429128, 7.92011631394195]

# source_type='15_1'
# [1.810135821430845, 1.7099339741174562, 1.7695246108720204,2.070649402913181, 2.32637054816281, 2.7084430133850557]
# [1.1234464508832904, 1.5179084217621288, 1.9107425221388814,2.753267269162375, 4.476732071453763, 7.920156250319403]

# source_type='15_2'
# [1.009229166238539, -0.050695103796850716, 0.01431157167495889, 0.32358454973228223, 0.5810701941463368, 0.963528989401464]
# [0.7982684964277762, -0.24272065615217808, 0.16187799597653685, 1.0077411271961072, 2.731783240497336, 6.175324962899509]

# a=20
# source_type='8'
# [1.9368200430122704, ]
# [1.2501306724647159, ]

# example (b) at 0.005 Faber coefficients reaches 10^13 of magnitude the error and is not longer reliable
# dx= [0.08, 0.04, 0.02, 0.01]
# [2.645601496872273, 2.4759719736732637, 2.3778534863575516, 2.5405739255660915]
# [1.4577652387655091, 1.9620644843064903, 2.6236410162909785, 4.198392308736333]



