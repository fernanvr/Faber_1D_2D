import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy import optimize
from auxiliar_functions import *
from numpy.linalg import matrix_power as m_pow
import mpmath as mp
import sys # to run with a Bash script


def app_H(lamb,theta_x,theta_y,spatial_discr,dim,equ,param):

    if dim==1:
        if spatial_discr=='4':
            if equ=='scalar':
                aux1=np.exp(-2j*theta_x)-np.exp(1j*theta_x)+27*(1-np.exp(-1j*theta_x))
                aux2=np.exp(-1j*theta_x)-np.exp(2j*theta_x)+27*(np.exp(1j*theta_x)-1)
                return lamb/24*np.array([[0,aux1],[aux2,0]])
            elif equ=='scalar_dx2':
                return lamb*np.array([[0,1],[-np.cos(2*theta_x)/6+8/3*np.cos(theta_x)-5/2,0]])
        elif spatial_discr=='8':
            if equ=='scalar':
                aux1=1225/1024*(1-np.exp(-1j*theta_x)+1/15*(np.exp(-2j*theta_x)-np.exp(1j*theta_x))+1/125*(np.exp(2j*theta_x)-np.exp(-3j*theta_x))+1/1715*(np.exp(-4j*theta_x)-np.exp(3j*theta_x)))
                aux2=1225/1024*(np.exp(1j*theta_x)-1+1/15*(np.exp(-1j*theta_x)-np.exp(2j*theta_x))+1/125*(np.exp(3j*theta_x)-np.exp(-2j*theta_x))+1/1715*(np.exp(-3j*theta_x)-np.exp(4j*theta_x)))
                return lamb*np.array([[0,aux1],[aux2,0]])
            elif equ=='scalar_dx2':
                return lamb*np.array([[0,1],[-1/280*np.cos(4*theta_x)+16/315*np.cos(3*theta_x)-2/5*np.cos(2*theta_x)+16/5*np.cos(theta_x)-205/72,0]])
        elif spatial_discr=='inf':
            if equ=='scalar':
                return lamb*np.array([[0,1j*theta_x],[1j*theta_x,0]])*mp.exp(0)
            elif equ=='scalar_dx2':
                return lamb*np.array([[0,1],[-theta_x**2,0]])*mp.exp(0)
    elif dim==2:
        if spatial_discr=='4':
            if equ=='scalar_dx2':
                return lamb*np.array([[0,1],[-np.cos(2*theta_x)/6+8/3*np.cos(theta_x)-np.cos(2*theta_y)/6+8/3*np.cos(theta_y)-5,0]])
            else:
                aux1_x=np.exp(-2j*theta_x)-np.exp(1j*theta_x)+27*(1-np.exp(-1j*theta_x))
                aux1_y=np.exp(-2j*theta_y)-np.exp(1j*theta_y)+27*(1-np.exp(-1j*theta_y))
                aux2_x=np.exp(-1j*theta_x)-np.exp(2j*theta_x)+27*(np.exp(1j*theta_x)-1)
                aux2_y=np.exp(-1j*theta_y)-np.exp(2j*theta_y)+27*(np.exp(1j*theta_y)-1)
                if equ=='scalar':
                    return lamb/24*np.array([[0,aux1_x,aux1_y],[aux2_x,0,0],[aux2_y,0,0]])
                elif equ=='elastic':
                    return lamb*np.array([[0,0, 1/(2*param[1]+param[2])*aux1_x,1/(2*param[1]+param[2])*aux1_y,0],
                                          [0,0,0, 1/(2*param[1]+param[2])*aux2_x,1/(2*param[1]+param[2])*aux2_y],
                                          [param[0]*aux2_x,param[2]*param[0]/(2*param[1]+param[2])*aux1_y,0,0,0],
                                          [param[1]*param[0]/(2*param[1]+param[2])*aux2_y,param[1]*param[0]/(2*param[1]+param[2])*aux1_x,0,0,0],
                                          [param[2]*param[0]/(2*param[1]+param[2])*aux2_x,param[0]*aux1_y,0,0,0]])*np.sqrt((2*param[1]+param[2])/param[0])
        elif spatial_discr=='8':
            if equ=='scalar_dx2':
                return lamb*np.array([[0,1],[-1/280*np.cos(4*theta_x)+16/315*np.cos(3*theta_x)-2/5*np.cos(2*theta_x)+16/5*np.cos(theta_x)+
                    -1/280*np.cos(4*theta_y)+16/315*np.cos(3*theta_y)-2/5*np.cos(2*theta_y)+16/5*np.cos(theta_y)-205/36,0]])
            else:
                aux1_x=1225/1024*(1-np.exp(-1j*theta_x)+1/15*(np.exp(-2j*theta_x)-np.exp(1j*theta_x))+1/125*(np.exp(2j*theta_x)-np.exp(-3j*theta_x))+1/1715*(np.exp(-4j*theta_x)-np.exp(3j*theta_x)))
                aux1_y=1225/1024*(1-np.exp(-1j*theta_y)+1/15*(np.exp(-2j*theta_y)-np.exp(1j*theta_y))+1/125*(np.exp(2j*theta_y)-np.exp(-3j*theta_y))+1/1715*(np.exp(-4j*theta_y)-np.exp(3j*theta_y)))
                aux2_x=1225/1024*(np.exp(1j*theta_x)-1+1/15*(np.exp(-1j*theta_x)-np.exp(2j*theta_x))+1/125*(np.exp(3j*theta_x)-np.exp(-2j*theta_x))+1/1715*(np.exp(-3j*theta_x)-np.exp(4j*theta_x)))
                aux2_y=1225/1024*(np.exp(1j*theta_y)-1+1/15*(np.exp(-1j*theta_y)-np.exp(2j*theta_y))+1/125*(np.exp(3j*theta_y)-np.exp(-2j*theta_y))+1/1715*(np.exp(-3j*theta_y)-np.exp(4j*theta_y)))
            if equ=='scalar':
                return lamb*np.array([[0,aux1_x,aux1_y],[aux2_x,0,0],[aux2_y,0,0]])
            elif equ=='elastic':
                return lamb*np.array([[0,0, 1/(2*param[1]+param[2])*aux1_x,1/(2*param[1]+param[2])*aux1_y,0],
                                          [0,0,0, 1/(2*param[1]+param[2])*aux2_x,1/(2*param[1]+param[2])*aux2_y],
                                          [param[0]*aux2_x,param[2]*param[0]/(2*param[1]+param[2])*aux1_y,0,0,0],
                                          [param[1]*param[0]/(2*param[1]+param[2])*aux2_y,param[1]*param[0]/(2*param[1]+param[2])*aux1_x,0,0,0],
                                          [param[2]*param[0]/(2*param[1]+param[2])*aux2_x,param[0]*aux1_y,0,0,0]])*np.sqrt((2*param[1]+param[2])/param[0])
        elif spatial_discr=='inf':
            if equ=='scalar':
                return lamb*np.array([[0,1j*theta_x,1j*theta_y],[1j*theta_x,0,0],[1j*theta_y,0,0]])*mp.exp(0)
            elif equ=='scalar_dx2':
                return lamb*np.array([[0,1],[-(theta_x**2+theta_y**2),0]])*mp.exp(0)


def radio(A):

    A=np.array(A.tolist(),dtype=np.complex_)

    return np.max(np.abs(np.linalg.eigvals(A)))


def radio_rk2(theta,phi,lamb,spatial_discr,dim,equ,param):

    result=0
    for i in range(len(theta)):
        for j in range(len(phi)):
            H=app_H(lamb,theta[i]*np.cos(phi[j]),theta[i]*np.sin(phi[j]),spatial_discr,dim,equ,param)
            rk2=np.identity(len(H))*mp.exp(0)+H+m_pow(H,2)/2+m_pow(H,3)/4
            result=np.max(np.array([result,radio(rk2)]))

    return result


def radio_rk4(theta,phi,lamb,spatial_discr,dim,equ,param):

    result=0
    for i in range(len(theta)):
        for j in range(len(phi)):
            H=app_H(lamb,theta[i]*np.cos(phi[j]),theta[i]*np.sin(phi[j]),spatial_discr,dim,equ,param)
            rk4=np.identity(len(H))*mp.exp(0)+H+m_pow(H,2)/2+m_pow(H,3)/6+m_pow(H,4)/24
            result=np.max(np.array([result,radio(rk4)]))
    return result


def radio_rk7(theta,phi,lamb,spatial_discr,dim,equ,param):

    result=0
    for i in range(len(theta)):
        for j in range(len(phi)):
            H=app_H(lamb,theta[i]*np.cos(phi[j]),theta[i]*np.sin(phi[j]),spatial_discr,dim,equ,param)
            k1=H
            k2=H.dot(np.identity(len(H))+4/63*k1)
            k3=H.dot(np.identity(len(H))+(1/42*k1+1/14*k2))
            k4=H.dot(np.identity(len(H))+(1/28*k1+3/28*k3))
            k5=H.dot(np.identity(len(H))+(12551/19652*k1-48363/19652*k3+10976/4913*k4))
            k6=H.dot(np.identity(len(H))+(-36616931/27869184*k1+2370277/442368*k3-255519173/63700992*k4+226798819/445906944*k5))
            k7=H.dot(np.identity(len(H))+(-10401401/7164612*k1+47383/8748*k3-4914455/1318761*k4-1498465/7302393*k5+2785280/3739203*k6))
            k8=H.dot(np.identity(len(H))+(181002080831/17500000000*k1-14827049601/400000000*k3+23296401527134463/857600000000000*k4+2937811552328081/949760000000000*k5-243874470411/69355468750*k6+2857867601589/3200000000000*k7))
            k9=H.dot(np.identity(len(H))+(-228380759/19257212*k1+4828803/113948*k3-331062132205/10932626912*k4-12727101935/3720174304*k5+22627205314560/4940625496417*k6-268403949/461033608*k7+3600000000000/19176750553961*k8))
            rk7=np.identity(len(H))*mp.exp(0)+(95/2366*k1+3822231133/16579123200*k4+555164087/2298419200*k5+1279328256/9538891505*k6+5963949/25894400*k7+50000000000/599799373173*k8+28487/712800*k9)
            result=np.max(np.array([result,radio(rk7)]))

    return result


def radio_time_2step(theta,phi,lamb,spatial_discr,dim,equ,param):

    result=0
    for i in range(len(theta)):
        for j in range(len(phi)):
            if spatial_discr=='4':
                if dim==1:
                    time_2step=np.array([[2+pow(lamb,2)*(-mp.cos(2*theta[i])/6+mp.cos(theta[i])*8/3-5/2),-1],[1,0]])*mp.exp(0)
                elif dim==2:
                    kx=theta[i]*np.cos(phi[j])
                    ky=theta[i]*np.sin(phi[j])
                    time_2step=np.array([[2+pow(lamb,2)*((-mp.cos(2*kx)/6+mp.cos(kx)*8/3-5/2)+(-mp.cos(2*ky)/6+mp.cos(ky)*8/3-5/2)),-1],[1,0]])*mp.exp(0)
            elif spatial_discr=='inf':
                if dim==1:
                    time_2step=np.array([[2-pow(lamb*theta[i],2),-1],[1,0]])*mp.exp(0)
                elif dim==2:
                    kx=theta[i]*np.cos(phi[j])
                    ky=theta[i]*np.sin(phi[j])
                    time_2step=np.array([[2-pow(lamb,2)*(pow(kx,2)+pow(ky,2)),-1],[1,0]])*mp.exp(0)
            else:
                time_2step=np.array([[0,10j],[-10j,0]])
            result=np.max(np.array([result,radio(time_2step)]))

    return result


def radio_mtime_steps(lamb,grau,spatial_discr):

    coeff=coeff_mtime_steps(degree=grau)

    theta=np.linspace(0,np.pi,100)
    result=0
    for i in range(100):
        if spatial_discr=='4':
            D=pow(lamb,2)*(np.cos(2*theta[i])/6+np.cos(theta[i])*8/3-5/2)
        elif spatial_discr=='inf':
            D=-pow(lamb*theta[i],2)
        coeff_sum=np.sum(coeff)+1
        mtime_steps=np.zeros((grau+1,grau+1))
        mtime_steps[0,0]=coeff_sum*(1+D/2)
        for i in range(grau):
            mtime_steps[0,i+1]=-coeff[i]
            mtime_steps[i+1,i]=1
        result=np.max(np.array([result,radio(mtime_steps)]))

    return result


def radio_rkn(theta,phi,lamb,grau,spatial_discr,dim,equ,param):

    mu=1
    alpha=ssprk_alpha(mu,grau)

    result=0
    for i in range(len(theta)):
        for j in range(len(phi)):
            H=app_H(lamb,theta[i]*np.cos(phi[j]),theta[i]*np.sin(phi[j]),spatial_discr,dim,equ,param)
            rkn=alpha[0]*np.identity(len(H))
            for j in range(1,grau-1):
                rkn=rkn+m_pow(np.identity(len(H))*mp.exp(0)+mu*H,j)*alpha[j]
            rkn=rkn+m_pow(np.identity(len(H))+mu*H,grau)*alpha[grau-1]
            result=np.max(np.array([result,radio(rkn)]))

    return result


def radio_faber(theta,phi,lamb,grau,gamma,c,d,coeff,spatial_discr,dim,equ,param):

    c0=d/gamma*mp.exp(0)
    c1=c**2/(4*gamma**2)*mp.exp(0)

    result=0
    for i in range(len(theta)):
        for j in range(len(phi)):
            H=app_H(lamb,theta[i]*np.cos(phi[j]),theta[i]*np.sin(phi[j]),spatial_discr,dim,equ,param)
            faber=coeff[0]*np.identity(len(H))*mp.exp(0)
            F1=H/gamma-c0*np.identity(len(H))
            F1_fix=F1
            faber=faber+F1*coeff[1]
            F2=F1_fix.dot(F1)-2*c1*np.identity(len(H))
            faber=faber+F2*coeff[2]

            for j in range(3,grau):
                F0=F1
                F1=F2
                F2=F1_fix.dot(F1)-c1*F0
                faber=faber+F2*coeff[j]
            result=np.max(np.array([result,radio(faber)]))

    return result


def spectral_dist_stability(equ,dim,ord,param):

    if equ=='elastic':
        c2=np.sqrt(np.max((2*param[1]+param[2])*np.reciprocal(param[0])))
        if ord=='4':
            c2=c2*7.95
            beta0=7.95/100
        elif ord=='8':
            c2=c2*7.37
            beta0=7.37/100
    elif equ=='scalar':
        c2=np.sqrt(np.max(param))
        if ord=='4':
            if dim==1:
                c2=c2*2.34
                beta0=2.34/100
            elif dim==2:
                c2=c2*3.31
                beta0=3.31/100
        elif ord=='8':
            if dim==1:
                c2=c2*2.58
                beta0=2.58/100
            elif dim==2:
                c2=c2*3.65
                beta0=3.65/100
    elif equ=='scalar_dx2':
        c2=np.sqrt(np.max(param))
        if ord=='4':
            if dim==1:
                c2=c2*2.31
                beta0=2.31/100
            elif dim==2:
                c2=c2*3.27
                beta0=3.27/100
        elif ord=='8':
            if dim==1:
                c2=c2*2.55
                beta0=2.55/100
            elif dim==2:
                c2=c2*3.62
                beta0=3.62/100
    vals=np.array([-c2/10,-1j*c2,1j*c2])

    return vals


def stability(N_lamb,lamb_max,degree,spatial_discr,dim,equ,param):

    lamb=np.linspace(pow(10,-3),lamb_max,N_lamb)
    np.save('Stability/lamb_'+str(equ),lamb)

    rk2=lamb*0
    rk4=lamb*0
    rk7=lamb*0
    time_2step=lamb*0
    rkn=np.zeros((N_lamb,len(degree)))
    faber=np.zeros((N_lamb,len(degree)))+2
    # mtime_steps=np.zeros((N_lamb,len(degree)))

    n_points=20
    theta=np.linspace(0,np.pi,n_points)
    if dim==1:
        phi=np.array([0])
    elif dim==2:
        phi=np.linspace(0,2*np.pi*(1-1/10),10)

    ind_faber=0
    for i in range(N_lamb):
        print('0000000000000000000000000000000000000 i:',i)
        if ind_faber==0:
            # if dim==1:
            #     beta0=2.4/100
            #     # gamma,c,d,a_e=ellipse_properties(np.array([-beta0*lamb[i],-beta0*lamb[i]+1j*np.sqrt(np.max(c2))*2.4*lamb[i],-beta0*lamb[i]-1j*np.sqrt(np.max(c2))*2.4*lamb[i],-1j*np.sqrt(np.max(c2))*2.4*lamb[i],1j*np.sqrt(np.max(c2))*2.4*lamb[i]]),1)
            #     gamma,c,d,a_e=ellipse_properties(np.array([-beta0*lamb[i],-1j*np.sqrt(np.max(c2))*2.4*lamb[i],1j*np.sqrt(np.max(c2))*2.4*lamb[i]]),1)
            # elif dim==2:
            #     beta0=3.3/100
            #     # gamma,c,d,a_e=ellipse_properties(np.array([-beta0*lamb[i],-beta0*lamb[i]+1j*np.sqrt(np.max(c2))*3.3*lamb[i],-beta0*lamb[i]-1j*np.sqrt(np.max(c2))*3.3*lamb[i],-1j*np.sqrt(np.max(c2))*3.3*lamb[i],1j*np.sqrt(np.max(c2))*3.3*lamb[i]]),1)
            #     gamma,c,d,a_e=ellipse_properties(np.array([-beta0*lamb[i],-1j*np.sqrt(np.max(c2))*3.3*lamb[i],1j*np.sqrt(np.max(c2))*3.3*lamb[i]]),1)
            gamma,c,d,a_e=ellipse_properties(spectral_dist_stability(equ,dim,spatial_discr,param)*lamb[i],1)
            coefficients_faber=np.array(Faber_approx_coeff(degree[-1]+1,gamma,c,d).tolist(),dtype=np.float_)
            if coefficients_faber[-1]==0:
                ind_faber=1
        # rk2[i]=radio_rk2(theta,phi,lamb[i],spatial_discr,dim,equ,param)
        # rk4[i]=radio_rk4(theta,phi,lamb[i],spatial_discr,dim,equ,param)
        # rk7[i]=radio_rk7(theta,phi,lamb[i],spatial_discr,dim,equ,param)
        # time_2step[i]=radio_time_2step(theta,phi,lamb[i],spatial_discr,dim,equ,param)
        for j in range(len(degree)):
            # rkn[i,j]=radio_rkn(theta,phi,lamb[i],degree[j],spatial_discr,dim,equ,param)
            if ind_faber==0:
                faber[i,j]=radio_faber(theta,phi,lamb[i],degree[j],gamma,c,d,coefficients_faber,spatial_discr,dim,equ,param)
            # mtime_steps[i,j]=radio_mtime_steps(c2,lamb[i],degree[j],spatial_discr)

    # np.save('Stability/rk2_dim_'+str(dim)+'_'+str(spatial_discr),rk2)
    # np.save('Stability/rk4_dim_'+str(dim)+'_'+str(spatial_discr),rk4)
    # np.save('Stability/rk7_dim_'+str(dim)+'_'+str(spatial_discr),rk7)
    # np.save('Stability/time_2step_dim_'+str(dim)+'_'+str(spatial_discr),time_2step)
    # np.save('Stability/rkn_dim_'+str(dim)+'_'+str(spatial_discr),rkn)
    np.save('Stability/faber_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr),faber)
    # np.save('Stability/mtime_steps_'+str(spatial_discr),mtime_steps)


def graph(degree,spatial_discr,dim,equ):

    lamb=np.load('Stability/lamb_'+str(equ)+'.npy')
    # rk2=np.load('Stability/rk2_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    # rk4=np.load('Stability/rk4_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    # rk7=np.load('Stability/rk7_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    # time_2step=np.load('Stability/time_2step_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    #
    lim_sup=1+pow(10,-5)
    tol=1+pow(10,-5)
    # rk2[rk2>lim_sup]=lim_sup
    # rk4[rk4>lim_sup]=lim_sup
    # rk7[rk7>lim_sup]=lim_sup
    # time_2step[time_2step>lim_sup]=lim_sup
    #
    # print('rk2: ',np.max(lamb[rk2<tol]))
    # print('rk4: ',np.max(lamb[rk4<tol]))
    # print('rk7: ',np.max(lamb[rk7<tol]))
    # print('time_2step: ',np.max(lamb[time_2step<tol]))
    #
    # plt.plot(lamb,rk2,label='RK3-2')
    # plt.plot(lamb,rk4,label='RK4-4')
    # plt.plot(lamb,rk7,label='RK9-7')
    # plt.plot(lamb,time_2step,label='2MS')
    # plt.xlabel('$\lambda$',fontsize=20)
    # plt.ylabel(r'$\rho$',fontsize=20)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(prop={'size': 15})
    # plt.savefig('Stability/low_order_dim_'+str(dim)+'_'+str(spatial_discr)+'.pdf')
    # plt.show()
    #
    # rkn=np.load('Stability/rkn_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    # rkn=rkn[:,degree-3]
    # rkn[rkn>lim_sup]=lim_sup
    # for i in range(len(degree)):
    #     if rkn[0,i]<lim_sup:
    #         print('rkn_'+str(degree[i])+' ',np.max(lamb[rkn[:,i]<tol]))
    #     else:
    #         print('rkn_'+str(degree[i])+' 0')
    #     if np.max(lamb[rkn[:,i]<lim_sup])<4:
    #         order=degree[i]-2
    #         if degree[i]==2:
    #             order=1
    #     plt.plot(lamb,rkn[:,i],label='RKHO'+str(degree[i])+'-'+str(order))
    # plt.xlabel('$\lambda$',fontsize=20)
    # plt.ylabel(r'$\rho$',fontsize=20)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(prop={'size': 15})
    # plt.savefig('Stability/rkn_dim_'+str(dim)+'_'+str(spatial_discr)+'.pdf')
    # plt.show()

    faber=np.load('Stability/faber_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    faber=faber[:,degree-3]
    # faber[faber>lim_sup]=lim_sup
    faber[faber==0]=lim_sup
    # faber[0,:]=pow(10,-5)
    for i in range(len(degree)):
        if faber[0,i]<lim_sup:
            print('faber_'+str(degree[i])+' ',np.max(lamb[faber[:,i]<tol]))
        else:
            print('faber_'+str(degree[i])+' 0')
        plt.plot(lamb,faber[:,i],label='FA'+str(degree[i]),linewidth=2)
    plt.xlabel('$\lambda$',fontsize=20)
    plt.ylabel(r'$\rho$',fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(prop={'size': 15})
    plt.ylim([1-2*pow(10,-10), 1+1.02*pow(10,-5)])
    plt.xlim([-0.01, 1])
    plt.subplots_adjust(left=0.23, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('Stability/faber_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.pdf')
    plt.show()

    # mtime_steps=np.load('Stability/mtime_steps_'+str(spatial_discr)+'.npy')
    # mtime_steps[mtime_steps>lim_sup]=lim_sup
    # for i in range(len(degree)):
    #     plt.plot(lamb,mtime_steps[:,i],label='mtime_steps'+str(degree[i]))
    # plt.legend()
    # plt.savefig('Stability/mtime_steps_'+str(spatial_discr)+'.pdf')
    # plt.show()


def graph_experiments(spatial_discr,dim,equ,ind):
    # This function is to draw stability graphics for high order methods under several discretization parameters

    tol=1+pow(10,-7)

    # rkn=np.load('Stability/rkn_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    # rkn=rkn[:,degree-3]
    # rkn[rkn>lim_sup]=lim_sup
    # for i in range(len(degree)):
    #     if rkn[0,i]<lim_sup:
    #         print('rkn_'+str(degree[i])+' ',np.max(lamb[rkn[:,i]<tol]))
    #     else:
    #         print('rkn_'+str(degree[i])+' 0')
    #     if np.max(lamb[rkn[:,i]<lim_sup])<4:
    #         order=degree[i]-2
    #         if degree[i]==2:
    #             order=1
    #     plt.plot(lamb,rkn[:,i],label='RKHO'+str(degree[i])+'-'+str(order))
    # plt.xlabel('$\lambda$',fontsize=20)
    # plt.ylabel(r'$\rho$',fontsize=20)
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(prop={'size': 15})
    # plt.savefig('Stability/rkn_dim_'+str(dim)+'_'+str(spatial_discr)+'.pdf')
    # plt.show()

    ax=plt.gca()
    graph_marker=np.array(['D','o','s','v'])
    graph_type=np.array(['-','--','-.',':'])
    for i in range(len(equ)):
        lamb=np.load('Stability/lamb_'+str(equ[i])+'.npy')
        for j in range(len(dim)):
            for k in range(len(spatial_discr)):
                faber=np.load('Stability/faber_dim_'+dim[j]+'_equ_'+equ[i]+'_ord_'+spatial_discr[k]+'.npy')
                faber=faber[:,:40-3]
                faber[faber==0]=tol
                max_lamb=np.zeros(faber.shape[1])
                for l in range(faber.shape[1]):
                    if faber[0,l]<tol:
                        max_lamb[l]=np.max(lamb[faber[:,l]<tol])
                label_str=dim[j]+'D_'
                if equ[i]=='elastic':
                    label_str=label_str+'Elastic_'
                elif equ[i]=='scalar':
                    label_str=label_str+'1SD_'
                else:
                    label_str=label_str+'2SD_'
                label_str=label_str+'ord'+spatial_discr[k]
                pos=i*len(dim)*len(spatial_discr)+j*len(spatial_discr)+k
                # plt.plot(np.arange(3,3+faber.shape[1]),max_lamb,label=label_str,linewidth=2,marker=graph_marker[pos],linestyle=graph_type[pos])
                lin,=ax.plot(np.arange(3,3+faber.shape[1]),max_lamb,linewidth=2,linestyle=graph_type[pos],alpha=0.9)
                ax.scatter(np.arange(3,3+faber.shape[1]),max_lamb, linewidth=0.5,marker=graph_marker[pos],alpha=0.5)
                ax.scatter(np.arange(3,3+faber.shape[1]),max_lamb,marker=graph_marker[pos],color=lin.get_color(),s=5)
                ax.plot([],[],label=label_str,color=lin.get_color(),linewidth=2,marker=graph_marker[pos],linestyle=graph_type[pos])
    plt.xlabel('Polynomial degree',fontsize=22)
    plt.ylabel('$c_{\mathrm{CFL}}$',fontsize=23)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={'size': 18})
    # plt.ylim([1-2*pow(10,-10), 1+1.02*pow(10,-5)])
    # plt.xlim([-0.01, 1])
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('Stability/faber_experiments_'+ind+'.pdf')
    plt.show()

    # mtime_steps=np.load('Stability/mtime_steps_'+str(spatial_discr)+'.npy')
    # mtime_steps[mtime_steps>lim_sup]=lim_sup
    # for i in range(len(degree)):
    #     plt.plot(lamb,mtime_steps[:,i],label='mtime_steps'+str(degree[i]))
    # plt.legend()
    # plt.savefig('Stability/mtime_steps_'+str(spatial_discr)+'.pdf')
    # plt.show()


def eff_graph(degree,spatial_discr,dim,equ,ind):

    # lamb=np.load('Stability/lamb_'+str(equ)+'.npy')
    # rk2=np.load('Stability/rk2_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    # rk4=np.load('Stability/rk4_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    # rk7=np.load('Stability/rk7_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    # time_2step=np.load('Stability/time_2step_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    #
    # eff_low_order=np.zeros(4)
    # eff_rkn=np.zeros(len(degree))
    tol=1+pow(10,-7)
    #
    # ind_time_2step=0
    # ind_rk2=0
    # ind_rk4=0
    # ind_rk7=0
    # for i in range(len(lamb)):
    #     if ind_time_2step==0 and time_2step[i]>tol:
    #         ind_time_2step=1
    #         eff_low_order[0]=lamb[i-1]
    #     if ind_rk2==0 and rk2[i]>tol:
    #         ind_rk2=1
    #         eff_low_order[1]=lamb[i-1]/3
    #     if ind_rk4==0 and rk4[i]>tol:
    #         ind_rk4=1
    #         eff_low_order[2]=lamb[i-1]/4
    #     if ind_rk7==0 and rk7[i]>tol:
    #         ind_rk7=1
    #         eff_low_order[3]=lamb[i-1]/9
    #     if ind_time_2step+ind_rk2+ind_rk4+ind_rk7==4:
    #         break

    # rkn=np.load('Stability/rkn_dim_'+str(dim)+'_'+str(spatial_discr)+'.npy')
    # for i in range(len(degree)):
    #     for j in range(len(lamb)):
    #         if rkn[j,i]>tol:
    #             eff_rkn[i]=lamb[j-1]/degree[i]
    #             break
    #         if j==len(lamb)-1:
    #             eff_rkn[i]=lamb[j]/degree[i]

    ax=plt.gca()
    graph_marker=np.array(['D','o','s','v'])
    graph_type=np.array(['-','--','-.',':'])
    for i in range(len(equ)):
        lamb=np.load('Stability/lamb_'+str(equ[i])+'.npy')
        for j in range(len(dim)):
            for k in range(len(spatial_discr)):
                faber=np.load('Stability/faber_dim_'+dim[j]+'_equ_'+equ[i]+'_ord_'+spatial_discr[k]+'.npy')
                faber=faber[:,:40-3]
                faber[faber==0]=tol
                eff_faber=np.zeros(faber.shape[1])
                for l in range(faber.shape[1]):
                    if faber[0,l]<tol:
                        eff_faber[l]=np.max(lamb[faber[:,l]<tol])/(l+3)
                label_str=dim[j]+'D_'
                if equ[i]=='elastic':
                    label_str=label_str+'Elastic_'
                elif equ[i]=='scalar':
                    label_str=label_str+'1SD_'
                else:
                    label_str=label_str+'2SD_'
                label_str=label_str+'ord'+spatial_discr[k]
                pos=i*len(dim)*len(spatial_discr)+j*len(spatial_discr)+k

                lin,=ax.plot(np.arange(3,3+faber.shape[1]),1/eff_faber,linewidth=2,linestyle=graph_type[pos],alpha=0.9)
                ax.scatter(np.arange(3,3+faber.shape[1]),1/eff_faber, linewidth=2,marker=graph_marker[pos],alpha=0.5)
                ax.scatter(np.arange(3,3+faber.shape[1]),1/eff_faber,marker=graph_marker[pos],color=lin.get_color(),s=5)
                ax.plot([],[],label=label_str,color=lin.get_color(),linewidth=2,marker=graph_marker[pos],linestyle=graph_type[pos])

    plt.xlabel('Polynomial degree',fontsize=22)
    plt.ylabel(r'$\mathrm{N}^{\mathrm{CFL}}_{\mathrm{op}}$',fontsize=23)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={'size': 18})
    # plt.ylim([1-2*pow(10,-10), 1+1.02*pow(10,-5)])
    # plt.xlim([-0.01, 1])
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.9)
    plt.ylim(0,1000)
    plt.savefig('Stability/stability_eff_'+ind+'.pdf')
    plt.show()

    # # plt.scatter(1,eff_low_order[0],label='2MS',color='b')
    # # plt.scatter(3,eff_low_order[1],label='RK3-2',color='g')
    # # plt.scatter(4,eff_low_order[2],label='RK4-4',color='purple')
    # # plt.scatter(9,eff_low_order[3],label='RK9-7',color='cyan')
    # # plt.plot(degree,eff_rkn,label='RKHO',color='lawngreen',linewidth=2)
    # # plt.plot(degree,eff_faber,label='FA',color='palevioletred',linewidth=2)
    # plt.plot(degree,eff_faber,label='FA',linewidth=2,marker='D')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # # plt.legend(prop={'size': 15})
    # plt.ylabel('$E_{ff}$',fontsize=20)
    # # plt.xlabel('# MVO',fontsize=20)
    # plt.xlabel('Polynomial degree',fontsize=20)
    # plt.subplots_adjust(left=0.17, bottom=0.15, right=0.9, top=0.9)
    # plt.savefig('Stability/stability_eff_dim'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.pdf')
    # plt.show()


mp.mp.dps=30

c2=1.524**2
# lamb=1/(8*np.sqrt(c2))
# beta0=3.3/100
# beta0=2.4/100
# degree=np.array([3,4,9,15,20,25,30,35,40])
degree=np.arange(3,41)
# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='4',dim=1,equ='scalar',param=c2)
# stability(N_lamb=350,lamb_max=7,degree=degree,c2=c2,spatial_discr='inf',dim=1)
# stability(N_lamb=350,lamb_max=7,degree=degree,c2=c2,spatial_discr='4',dim=2)
# stability(N_lamb=350,lamb_max=7,degree=degree,c2=c2,spatial_discr='inf',dim=2)
# graph(degree=degree,spatial_discr="4",dim=2)
# eff_graph(degree=degree,spatial_discr='4',dim=1)
# ""
# graph(10,100,50,100,20)
# graph(10,100,50,100,20)

# ord=sys.argv[1]
# dim=int(sys.argv[2])
# equ=sys.argv[3]
# if equ=='elastic':
#     c2=np.array([1,2.5,18])
#
# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr=ord,dim=dim,equ=equ,param=c2)

# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='4',dim=1,equ='scalar',param=c2)
# stability(N_lamb=150,lamb_max=7,degree=degree,spatial_discr='4',dim=1,equ='scalar_dx2',param=c2)
# stability(N_lamb=150,lamb_max=7,degree=degree,spatial_discr='4',dim=2,equ='scalar',param=c2)
# stability(N_lamb=150,lamb_max=7,degree=degree,spatial_discr='4',dim=2,equ='scalar_dx2',param=c2)

# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='8',dim=1,equ='scalar',param=c2)
# stability(N_lamb=150,lamb_max=7,degree=degree,spatial_discr='8',dim=1,equ='scalar_dx2',param=c2)
# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='8',dim=2,equ='scalar',param=c2)
# stability(N_lamb=150,lamb_max=7,degree=degree,spatial_discr='8',dim=2,equ='scalar_dx2',param=c2)
# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='4',dim=2,equ='elastic',param=np.array([1,2.5,18]))
# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='8',dim=2,equ='elastic',param=np.array([1,2.5,18]))
# degree=np.arange(5,41,5)
# graph(degree=degree,spatial_discr="4",dim=1,equ='scalar')
# graph(degree=degree,spatial_discr="4",dim=1,equ='scalar_dx2')
# graph(degree=degree,spatial_discr="4",dim=2,equ='scalar')
# graph(degree=degree,spatial_discr="4",dim=2,equ='scalar_dx2')
# graph(degree=degree,spatial_discr="8",dim=1,equ='scalar')
# graph(degree=degree,spatial_discr="8",dim=1,equ='scalar_dx2')
# graph(degree=degree,spatial_discr="8",dim=2,equ='scalar')
# graph(degree=degree,spatial_discr="8",dim=2,equ='scalar_dx2')
# graph(degree=degree,spatial_discr="4",dim=2,equ='elastic')
# graph(degree=degree,spatial_discr="8",dim=2,equ='elastic')


# degree=np.arange(3,41)
# eff_graph(degree=degree,spatial_discr='4',dim=1,equ='scalar')
# eff_graph(degree=degree,spatial_discr='4',dim=1,equ='scalar_dx2')
# eff_graph(degree=degree,spatial_discr='4',dim=2,equ='scalar')
# eff_graph(degree=degree,spatial_discr='4',dim=2,equ='scalar_dx2')
# eff_graph(degree=degree,spatial_discr='8',dim=1,equ='scalar')
# eff_graph(degree=degree,spatial_discr='8',dim=1,equ='scalar_dx2')
# eff_graph(degree=degree,spatial_discr='8',dim=2,equ='scalar')
# eff_graph(degree=degree,spatial_discr='8',dim=2,equ='scalar_dx2')
# eff_graph(degree=degree,spatial_discr='4',dim=2,equ='elastic')
# eff_graph(degree=degree,spatial_discr='8',dim=2,equ='elastic')

# graph_experiments(spatial_discr=np.array(["4",'8']),dim=np.array(['1']),equ=np.array(['scalar','scalar_dx2']),ind='1D')
# graph_experiments(spatial_discr=np.array(["4",'8']),dim=np.array(['2']),equ=np.array(['scalar','scalar_dx2']),ind='2D')
# graph_experiments(spatial_discr=np.array(["4",'8']),dim=np.array(['2']),equ=np.array(['elastic']),ind='elastic')

# eff_graph(degree=0,spatial_discr=np.array(["4",'8']),dim=np.array(['1']),equ=np.array(['scalar','scalar_dx2']),ind='1D')
# eff_graph(degree=0,spatial_discr=np.array(["4",'8']),dim=np.array(['2']),equ=np.array(['scalar','scalar_dx2']),ind='2D')
eff_graph(degree=0,spatial_discr=np.array(["4",'8']),dim=np.array(['2']),equ=np.array(['elastic']),ind='elastic')