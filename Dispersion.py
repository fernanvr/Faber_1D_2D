import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy import optimize
from auxiliary_functions import *
from numpy.linalg import matrix_power as m_pow
import mpmath as mp


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
                    return lamb/24*np.array([[0,0, 1/(2*param[1]+param[2])*aux1_x,1/(2*param[1]+param[2])*aux1_y,0],
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


def radio(A,lamb,theta_x,theta_y,equ,param):

    A=np.array(A.tolist(),dtype=np.complex_)
    if equ=='elastic':
        c=(2*param[1]+param[2])/param[0]
    else:
        c=param
    # print(A)

    # d=np.sqrt(pow(A[0,0]+A[1,1],2)-4*(A[0,0]*A[1,1]-A[0,1]*A[1,0]))
    #
    # angle=np.angle((A[0,0]+A[1,1]+d)/2)
    # if angle<0:
    #     angle=2*np.pi+angle
    # c=angle/(lamb*theta)
    #
    # result1=c/np.sqrt(c2)
    #
    # # print((A[0,0]+A[1,1]-d)/2)
    # angle=np.angle((A[0,0]+A[1,1]-d)/2)
    # if angle<0:
    #     angle=2*np.pi+angle
    # c=angle/(lamb*theta)
    #
    # result2=c/np.sqrt(c2)
    #
    # if np.abs(result1-1)>np.abs(result2-1):
    #     return result2
    # else:
    #     return result1
    angle=np.angle(np.linalg.eigvals(A))
    # for i in range(len(angle)):
    #     if angle[i]<0:
    #         angle[i]=2*np.pi+angle[i]
    # print(angle)
    result=angle/(lamb*np.sqrt(pow(theta_x,2)+pow(theta_y,2)))

    return np.abs(result[np.argmin(np.abs(result-1))])


def radio_rk2(lamb,theta_x,theta_y,spatial_discr,dim,equ,param):

    H=app_H(lamb,theta_x,theta_y,spatial_discr,dim,equ,param)
    rk2=np.identity(len(H))+H+m_pow(H,2)/2+m_pow(H,3)/4

    # print(H)

    return radio(rk2,lamb,theta_x,theta_y,equ,param)


def radio_rk4(lamb,theta_x,theta_y,spatial_discr,dim,equ,param):

    H=app_H(lamb,theta_x,theta_y,spatial_discr,dim,equ,param)

    rk4=np.identity(len(H))+H+m_pow(H,2)/2+m_pow(H,3)/6+m_pow(H,4)/24

    # print(H)
    return radio(rk4,lamb,theta_x,theta_y,equ,param)


def radio_rk7(lamb,theta_x,theta_y,spatial_discr,dim,equ,param):

    H=app_H(lamb,theta_x,theta_y,spatial_discr,dim,equ,param)

    k1=H
    k2=H.dot(np.identity(len(H))+4/63*k1)
    k3=H.dot(np.identity(len(H))+(1/42*k1+1/14*k2))
    k4=H.dot(np.identity(len(H))+(1/28*k1+3/28*k3))
    k5=H.dot(np.identity(len(H))+(12551/19652*k1-48363/19652*k3+10976/4913*k4))
    k6=H.dot(np.identity(len(H))+(-36616931/27869184*k1+2370277/442368*k3-255519173/63700992*k4+226798819/445906944*k5))
    k7=H.dot(np.identity(len(H))+(-10401401/7164612*k1+47383/8748*k3-4914455/1318761*k4-1498465/7302393*k5+2785280/3739203*k6))
    k8=H.dot(np.identity(len(H))+(181002080831/17500000000*k1-14827049601/400000000*k3+23296401527134463/857600000000000*k4+2937811552328081/949760000000000*k5-243874470411/69355468750*k6+2857867601589/3200000000000*k7))
    k9=H.dot(np.identity(len(H))+(-228380759/19257212*k1+4828803/113948*k3-331062132205/10932626912*k4-12727101935/3720174304*k5+22627205314560/4940625496417*k6-268403949/461033608*k7+3600000000000/19176750553961*k8))

    rk7=np.identity(len(H))+(95/2366*k1+3822231133/16579123200*k4+555164087/2298419200*k5+1279328256/9538891505*k6+5963949/25894400*k7+50000000000/599799373173*k8+28487/712800*k9)

    return radio(rk7,lamb,theta_x,theta_y,equ,param)


def radio_time_2step(lamb,theta_x,theta_y,spatial_discr,dim,param):

    if dim==1:
        if spatial_discr=='4':
            if np.abs(param*pow(lamb,2)*(-np.cos(2*theta_x)/6+np.cos(theta_x)*8/3-5/2)/2+1)<1:
                c=np.arccos(param*pow(lamb,2)*(-np.cos(2*theta_x)/6+np.cos(theta_x)*8/3-5/2)/2+1)/(lamb*theta_x)
            else:
                c=0
        elif spatial_discr=='inf':
            c=np.arccos(-param*pow(lamb*theta_x,2)/2+1)/(lamb*theta_x)
        return c/param
    elif dim==2:
        if spatial_discr=='4':
            if np.abs(param*pow(lamb,2)*(-np.cos(2*theta_x)/6+np.cos(theta_x)*8/3-5/2-np.cos(2*theta_y)/6+np.cos(theta_y)*8/3-5/2)/2+1)<1:
                c=np.arccos(param*pow(lamb,2)*(-np.cos(2*theta_x)/6+np.cos(theta_x)*8/3-5/2-np.cos(2*theta_y)/6+np.cos(theta_y)*8/3-5/2)/2+1)/(lamb*np.sqrt(pow(theta_x,2)+pow(theta_y,2)))
            else:
                c=0
        elif spatial_discr=='inf':
            c=np.arccos(-param*(pow(lamb*theta_x,2)+pow(lamb*theta_y,2))/2+1)/(lamb*np.sqrt(pow(theta_x,2)+pow(theta_y,2)))
        return c/param


def radio_rkn(lamb,theta_x,theta_y,grau,spatial_discr,dim,equ,param):

    H=app_H(lamb,theta_x,theta_y,spatial_discr,dim,equ,param)

    mu=1
    alpha=ssprk_alpha(mu,grau)


    rkn = alpha[0]*np.identity(len(H))
    for j in range(1,grau-1):
        rkn=rkn+m_pow(np.identity(len(H))+mu*H,j)*alpha[j]

    rkn=rkn+m_pow(np.identity(len(H))+mu*H,grau)*alpha[grau-1]
    return radio(rkn,lamb,theta_x,theta_y,equ,param)


def radio_faber(lamb,theta_x,theta_y,grau,gamma,c,d,coeff,spatial_discr,dim,equ,param):

    c0=d/gamma*mp.exp(0)
    c1=c**2/(4*gamma**2)*mp.exp(0)

    H=app_H(lamb,theta_x,theta_y,spatial_discr,dim,equ,param)*mp.exp(0)

    result=coeff[0]*np.identity(len(H))

    F1=H/gamma-c0*np.identity(len(H))
    F1_fix=F1

    result=result+F1*coeff[1]

    F2=F1_fix.dot(F1)-2*c1*np.identity(len(H))
    result=result+F2*coeff[2]

    for i in range(3,grau):
        F0=F1
        F1=F2
        F2=F1_fix.dot(F1)-c1*F0
        result=result+F2*coeff[i]

    return radio(result,lamb,theta_x,theta_y,equ,param)


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


def dispersion(N_theta,degree,c2,lamb,spatial_discr,dim,label,equ,param):

    theta=np.linspace(pow(10,-3),np.pi,N_theta)
    phi=np.pi/6
    if dim==1:
        phi=0
    theta_x=theta*np.cos(phi)
    theta_y=theta*np.sin(phi)
    np.save('Dispersion/theta',theta)

    rk2=theta*0
    rk4=theta*0
    rk7=theta*0
    time_2step=theta*0
    rkn=np.zeros((N_theta,len(degree)))
    faber=np.zeros((N_theta,len(degree)))

    # gamma,c,d,a_e=ellipse_properties(spectral_dist_stability(equ,dim,spatial_discr,param)*lamb[5],1)
    # coefficients_faber=np.array(Faber_approx_coeff(degree[-1]+1,gamma,c,d).tolist(),dtype=np.float_)

    for i in range(N_theta):
        # print('i: ',i)
        # print('rk2: ')
        rk2[i]=radio_rk2(lamb[0],theta_x[i],theta_y[i],spatial_discr,dim,equ,param)
        # print('rk4: ')
        rk4[i]=radio_rk4(lamb[1],theta_x[i],theta_y[i],spatial_discr,dim,equ,param)
        # print('rk7: ')
        rk7[i]=radio_rk7(lamb[2],theta_x[i],theta_y[i],spatial_discr,dim,equ,param)
        # print('time_2step: ')
        time_2step[i]=radio_time_2step(lamb[3],theta_x[i],theta_y[i],spatial_discr,dim)
        # print('rkn y faber: ')
        for j in range(len(degree)):
            # print('degree[j]: ',degree[j])
            rkn[i,j]=radio_rkn(lamb[4],theta_x[i],theta_y[i],degree[j],spatial_discr,dim,equ,param)
            # faber[i,j]=radio_faber(lamb[5],theta_x[i],theta_y[i],degree[j],gamma,c,d,coefficients_faber,spatial_discr,dim,equ,param)

    np.save('Dispersion/rk2_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label),rk2)
    np.save('Dispersion/rk4_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label),rk4)
    np.save('Dispersion/rk7_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label),rk7)
    np.save('Dispersion/time_2step_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label),time_2step)
    np.save('Dispersion/rkn_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label),rkn)
    # np.save('Dispersion/faber_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label),faber)


def graph(degree,dim,spatial_discr,label,equ):

    theta=np.load('Dispersion/theta.npy')/np.pi*100


    rk2=np.load('Dispersion/rk2_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label)+'.npy')
    rk4=np.load('Dispersion/rk4_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label)+'.npy')
    rk7=np.load('Dispersion/rk7_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label)+'.npy')
    time_2step=np.load('Dispersion/time_2step_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label)+'.npy')

    plt.plot(theta,np.log(rk2),label='RK3-2')
    plt.plot(theta,np.log(rk4),label='RK4-4')
    plt.plot(theta,np.log(rk7),label='RK9-7')
    plt.plot(theta,np.log(time_2step),label='2MS')
    plt.legend()
    plt.xlabel('Percentage of Nyquist frequency',size='16')
    plt.ylabel('$c/c_{num}$',size='16')
    plt.savefig('Dispersion/dispersion_low_order_dim_'+str(dim)+'_'+str(spatial_discr)+str(label)+'.pdf')
    plt.show()


    rkn=np.load('Dispersion/rkn_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+tr(spatial_discr)+str(label)+'.npy')
    for i in range(len(degree)):
        order=degree[i]-2
        if degree[i]==3 or degree[i]==4:
            order=degree[i]
        plt.plot(theta,np.log(rkn[:,i]),label='RKHO'+str(degree[i])+'-'+str(order))
    # plt.legend()
    # plt.savefig('Dispersion/dispersion_rkn_dim_'+str(dim)+'_'+str(spatial_discr)+str(label)+'.pdf')
    # plt.show()
    #
    # for i in range(len(degree)):
    #     order=degree[i]-2
    #     if degree[i]==2:
    #         order=1
    #     plt.plot(theta,np.log(np.load('Dispersion/rkn_dim_'+str(dim)+'_'+str(spatial_discr)+'_'+str(degree[i])+'.npy')[:,0]),label='RKn('+str(degree[i])+','+str(order)+')')
    plt.legend()
    plt.xlabel('Percentage of Nyquist frequency',size='16')
    plt.ylabel('$c/c_{num}$',size='16')
    plt.savefig('Dispersion/dispersion_rkn_dim_'+str(dim)+'_'+str(spatial_discr)+str(label)+'.pdf')
    plt.show()

    faber=np.load('Dispersion/faber_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label)+'.npy')
    for i in range(len(degree)):
        plt.plot(theta,np.log(faber[:,i]),label='FA'+str(degree[i]))
    # plt.legend()
    # plt.savefig('Dispersion/dispersion_faber_dim_'+str(dim)+'_'+str(spatial_discr)+str(label)+'.pdf')
    # plt.show()
    #
    # for i in range(len(degree)):
    #     plt.plot(theta,np.log(np.load('Dispersion/faber_dim_'+str(dim)+'_'+str(spatial_discr)+'_'+str(degree[i])+'.npy')[:,0]),label='FA'+str(degree[i]))
    plt.legend()
    plt.xlabel('Percentage of Nyquist frequency',size='16')
    plt.ylabel('$c/c_{num}$',size='16')
    plt.savefig('Dispersion/dispersion_faber_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+str(label)+'.pdf')
    plt.show()


def dispersion_courant(lamb_max,degree,spatial_discr,dim,equ,param):
    # This function calculates the maximum dispersion for the range of Nyquist frequency

    lamb=np.linspace(pow(10,-3),lamb_max,50)
    np.save('Dispersion/lamb',lamb)

    theta=np.linspace(pow(10,-3),np.pi,50)
    phi=np.pi/6
    if dim==1:
        phi=0
    theta_x=theta*np.cos(phi)
    theta_y=theta*np.sin(phi)

    # rk2=lamb*0
    # rk4=lamb*0
    # rk7=lamb*0
    # time_2step=lamb*0
    # rkn=np.zeros((len(lamb),len(degree)))
    faber=np.zeros((len(lamb),len(degree)))

    ind_faber=0
    for i in range(len(lamb)):
        print('0000000000000000000000000000000000000 i:',i)
        if ind_faber==0:
            gamma,c,d,a_e=ellipse_properties(spectral_dist_stability(equ,dim,spatial_discr,param)*lamb[i],1)
            coefficients_faber=np.array(Faber_approx_coeff(degree[-1]+1,gamma,c,d).tolist(),dtype=np.float_)
            if coefficients_faber[-1]==0:
                ind_faber=1

        for j in range(len(theta)):
            # rk2[i]=np.maximum(rk2[i],radio_rk2(lamb[i],theta_x[j],theta_y[j],spatial_discr,dim,equ,param))
            # rk4[i]=np.maximum(rk4[i],radio_rk4(lamb[i],theta_x[j],theta_y[j],spatial_discr,dim,equ,param))
            # rk7[i]=np.maximum(rk7[i],radio_rk7(lamb[i],theta_x[j],theta_y[j],spatial_discr,dim,equ,param))
            # time_2step[i]=np.maximum(time_2step[i],radio_time_2step(lamb[i],theta_x[j],theta_y[j],spatial_discr,dim,param))
            for k in range(len(degree)):
                # rkn[i,k]=np.maximum(rkn[i,k],radio_rkn(lamb[i],theta_x[j],theta_y[j],degree[k],spatial_discr,dim,equ,param))
                if ind_faber==0:
                    faber[i,k]=np.maximum(faber[i,k],radio_faber(lamb[i],theta_x[j],theta_y[j],degree[k],gamma,c,d,coefficients_faber,spatial_discr,dim,equ,param))

    # np.save('Dispersion/rk2_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr),rk2)
    # np.save('Dispersion/rk4_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr),rk4)
    # np.save('Dispersion/rk7_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr),rk7)
    # np.save('Dispersion/time_2step_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr),time_2step)
    # np.save('Dispersion/rkn_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr),rkn)
    np.save('Dispersion/faber_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr),faber)


def graph_courant(degree,dim,spatial_discr,equ):

    tol=pow(10,-5)
    lamb=np.load('Dispersion/lamb.npy')

    rk2=np.load('Dispersion/rk2_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    rk4=np.load('Dispersion/rk4_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    rk7=np.load('Dispersion/rk7_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    time_2step=np.load('Dispersion/time_2step_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')

    # rk2[(rk2>10)+(np.isnan(rk2))]=10
    # rk4[(rk4>10)+(np.isnan(rk4))]=10
    # rk7[(rk7>10)+(np.isnan(rk7))]=10
    # time_2step[(time_2step>10)+(np.isnan(time_2step))]=10

    # print('rk2: ',np.max(lamb[np.abs(rk2-1)<tol]))
    # print('rk4: ',np.max(lamb[np.abs(rk4-1)<tol]))
    # print('rk7: ',np.max(lamb[np.abs(rk7-1)<tol]))
    # print('time_2step: ',np.max(lamb[np.abs(time_2step-1)<tol]))

    plt.plot(lamb,rk2,label='RK3-2')
    plt.plot(lamb,rk4,label='RK4-4')
    plt.plot(lamb,rk7,label='RK9-7')
    plt.plot(lamb,time_2step,label='2MS')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={'size': 17})
    plt.xlabel('Courant number',fontsize=22)
    plt.ylabel('$|c/c_{num}|_{max}$',fontsize=24)
    plt.ylim([1-0.1, 1+0.3])
    plt.savefig('Dispersion/dispersion_low_order_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.pdf')
    plt.show()

    rkn=np.load('Dispersion/rkn_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    # rkn[(rkn>10)+(np.isnan(rkn))]=10
    for i in range(len(degree)):
        order=degree[i]-2
        if degree[i]==3 or degree[i]==4:
            order=degree[i]
        if np.abs(rkn[0,i]-1)<tol:
            print('rkn_'+str(order)+' ',np.max(lamb[np.abs(rkn[:,i]-1)<tol]))
        else:
            print('rkn_'+str(degree[i])+' 0')
        plt.plot(lamb,rkn[:,i],label='RKHO'+str(degree[i])+'-'+str(order))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={'size': 17})
    plt.xlabel('Courant number',fontsize=22)
    plt.ylabel('$|c/c_{num}|_{max}$',fontsize=24)
    plt.ylim([1-0.1, 1+0.3])
    plt.savefig('Dispersion/(dispersion_rkn_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.pdf')
    plt.show()

    faber=np.load('Dispersion/faber_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    faber=faber[:,degree-3]
    # faber[np.isnan(faber)]=1+2*tol
    # faber[(np.abs(faber-1)>2*tol)+(np.isnan(faber))]=1+2*tol
    # faber[faber==0]=1+2*tol
    plt.figure(figsize=(10, 6))
    for i in range(len(degree)):
        if np.abs(faber[0,i]-1)<tol:
            print('faber_'+str(degree[i])+' ',np.max(lamb[np.abs(faber[:,i]-1)<tol]))
        else:
            print('faber_'+str(degree[i])+' 0')
        plt.plot(lamb,faber[:,i],label='FA'+str(degree[i]),linewidth=2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={'size': 17},loc = "upper right")
    plt.xlabel('Courant number',fontsize=22)
    plt.ylabel('$|c/c_{num}|_{max}$',fontsize=24)
    plt.ylim([1-0.1, 1+0.3])
    # plt.xlim([0-0.01, 1])
    # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('Dispersion/dispersion_faber_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.pdf')
    plt.show()


def graph_experiments(spatial_discr,dim,equ,ind):
    # This function is to draw stability graphics for high order methods under several discretization parameters

    tol=pow(10,-5)

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
    lamb=np.load('Dispersion/lamb.npy')
    graph_marker=np.array(['D','o','s','v'])
    graph_type=np.array(['-','--','-.',':'])
    for i in range(len(equ)):
        for j in range(len(dim)):
            for k in range(len(spatial_discr)):
                faber=np.load('Dispersion/faber_cou_dim_'+dim[j]+'_equ_'+equ[i]+'_ord_'+spatial_discr[k]+'.npy')
                faber=faber[:,:40-3]
                faber[faber==0]=tol
                max_lamb=np.zeros(faber.shape[1])
                for l in range(faber.shape[1]):
                    if np.abs(faber[0,l]-1)<tol:
                        max_lamb[l]=np.max(lamb[np.abs(faber[:,l]-1)<tol])
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
                lin,=ax.plot(np.arange(3,3+faber.shape[1]),max_lamb,linewidth=2,linestyle=graph_type[pos])
                ax.scatter(np.arange(3,3+faber.shape[1]),max_lamb, linewidth=2,marker=graph_marker[pos],alpha=0.5)
                ax.plot([],[],label=label_str,color=lin.get_color(),linewidth=2,marker=graph_marker[pos],linestyle=graph_type[pos])
    plt.xlabel('Polynomial degree',fontsize=22)
    plt.ylabel(r'$\alpha_{R}$',fontsize=23)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={'size': 18})
    # plt.ylim([1-2*pow(10,-10), 1+1.02*pow(10,-5)])
    # plt.xlim([-0.01, 1])
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('Dispersion_images/faber_experiments_dispersion_'+ind+'.pdf')
    plt.show()

    # mtime_steps=np.load('Stability/mtime_steps_'+str(spatial_discr)+'.npy')
    # mtime_steps[mtime_steps>lim_sup]=lim_sup
    # for i in range(len(degree)):
    #     plt.plot(lamb,mtime_steps[:,i],label='mtime_steps'+str(degree[i]))
    # plt.legend()
    # plt.savefig('Stability/mtime_steps_'+str(spatial_discr)+'.pdf')
    # plt.show()


def eff_graph(degree,spatial_discr,dim,equ,ind):

    tol=pow(10,-5)
    lamb=np.load('Dispersion/lamb.npy')
    # rk2=np.load('Dispersion/rk2_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    # rk4=np.load('Dispersion/rk4_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    # rk7=np.load('Dispersion/rk7_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    # time_2step=np.load('Dispersion/time_2step_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    # #
    # eff_low_order=np.zeros(4)
    # eff_rkn=np.zeros(len(degree))
    # eff_faber=np.zeros(len(degree))
    #
    # if np.abs(time_2step[0]-1)>tol:
    #     eff_low_order[0]=0
    # else:
    #     eff_low_order[0]=lamb[np.argmax(lamb[np.abs(time_2step-1)<tol])]
    # if np.abs(rk2[0]-1)>tol:
    #     eff_low_order[1]=0
    # else:
    #     eff_low_order[1]=lamb[np.argmax(lamb[np.abs(rk2-1)<tol])]/3
    # if np.abs(rk4[0]-1)>tol:
    #     eff_low_order[2]=0
    # else:
    #     eff_low_order[2]=lamb[np.argmax(lamb[np.abs(rk4-1)<tol])]/4
    # if np.abs(rk7[0]-1)>tol:
    #     eff_low_order[3]=0
    # else:
    #     eff_low_order[3]=lamb[np.argmax(lamb[np.abs(rk7-1)<tol])]/9
    #
    #
    # rkn=np.load('Dispersion/rkn_cou_dim_'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.npy')
    # for i in range(len(degree)):
    #     if np.abs(rkn[0,i]-1)>tol:
    #         eff_rkn[i]=0
    #     else:
    #         eff_rkn[i]=lamb[np.argmax(lamb[np.abs(rkn[:,i]-1)<tol])]/degree[i]

    ax=plt.gca()
    graph_marker=np.array(['D','o','s','v'])
    graph_type=np.array(['-','--','-.',':'])
    for i in range(len(equ)):
        for j in range(len(dim)):
            for k in range(len(spatial_discr)):
                faber=np.load('Dispersion/faber_cou_dim_'+dim[j]+'_equ_'+equ[i]+'_ord_'+spatial_discr[k]+'.npy')
                faber=faber[:,:40-3]
                faber[faber==0]=tol
                eff_faber=np.zeros(faber.shape[1])
                for l in range(faber.shape[1]):
                    if np.abs(faber[0,l]-1)<tol:
                        eff_faber[l]=np.max(lamb[np.abs(faber[:,l]-1)<tol])/(l+3)
                label_str=dim[j]+'D_'
                if equ[i]=='elastic':
                    label_str=label_str+'Elastic_'
                elif equ[i]=='scalar':
                    label_str=label_str+'1SD_'
                else:
                    label_str=label_str+'2SD_'
                label_str=label_str+'ord'+spatial_discr[k]
                pos=i*len(dim)*len(spatial_discr)+j*len(spatial_discr)+k
                # plt.plot(np.arange(3,3+faber.shape[1]),1/eff_faber,label=label_str,linewidth=2,marker=graph_marker[pos],linestyle=graph_type[pos])
                lin,=ax.plot(np.arange(3,3+faber.shape[1]),1/eff_faber,linewidth=2,linestyle=graph_type[pos])
                ax.scatter(np.arange(3,3+faber.shape[1]),1/eff_faber, linewidth=2,marker=graph_marker[pos],alpha=0.5)
                ax.plot([],[],label=label_str,color=lin.get_color(),linewidth=2,marker=graph_marker[pos],linestyle=graph_type[pos])
    plt.xlabel('Polynomial degree',fontsize=22)
    plt.ylabel(r'$N^{\alpha}_{op}$',fontsize=23)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(prop={'size': 18})
    # plt.ylim([1-2*pow(10,-10), 1+1.02*pow(10,-5)])
    # plt.xlim([-0.01, 1])
    plt.subplots_adjust(left=0.186, bottom=0.15, right=0.952, top=0.929)
    plt.ylim(pow(10, -16), 160)
    plt.savefig('Dispersion_images/dispersion_eff_'+ind+'.pdf')
    plt.show()

    # plt.scatter(1,eff_low_order[0],label='2MS',color='b',linewidth=2,marker='D',s=20)
    # plt.scatter(3,eff_low_order[1],label='RK3-2',color='g',linewidth=2,marker='D',s=60)
    # plt.scatter(4,eff_low_order[2],label='RK4-4',color='purple',linewidth=2,marker='D',s=60)
    # plt.scatter(9,eff_low_order[3],label='RK9-7',color='cyan',linewidth=2,marker='D',s=60)
    # plt.plot(degree,eff_rkn,label='RKHO',color='lawngreen',linewidth=2,marker='D')
    # plt.plot(degree,eff_faber,label='FA',color='palevioletred',linewidth=2,marker='D')
    # # plt.plot(degree,eff_faber,label='FA',linewidth=2,marker='D')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(prop={'size': 17})
    # plt.ylabel('$E_{ff}$',fontsize=20)
    # plt.xlabel('Polynomial degree',fontsize=20)
    # plt.subplots_adjust(left=0.17, bottom=0.15, right=0.9, top=0.9)
    # plt.savefig('Dispersion/dispersion_eff_dim'+str(dim)+'_equ_'+str(equ)+'_ord_'+str(spatial_discr)+'.pdf')
    # plt.show()


mp.mp.dps=30

c2=1.524
degree=np.arange(3,41)

# ord=sys.argv[1]
# dim=int(sys.argv[2])
# equ=sys.argv[3]
# if equ=='elastic':
#     c2=np.array([1,2.5,18])

# dispersion_courant(lamb_max=7,degree=degree,spatial_discr="4",dim=1,equ='scalar',param=c2)
# dispersion_courant(lamb_max=7,degree=degree,spatial_discr="4",dim=1,equ='scalar_dx2',param=c2)
# dispersion_courant(lamb_max=7,degree=degree,spatial_discr="8",dim=1,equ='scalar',param=c2)
# dispersion_courant(lamb_max=7,degree=degree,spatial_discr="8",dim=1,equ='scalar_dx2',param=c2)
#
# dispersion_courant(lamb_max=7,degree=degree,spatial_discr="4",dim=2,equ='scalar',param=c2)
# dispersion_courant(lamb_max=7,degree=degree,spatial_discr="4",dim=2,equ='scalar_dx2',param=c2)
# dispersion_courant(lamb_max=7,degree=degree,spatial_discr="8",dim=2,equ='scalar',param=c2)
# dispersion_courant(lamb_max=7,degree=degree,spatial_discr="8",dim=2,equ='scalar_dx2',param=c2)

# c2=np.array([1,2.5,18])
# param=c2
# # print(np.sqrt((2*param[1]+param[2])/param[0]))
# dispersion_courant(lamb_max=1,degree=degree,spatial_discr="4",dim=2,equ='elastic',param=c2)
# dispersion_courant(lamb_max=1,degree=degree,spatial_discr="8",dim=2,equ='elastic',param=c2)

# degree=np.arange(5,41,5)
# graph_courant(degree=degree,dim=2,spatial_discr="8",equ='elastic')
# degree=np.arange(3,41)
# eff_graph(degree=degree,spatial_discr="4",dim=2,equ='elastic')

# graph_courant(degree=degree,dim=1,spatial_discr="4",equ='scalar')
# degree=np.arange(3,41)



degree=np.arange(5,41,5)
# graph_courant(degree=degree,dim=1,spatial_discr="4",equ='scalar')
# graph_courant(degree=degree,dim=1,spatial_discr="4",equ='scalar_dx2')
# graph_courant(degree=degree,dim=1,spatial_discr="8",equ='scalar') # dfd
# graph_courant(degree=degree,dim=1,spatial_discr="8",equ='scalar_dx2')
# graph_courant(degree=degree,dim=2,spatial_discr="4",equ='scalar')
# graph_courant(degree=degree,dim=2,spatial_discr="4",equ='scalar_dx2')
# graph_courant(degree=degree,dim=2,spatial_discr="8",equ='scalar') # 15461
# graph_courant(degree=degree,dim=2,spatial_discr="8",equ='scalar_dx2')
#
# graph_courant(degree=degree,dim=2,spatial_discr="4",equ='elastic')
# graph_courant(degree=degree,dim=2,spatial_discr="8",equ='elastic')

degree=np.arange(3,41)
# eff_graph(degree=degree,spatial_discr="4",dim=1,equ='scalar')
# eff_graph(degree=degree,spatial_discr="4",dim=1,equ='scalar_dx2')
# eff_graph(degree=degree,spatial_discr="8",dim=1,equ='scalar')
# eff_graph(degree=degree,spatial_discr="8",dim=1,equ='scalar_dx2')
# eff_graph(degree=degree,spatial_discr="4",dim=2,equ='scalar')
# eff_graph(degree=degree,spatial_discr="4",dim=2,equ='scalar_dx2')
# eff_graph(degree=degree,spatial_discr="8",dim=2,equ='scalar')
# eff_graph(degree=degree,spatial_discr="8",dim=2,equ='scalar_dx2')
#
# eff_graph(degree=degree,spatial_discr="4",dim=2,equ='elastic')
# eff_graph(degree=degree,spatial_discr="8",dim=2,equ='elastic')
#

# degree=np.arange(3,41)
# eff_graph(degree=degree,spatial_discr="4",dim=1,equ='scalar')

# # mp.mp.dps=30
# #
# # c2=1.524**2
# # lamb=1/(8*np.sqrt(c2))+np.zeros(6)
# # lamb=0.013/0.01+np.zeros(6)
# lamb=0.04/1.542+np.zeros(6)
# # lamb=np.array([0.04,1.2, 1.95,1.6*pow(10,-5),0.72,0.02])/1.524
# # const=np.array([[0.18,0.62],[1.44,0.22],[0.92,0.44],[1.63,0.88],[2.35,1.36],[3.09,1.89],[3.85,2.43],[4.62,2.99],[5.38,3.57]])/1.524
# degree=np.array([3,4,9,10,15,20,25,30,35,40])
# dispersion(N_theta=100,degree=degree,c2=c2,lamb=lamb,spatial_discr='inf',dim=1,label='_stability')
# # dispersion(N_theta=100,degree=np.array([degree[0]]),c2=c2,lamb=lamb,beta0=beta0,spatial_discr='inf',dim=2,label='_2')
# # for i in range(len(degree)-1):
# #     lamb[4:]=const[i,:]
# #     dispersion(N_theta=100,degree=np.array([degree[i+1]]),c2=c2,lamb=lamb,beta0=3.3*np.sqrt(c2)*lamb[5]/100,spatial_discr='inf',dim=2,label='_'+str(degree[i+1]))
# graph(degree=degree,dim=1,spatial_discr='inf',label='_stability')
# # ""
# # graph(10,100,50,100,20)
# # graph(10,100,50,100,20)


# 2D
# faber=np.array([0.0004, 0.0016, 0.002, 0.0044, 0.006, 0.0076, 0.0092, 0.0108])
# rkn=-np.array([0.0004, 0.0024, 0.0024, 0.0048, 0.0068, 0.0084, 0.0104, 0.012, 0.014, 0.014])
# low_order=np.array([0.0004, 0.0024, 0.0044])
#
# xlabel=np.array(['RK2','RK4','RK7','RKn2','RKn4','RKn7','RKn10','RKn15','RKn20','RKn25','RKn30','RKn35','RKn40','Faber2','Faber4','Faber7','Faber10','Faber15','Faber20',
#                  'Faber25','Faber30'])
# grau=np.array([2,4,7,2,4,7,10,15,20,25,30,35,40,2,4,7,10,15,20,25,30])
# dt=np.array([0.0004, 0.0024, 0.0044,0.0004, 0.0024, 0.0024, 0.0048, 0.0068, 0.0084, 0.0104, 0.012, 0.014, 0.014,0.0004, 0.0016, 0.002, 0.0044, 0.006, 0.0076, 0.0092, 0.0108])
# plt.ylabel('$\Delta t/$# matrix-vector',size=16)
# plt.scatter(xlabel,dt/grau,alpha=0.9)
# plt.xticks(rotation=90,size=14)
# plt.show()
#
# # stability
# faber=np.array([0.02, 0.62, 0.22, 0.44, 0.88, 1.36, 1.89, 2.43, 2.99, 3.57])
# rkn=np.array([0.72, 0.18, 1.44, 0.92, 1.63, 2.35, 3.09, 3.85, 4.62, 5.38])
# low_order=np.array([0.04,1.6*pow(10,-5), 1.2, 1.95])

# xlabel=np.array(['RK2','2steps','RK4','RK7','RKn2','RKn4','RKn7','RKn10','RKn15','RKn20','RKn25','RKn30','RKn35','RKn40','Faber2','Faber4','Faber7','Faber10','Faber15','Faber20',
#                  'Faber25','Faber30','Faber35','Faber40'])
# grau=np.array([2,1,4,7,2,4,7,10,15,20,25,30,35,40,2,4,7,10,15,20,25,30,35,40])
# dt=np.array([0.04,1.6*pow(10,-5), 1.2, 1.95,0.72, 0.18, 1.44, 0.92, 1.63, 2.35, 3.09, 3.85, 4.62, 5.38,0.02, 0.62, 0.22, 0.44, 0.88, 1.36, 1.89, 2.43, 2.99, 3.57])
# plt.ylabel('$\Delta t/$# matrix-vector',size=16)
# plt.scatter(xlabel,dt/grau,alpha=0.9)
# plt.xticks(rotation=90,size=14)
# plt.show()
#
#
# # 1D
# faber=np.array([0.0066, 0.033, 0.079, 0.12, 0.16, 0.19, 0.22, 0.26])
# rkn=np.array([0.013, 0.059, 0.066, 0.12, 0.16, 0.2, 0.24])
# low_order=np.array([ 0.013, 0.0066, 0.059, 0.1])
#
# xlabel=np.array(['RK2','2steps','RK4','RK7','RKn2','RKn4','RKn7','RKn10','RKn15','RKn20','RKn25','Faber4','Faber7','Faber10','Faber15','Faber20',
#                  'Faber25','Faber30','Faber35'])
# grau=np.array([2,1,4,7,2,4,7,10,15,20,25,4,7,10,15,20,25,30,35])
# dt=np.array([0.013, 0.0066, 0.059, 0.1,0.013, 0.059, 0.066, 0.12, 0.16, 0.2, 0.24,0.0066, 0.033, 0.079, 0.12, 0.16, 0.19, 0.22, 0.26])
# plt.ylabel('$\Delta t/$# matrix-vector',size=16)
# plt.scatter(xlabel,dt/grau,alpha=0.9)
# plt.xticks(rotation=90,size=14)
# plt.show()
