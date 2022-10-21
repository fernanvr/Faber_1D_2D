import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from scipy import optimize
from auxiliar_functions import *
from numpy.linalg import matrix_power as m_pow
import mpmath as mp


def fun_rk2(x,y):
    z=x+1j*y
    return 1+z+pow(z,2)/2+pow(z,3)/4


def fun_2sm(x,y):
    z=x+1j*y
    return np.maximum(np.abs((2+z**2-z*np.sqrt(z**2+4))/2),np.abs((2+z**2+z*np.sqrt(z**2+4))/2))


def fun_rk4(x,y):
    z=x+1j*y
    return 1+z+pow(z,2)/2+pow(z,3)/6+pow(z,4)/24


def fun_rk7(x,y):
    z=x+1j*y

    k1=z
    k2=z*(1+4/63*k1)
    k3=z*(1+1/42*k1+1/14*k2)
    k4=z*(1+1/28*k1+3/28*k3)
    k5=z*(1+12551/19652*k1-48363/19652*k3+10976/4913*k4)
    k6=z*(1-36616931/27869184*k1+2370277/442368*k3-255519173/63700992*k4+226798819/445906944*k5)
    k7=z*(1-10401401/7164612*k1+47383/8748*k3-4914455/1318761*k4-1498465/7302393*k5+2785280/3739203*k6)
    k8=z*(1+181002080831/17500000000*k1-14827049601/400000000*k3+23296401527134463/857600000000000*k4+2937811552328081/949760000000000*k5-243874470411/69355468750*k6+2857867601589/3200000000000*k7)
    k9=z*(1-228380759/19257212*k1+4828803/113948*k3-331062132205/10932626912*k4-12727101935/3720174304*k5+22627205314560/4940625496417*k6-268403949/461033608*k7+3600000000000/19176750553961*k8)
    return 1+95/2366*k1+3822231133/16579123200*k4+555164087/2298419200*k5+1279328256/9538891505*k6+5963949/25894400*k7+50000000000/599799373173*k8+28487/712800*k9


def radio_time_2step(theta,phi,c2,lamb,spatial_discr,dim):
    z=x+1j*y
    result=0
    for i in range(len(theta)):
        for j in range(len(phi)):
            if spatial_discr=='4':
                if dim==1:
                    time_2step=np.array([[2+pow(lamb,2)*(mp.cos(2*theta[i])/6+mp.cos(theta[i])*8/3-5/2),-1],[1,0]])*mp.exp(0)
                elif dim==2:
                    kx=theta[i]*np.cos(phi[j])
                    ky=theta[i]*np.sin(phi[j])
                    time_2step=np.array([[2+pow(lamb,2)*((mp.cos(2*kx)/6+mp.cos(kx)*8/3-5/2)+(mp.cos(2*ky)/6+mp.cos(ky)*8/3-5/2)),-1],[1,0]])*mp.exp(0)
            elif spatial_discr=='inf':
                if dim==1:
                    time_2step=np.array([[2-pow(lamb*theta[i],2),-1],[1,0]])*mp.exp(0)
                elif dim==2:
                    kx=theta[i]*np.cos(phi[j])
                    ky=theta[i]*np.sin(phi[j])
                    time_2step=np.array([[2-pow(lamb,2)*(pow(kx,2)+pow(ky,2)),-1],[1,0]])*mp.exp(0)
            result=np.max(np.array([result,radio(time_2step)]))

    return result


def fun_rkn(x,y,grau):

    mu=1
    alpha=ssprk_alpha(mu,grau)

    z=x+1j*y
    rkn=alpha[0]

    aux=mp.exp(0)
    for j in range(1,grau-1):
        aux=aux*(1+mu*z)
        rkn=rkn+aux*alpha[j]
    rkn=rkn+pow(1+mu*z,2)*aux*alpha[grau-1]

    return rkn


def fun_faber(x,y,grau,gamma,c,d,coeff):
    z=x+1j*y

    c0=d/gamma
    c1=c**2/(4*gamma**2)

    faber=coeff[0]
    F1=z/gamma-c0
    F1_fix=F1
    faber=faber+F1*coeff[1]
    F2=F1_fix*F1-2*c1
    faber=faber+F2*coeff[2]

    for j in range(3,grau):
        F0=F1
        F1=F2
        F2=F1_fix*F1-c1*F0
        faber=faber+F2*coeff[j]

    return faber


def stability(degree):

    xmin=-10
    xmax=2
    ymin=-10
    ymax=10
    X, Y = np.meshgrid(np.hstack((np.linspace(xmin, pow(10,-8), 1500),np.array([-pow(10,-9),0,pow(10,-9)]),np.linspace(pow(10,-8),xmax, 1500))), np.linspace(ymin, ymax, 3003))

    np.save('Stability/X_1',X)
    np.save('Stability/Y_1',Y)

    # np.save('Stability/rk2',np.abs(fun_rk2(X,Y))-1)
    # np.save('Stability/2sm',fun_2sm(X,Y)-1)
    # np.save('Stability/rk4',np.abs(fun_rk4(X,Y))-1)
    # np.save('Stability/rk7',np.abs(fun_rk7(X,Y))-1)

    # gamma1,c1,d1,a_e=ellipse_properties(np.array([-10+10j,-10-10j,10j,-10j]),1)
    # gamma2,c2,d2,a_e=ellipse_properties(np.array([-5+5j,-5-5j,5j,-5j]),1)
    # gamma3,c3,d3,a_e=ellipse_properties(np.array([-1+1j,-1-1j,1j,-1j]),1)
    # gamma4,c4,d4,a_e=ellipse_properties(np.array([-1,-10j,10j]),1)

    # coefficients_faber1=np.array(Faber_approx_coeff(degree[-1]+1,gamma1,c1,d1).tolist(),dtype=np.float_)
    # coefficients_faber2=np.array(Faber_approx_coeff(degree[-1]+1,gamma2,c2,d2).tolist(),dtype=np.float_)
    # coefficients_faber3=np.array(Faber_approx_coeff(degree[-1]+1,gamma3,c3,d3).tolist(),dtype=np.float_)
    # coefficients_faber4=np.array(Faber_approx_coeff(degree[-1]+1,gamma4,c4,d4).tolist(),dtype=np.float_)
    for j in range(len(degree)):
        print(j)
        np.save('Stability/rkn_'+str(degree[j]),np.abs(fun_rkn(X,Y,degree[j]))-1)
        # np.save('Stability/faber_1_'+str(degree[j]),np.abs(np.abs(fun_faber(X,Y,degree[j],gamma1,c1,d1,coefficients_faber1))-1))
        # np.save('Stability/faber_2_'+str(degree[j]),np.abs(np.abs(fun_faber(X,Y,degree[j],gamma2,c2,d2,coefficients_faber2))-1))
        # np.save('Stability/faber_3_'+str(degree[j]),np.abs(np.abs(fun_faber(X,Y,degree[j],gamma3,c3,d3,coefficients_faber3))-1))
        # np.save('Stability/faber_4_'+str(degree[j]),np.abs(np.abs(fun_faber(X,Y,degree[j],gamma4,c4,d4,coefficients_faber4))-1))


def graph(degree):

    X=np.load('Stability/X_1.npy')
    Y=np.load('Stability/Y_1.npy')

    print(X[0,1000])
    print(X[0,1001])
    print(X[0,1002])

    # rk2=np.load('Stability/rk2.npy')
    # rk4=np.load('Stability/rk4.npy')
    # rk7=np.load('Stability/rk7.npy')
    # m_2sm=np.load('Stability/2sm.npy')
    #
    # fig, ax = plt.subplots()
    #
    # c1=ax.contour(X,Y,np.log(rk2),[-4],colors='b',linestyles='solid')
    # c2=ax.contour(X,Y,np.log(rk4),[-4],colors='g',linestyles='solid')
    # c3=ax.contour(X,Y,np.log(rk7),[-4],colors='purple',linestyles='solid')
    # c4=ax.contour(X,Y,np.log(m_2sm),[-4],colors='darkgoldenrod',linestyles='solid')
    #
    # lines = [ c1.collections[0], c2.collections[0], c3.collections[0],c4.collections[0]]
    # labels = ['RK3-2','RK4-4','RK9-7','2SM']
    # plt.legend(lines,labels,prop={'size': 15})
    # plt.axhline(0,color='k')
    # plt.axvline(0,color='k')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.savefig('Stability/low_order_domain.pdf')
    # plt.show()

    fig, ax = plt.subplots()
    color=np.array(['b','g','purple','cyan','lawngreen','palevioletred','darkgoldenrod','brown','navy','chocolate'])
    lines=np.zeros(0)
    labels=np.zeros(0)
    for i in range(len(degree)):
        print('i: ',i)
        rkn=np.array(np.load('Stability/rkn_'+str(degree[i])+'.npy',allow_pickle=True).tolist(), dtype=float)
        # plt.plot(Y[:,0],rkn[:,1000])
        # plt.show()
        c1=ax.contour(X-0.00671537,Y,np.log(rkn),[-4],colors=color[i],linestyles='solid')
        lines=np.hstack((lines,c1.collections[0]))
        order=degree[i]-2
        if degree[i]==2:
            order=1
        labels=np.hstack((labels,'RKHO'+str(degree[i])+'-'+str(order)))
    plt.axhline(0,color='k')
    plt.axvline(0,color='k')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(lines,labels,prop={'size': 15})
    plt.savefig('Stability/rkn_domain.pdf')
    plt.show()

    # fig, ax = plt.subplots()
    # lines=np.zeros(0)
    # labels=np.zeros(0)
    # for i in range(len(degree)):
    #     faber=np.array(np.load('Stability/faber_1_'+str(degree[i])+'.npy',allow_pickle=True).tolist(), dtype=float)
    #     c1=ax.contour(X,Y,np.log(faber),[-4],colors=color[i],linestyles='solid')
    #     lines=np.hstack((lines,c1.collections[0]))
    #     labels=np.hstack((labels,'FA'+str(degree[i])))
    # plt.axhline(0,color='k')
    # plt.axvline(0,color='k')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(lines,labels,prop={'size': 15})
    # plt.savefig('Stability/faber_1_ddomain.pdf')
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # lines=np.zeros(0)
    # labels=np.zeros(0)
    # for i in range(len(degree)):
    #     faber=np.array(np.load('Stability/faber_2_'+str(degree[i])+'.npy',allow_pickle=True).tolist(), dtype=float)
    #     c1=ax.contour(X,Y,np.log(faber),[-4],colors=color[i],linestyles='solid')
    #     lines=np.hstack((lines,c1.collections[0]))
    #     labels=np.hstack((labels,'FA'+str(degree[i])))
    # plt.axhline(0,color='k')
    # plt.axvline(0,color='k')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(lines,labels,prop={'size': 15})
    # plt.savefig('Stability/faber_2_domain.pdf')
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # lines=np.zeros(0)
    # labels=np.zeros(0)
    # for i in range(len(degree)):
    #     faber=np.array(np.load('Stability/faber_3_'+str(degree[i])+'.npy',allow_pickle=True).tolist(), dtype=float)
    #     c1=ax.contour(X,Y,np.log(faber),[-4],colors=color[i],linestyles='solid')
    #     lines=np.hstack((lines,c1.collections[0]))
    #     labels=np.hstack((labels,'FA'+str(degree[i])))
    # plt.axhline(0,color='k')
    # plt.axvline(0,color='k')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(lines,labels,prop={'size': 15})
    # plt.savefig('Stability/faber_3_domain.pdf')
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # lines=np.zeros(0)
    # labels=np.zeros(0)
    # for i in range(len(degree)):
    #     faber=np.array(np.load('Stability/faber_4_'+str(degree[i])+'.npy',allow_pickle=True).tolist(), dtype=float)
    #     c1=ax.contour(X,Y,np.log(faber),[-4],colors=color[i],linestyles='solid')
    #     lines=np.hstack((lines,c1.collections[0]))
    #     labels=np.hstack((labels,'FA'+str(degree[i])))
    # plt.axhline(0,color='k')
    # plt.axvline(0,color='k')
    # plt.xticks(fontsize=15)
    # plt.yticks(fontsize=15)
    # plt.legend(lines,labels,prop={'size': 15})
    # plt.savefig('Stability/faber_4_domain.pdf')
    # plt.show()


def interval(degree):

    tol=pow(10,-5)

    ymin=-16
    ymax=16

    N_points=5000
    X, Y = np.meshgrid(np.linspace(0, 0, 1), np.linspace(ymin, ymax, N_points))

    rk2=np.abs(fun_rk2(X,Y))-1
    m_2sm=np.abs(fun_2sm(X,Y))-1
    rk4=np.abs(fun_rk4(X,Y))-1
    rk7=np.abs(fun_rk7(X,Y))-1

    rk2_ind=0
    m_2sm_ind=0
    rk4_ind=0
    rk7_ind=0
    for i in range(int(N_points/2)):
        if rk2[i+int(N_points/2)]>tol and rk2_ind==0:
            print('rk2: ',Y[i+int(N_points/2)-1])
            rk2_ind=1
        if m_2sm[i+int(N_points/2)]>tol and m_2sm_ind==0:
            print('2SM: ',Y[i+int(N_points/2)-1])
            m_2sm_ind=1
        if rk4[i+int(N_points/2)]>tol and rk4_ind==0:
            print('rk4: ',Y[i+int(N_points/2)-1])
            rk4_ind=1
        if rk7[i+int(N_points/2)]>tol and rk7_ind==0:
            print('rk7: ',Y[i+int(N_points/2)-1])
            rk7_ind=1
        if rk2_ind+m_2sm_ind+rk4_ind+rk7_ind==4:
            break

    for i in range(len(degree)):
        rkn=np.array((np.abs(fun_rkn(X,Y,degree[i]))-1).tolist(), dtype=float)
        for j in range(int(N_points/2)):
            if rkn[j+int(N_points/2)]>tol:
                print('rkn_'+str(degree[i])+': ',Y[j+int(N_points/2)-1])
                break


mp.mp.dps=30


degree=np.array([3,4,9,15,20,25,30,35,40])
# degree=np.arange(3,41)

# degree=np.array([2,4,9])
# degree=np.array([9])
# stability(degree)
# interval(degree)
graph(degree)
# ""
# graph(10,100,50,100,20)
# graph(10,100,50,100,20)


# fun_rkn(0,0,3)

# g=lambda x,y: np.abs(np.abs(1+x+1j*y+pow(x+1j*y,2)/2)-1)
#
# xmin=-5
# xmax=5
# ymin=-5
# ymax=5
# X, Y = np.meshgrid(np.linspace(xmin, xmax, 2000), np.linspace(ymin, ymax, 2000))
# Z=g(X,Y)
# # plt.contourf(X, Y, np.log(Z), levels=[-10,-5])
# # plt.colorbar()
# plt.contour(np.log(Z),[-5])
# plt.show()


# rkn_3:  [1.73154631]
# rkn_4:  [2.82616523]
# rkn_5:  [0.44488898]
# rkn_6:  [0.70094019]
# rkn_7:  [1.76355271]
# rkn_8:  [3.3894779]
# rkn_9:  [4.57371474]
# rkn_10:  [1.71874375]
# rkn_11:  [2.17323465]
# rkn_12:  [3.38307662]
# rkn_13:  [5.00260052]
# rkn_14:  [6.30206041]
# rkn_15:  [3.31266253]
# rkn_16:  [3.81836367]
# rkn_17:  [4.98339668]
# rkn_18:  [6.59011802]
# rkn_19:  [7.9919984]
# rkn_20:  [5.02180436]
# rkn_21:  [5.52110422]
# rkn_22:  [6.59011802]
# rkn_23:  [8.16483297]
# rkn_24:  [9.65633127]
# rkn_25:  [6.78855771]
# rkn_26:  [7.24944989]
# rkn_27:  [8.20964193]
# rkn_28:  [9.73954791]
# rkn_29:  [11.27585517]
# rkn_30:  [8.6065213]
# rkn_31:  [9.00340068]
# rkn_32:  [9.84836967]
# rkn_33:  [11.30786157]
# rkn_34:  [12.87617524]
# rkn_35:  [10.47569514]
# rkn_36:  [10.77655531]
# rkn_37:  [11.5255051]
# rkn_38:  [12.8889778]
# rkn_39:  [14.45729146]
# rkn_40:  [15.84636927]