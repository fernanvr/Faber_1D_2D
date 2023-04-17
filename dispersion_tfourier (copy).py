import Methods as meth
import auxiliary_functions as aux_fun
import numpy as np
import scipy
import matplotlib.pyplot as plt


def domain_dispersion(dx,Nr,c,f0):
    # domain dimensions and amount of points
    # a=4*c*(Nr+3)/f0
    # b=14*c/f0
    a=0.3
    b=0.2
    nx=int(round(a/dx))
    ny=int(round(b/dx))
    print('nx: ',nx)
    print('ny: ',ny)

    # spatial grid points
    x=np.linspace(dx,a,nx)
    y=np.linspace(b,dx,ny)
    X,Y=np.meshgrid(x,y)
    np.save('Dispersion_S/X.npy',X)
    np.save('Dispersion_S/Y.npy',Y)
    X=np.expand_dims(X.flatten('F'),1)
    Y=np.expand_dims(Y.flatten('F'),1)

    # velocity field depending on the example
    param=np.zeros((ny,nx))+c**2
    param=np.expand_dims(param.flatten('F'),1)

    # source term position
    # x0=4*c/f0
    # y0=7*c/f0
    x0=0.05
    y0=0.1

    dt=dx/np.max(np.sqrt(param))/8

    return a,b,nx,ny,X,Y,param,dt,x0,y0


def source_disperion(x0,y0,X,Y,nx,ny,rad=0.02):
    f=aux_fun.source_x_2D(x0=x0,y0=y0,rad=rad,X=X,Y=Y,nx=nx,ny=ny,equ='scalar_dx2',delta=0)
    var0=np.zeros((len(f[:,0]),1))
    source_type='Dispersion_S'

    return var0,f,source_type


def solution_dispersion(method,degree,Ndt,dx=0.005,dx_factor=1,Nr=3,fig_ind=0):
    # Function to perform a dispersion analysis based in Fourier transform, and comparing the "method" with a reference
    # solution (RK9-7 with dx/2). For different receptors is computed the solution in a time interval where only a wavelet
    # is recorder, together with its fourier transform.

    # INPUT:
    # method: (string) the method to compute de dispersion
    # degree: degree used of the polynomial for the FA,HORK, and Krylov methods
    # Nr: (integer) number of receptors with a spacing of c/f0, the approximated wavelength of Ricker's wavelet, where c
    #      is the velocity
    # dx: (float) spatial discretization grid space
    # dx_factor: this is to know the factor between the reference solution and the solutions of the methods
    # Nr: numer of receivers used in the simulations
    # fig_ind: indicator if an image of the wave propagation at the three time cuts (see below)is saved


    # OUTPUT:
    # 4 files .npy:
    #   1 - with the solution using "method" until time T=NS*c/f*1.1
    #   2 - with the solution using the reference method until time T=NS*c/f*1.1
    #   3 - with the estimated dissipation functions
    #   4 - with the estimated phase change functions

    # velocity of the homogeneous medium and central frequency of Ricker wavelet
    c=0.2
    f0=15
    t0=1.2/f0+0.1
    param_ricker=np.array([f0,t0])

    # parameters of the domain where the numerical dispersion is computed
    a,b,nx,ny,X,Y,param,dt,x0,y0=domain_dispersion(dx,Nr,c,f0)

    # parameters of the source type (Ricker wavelet)
    var0,f,source_type=source_disperion(x0,y0,X,Y,nx,ny,0.02)

    # time steps given a smaller CFL condition
    # T=2/f0*Nr*1.1+t0
    # T=4/f0*(Nr+1)+t0
    T=1.2
    print(T)
    dt*=Ndt
    print('dt: ',dt)
    NDt=np.ceil(T/dt).astype(int)
    Dt=T/NDt
    print('NDt[0]: ',NDt[0])

    # receivers positions
    points=np.zeros(Nr).astype(int)
    print('a,b',a,b)
    for i in range(Nr):
        # points[i]=np.argmin(pow(X-(i+2)*4*c/f0,2)+pow(Y-7*c/f0,2))
        points[i]=np.argmin(pow(X-(i+2)*0.05,2)+pow(Y-0.1,2))
    # print(points)
    # print(X[points],Y[points])

    example='Dispersion_S'
    equ='scalar_dx2'
    T_frac_snapshot=1  # this value is only to not save the velocity field
    free_surf=1
    delta=0
    ord='8'
    dim=2
    beta0=30

    # time cuts for the three reciever positions register the wave
    cuts0=np.array([0.2710084,0.52209677,0.77277487])
    cuts=np.array([0.75,0.975,1.2])

    meth_ind,meth_label=method_label(method)
    meth.method_solver(method,var0,Ndt,NDt,Dt,T_frac_snapshot,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example,degree)
    var0=method_sol_load(meth_ind,meth_label,Ndt,dx,degree)

    if meth_ind<10:
        sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points.npy')[::dx_factor,:]
        for i in range(Nr):
            np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(i)+'.npy',sol[:int(cuts[i]/(dt*dx_factor*2)),i])
            transform=np.fft.fft(sol[:int(cuts[i]/(dt*dx_factor*2)),i])
            np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(i)+'_transform.npy',transform[:round(len(transform)/2)])
    else:
        if method=='FA':
            sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_1_ord_8_H_amplified_points_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')[::dx_factor,:]
        else:
            sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_1_ord_8_points_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')[::dx_factor,:]
        for i in range(Nr):
            np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(i)+'.npy',sol[:int(cuts[i]/(dt*dx_factor*2)),i])
            transform=np.fft.fft(sol[:int(cuts[i]/(dt*dx_factor*2)),i])
            np.save('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(i)+'_transform.npy',transform[:round(len(transform)/2)])

    if fig_ind==1:
        if meth_ind<10:
            sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'.npy')
        else:
            sol=np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')
        plt.imshow(sol.reshape((ny,nx),order='F'),extent=[0,0.3,0,0.2], aspect='auto')
        plt.scatter(np.array([x0]),np.array([y0]),s=50,color='b')
        plt.scatter(np.expand_dims(X.flatten('F'),1)[points],np.expand_dims(Y.flatten('F'),1)[points],s=50,color='k',marker='s')
        plt.savefig('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'.pdf')
        plt.show()


def method_label(method):
    if method=='RK7':
        return 9,'RK_ref'
    elif method=='RK2':
        return 3,'sol_rk2'
    elif method=='RK4':
        return 4,'sol_rk4'
    elif method=='2MS':
        return 1,'sol_2MS'
    elif method=='FA':
        return 10,'sol_faber'
    elif method=='HORK':
        return 10,'sol_rk'
    elif method=='KRY':
        return 10,'sol_krylov'

def method_sol_load(meth_ind,meth_label,Ndt,dx,degree):
    if meth_ind<10:
        return np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt)+'_dx_'+str(dx)+'_points.npy')
    else:
        if method_label=='FA':
            return np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_1_ord_8_H_amplified_points_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')
        else:
            return np.load('Dispersion_S/'+meth_label+'_equ_scalar_dx2_free_surf_1_ord_8_points_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'.npy')


def graph_wave_disp_diss(method,degree,Ndt,dx=0.005,Nr=np.array([0,1,2])):

    meth_ind,meth_label=method_label(method)
    cuts=np.array([0.75,0.975,1.2])

    for i in range(len(Nr)):
        sol_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_1_ord_8_Ndt_1_dx_'+str(dx/4)+'_points_'+str(Nr[i])+'.npy')[::Ndt[0]]
        if meth_ind<10:
            sol=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'.npy')
        else:
            sol=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'.npy')
        a=np.linspace(0,cuts[i],len(sol))
        print('t-----------------------------',a[np.abs(sol_ref)<1e-10])
        plt.plot(np.linspace(0,cuts[i],len(sol)),sol_ref,label='Reference',linewidth=2)
        plt.plot(np.linspace(0,cuts[i],len(sol)),sol,label=method,linewidth=2)
        plt.legend()
        plt.savefig('Dispersion_S/'+method+'_dx_'+str(dx)+'_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_points_'+str(i)+'.pdf')
        plt.show()
        plt.clf()

    # for i in range(len(Nr)):
    #     trans_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_1_ord_8_Ndt_1_dx_'+str(dx/4)+'_points_'+str(Nr[i])+'_transform.npy')[::Ndt[0]]
    #     if meth_ind<10:
    #         trans=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'_transform.npy')
    #     else:
    #         trans=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'_transform.npy')
    #     print(trans_ref.shape,trans.shape)
    #     a=trans_ref/trans
    #
    #     plt.plot(np.arange(len(trans))/cuts[i],np.abs(a),label='Receiver_'+str(i),linewidth=2)
    #
    # plt.legend()
    # plt.savefig('Dispersion_S/'+method+'_dx_'+str(dx)+'_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_amplitude.pdf')
    # plt.show()
    # # plt.clf()

    for i in range(len(Nr)):
        trans_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_1_ord_8_Ndt_1_dx_'+str(dx/4)+'_points_'+str(Nr[i])+'_transform.npy')[::Ndt[0]]
        if meth_ind<10:
            trans=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'_transform.npy')
        else:
            trans=np.load('Dispersion_S/'+method+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_dx_'+str(dx)+'_points_'+str(Nr[i])+'_transform.npy')

        a=trans_ref/trans
        # plt.plot(np.angle(trans_ref)*np.abs(trans_ref))
        # plt.plot(np.angle(trans)*np.abs(trans))
        # plt.show()
        # plt.plot(np.angle(trans_ref))
        # plt.plot(np.angle(trans))
        # plt.show()
        # print(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))
        plt.plot(np.angle(a))
        plt.show()
        a[(np.abs(trans_ref)+np.abs(trans))>0.01*(np.max(np.array([np.max(np.abs(trans_ref)),np.max(np.abs(trans))])))]=0
        plt.plot(np.angle(a))
        plt.show()

        # plt.plot(np.arange(len(trans))/cuts[i],np.angle(a),label='Receiver_'+str(i),linewidth=2)

    plt.legend()
    plt.savefig('Dispersion_S/'+method+'_dx_'+str(dx)+'_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[0])+'_phase.pdf')
    # plt.show()
    plt.clf()


def graph_estimate_diss_disp(methods,degree,Ndt,dx=0.005,Nr_ind=1,fig_ind='methods'):

    trans_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx/4)+'_points_'+str(Nr_ind)+'_transform.npy')

    cuts=np.array([0.75,0.975,1.2])

    # dispersion
    for i in range(len(methods)):
        meth_ind,meth_label=method_label(methods[i])
        if meth_ind<10:
            trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_transform.npy')
            a=trans_ref/trans
            dispersion_mes=np.trapz(np.abs(np.angle(a)),dx=1/cuts[Nr_ind])
            plt.scatter(np.array([meth_ind]),np.array([dispersion_mes]),label=methods[i],marker='s',s=40)
        else:
            dispersion_mes=np.zeros(len(degree))
            for j in range(len(degree)):
                trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_transform.npy')
                a=trans_ref/trans
                a[np.abs(trans)<1e-15]=0
                dispersion_mes[j]=np.trapz(np.abs(np.angle(a)),dx=1/cuts[Nr_ind])
            plt.plot(degree,dispersion_mes,label=methods[i],linewidth=2)
    plt.legend()
    plt.savefig('Dispersion_S/'+fig_ind+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_dispersion.pdf')
    plt.show()
    plt.clf()

    # dissipation
    for i in range(len(methods)):
        meth_ind,meth_label=method_label(methods[i])
        if meth_ind<10:
            trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_transform.npy')
            a=trans_ref/trans
            dissipation_mes=np.trapz(np.abs(a),dx=1/cuts[Nr_ind])
            plt.scatter(np.array([meth_ind]),np.array([dissipation_mes]),label=methods[i],marker='s',s=40)
        else:
            dissipation_mes=np.zeros(len(degree))
            for j in range(len(degree)):
                trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_transform.npy')
                a=trans_ref/trans
                dissipation_mes[j]=np.trapz(np.abs(a),dx=1/cuts[Nr_ind])
            plt.plot(degree,dissipation_mes,label=methods[i],linewidth=2)
    plt.legend()
    plt.savefig('Dispersion_S/'+fig_ind+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_dissipation.pdf')
    plt.show()


def graph_estimate_diss_disp_max_dt(methods,degree,Ndt,dx=0.005,Nr_ind=1,fig_ind='methods',tol=181,dt=0.003125):

    trans_ref=np.load('Dispersion_S/RK7_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx/4)+'_points_'+str(Nr_ind)+'_transform.npy')

    cuts=np.array([0.75,0.975,1.2])

    # dispersion max dt
    for i in range(len(methods)):
        meth_ind,meth_label=method_label(methods[i])
        if meth_ind<10:
            max_dt=0
            for j in range(len(Ndt)):
                trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_transform.npy')
                a=trans_ref/trans
                dispersion_mes=np.trapz(np.abs(np.angle(a)),dx=1/cuts[Nr_ind])
                if dispersion_mes>tol:
                    break
                else:
                    max_dt=dt*Ndt[j]
            plt.scatter(np.array([meth_ind]),np.array([max_dt]),label=methods[i],marker='s',s=40)
        else:
            max_dt=np.zeros(len(degree))
            for j in range(len(degree)):
                for k in range(len(Ndt)):
                    trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_transform.npy')
                    a=trans_ref/trans
                    dispersion_mes=np.trapz(np.abs(np.angle(a)),dx=1/cuts[Nr_ind])
                    if dispersion_mes>tol:
                        break
                    else:
                        max_dt[j]=dt*Ndt[k]
            plt.plot(degree,max_dt,label=methods[i],linewidth=2)
    plt.legend()
    plt.savefig('Dispersion_S/'+fig_ind+'_equ_scalar_dx2_free_surf_1_ord_8_dx_'+str(dx)+'_dispersion_max_dt.pdf')
    plt.show()
    plt.clf()

    # dissipation max dt
    for i in range(len(methods)):
        meth_ind,meth_label=method_label(methods[i])
        if meth_ind<10:
            max_dt=0
            for j in range(len(Ndt)):
                trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_transform.npy')
                a=trans_ref/trans
                dissipation_mes=np.trapz(np.abs(a),dx=1/cuts[Nr_ind])
                if dissipation_mes>tol:
                    break
                else:
                    max_dt=dt*Ndt[j]
            plt.scatter(np.array([meth_ind]),np.array([max_dt]),label=methods[i],marker='s',s=40)
        else:
            max_dt=np.zeros(len(degree))
            for j in range(len(degree)):
                for k in range(len(Ndt)):
                    trans=np.load('Dispersion_S/'+methods[i]+'_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[k])+'_degree_'+str(degree[j])+'_dx_'+str(dx)+'_points_'+str(Nr_ind)+'_transform.npy')
                    a=trans_ref/trans
                    dissipation_mes=np.trapz(np.abs(a),dx=1/cuts[Nr_ind])
                    if dissipation_mes>tol:
                        break
                    else:
                        max_dt[j]=dt*Ndt[k]
            plt.plot(degree,max_dt,label=methods[i],linewidth=2)
    plt.legend()
    plt.savefig('Dispersion_S/'+fig_ind+'_equ_scalar_dx2_free_surf_1_ord_8_dx_'+str(dx)+'_dissipation_max_dt.pdf')
    plt.show()









    # t=np.linspace(0,10,800)
    # fm=15
    # f=(1-2*(np.pi*fm*(t-5))**2)*np.exp(-(np.pi*fm*(t-5))**2)
    # print(np.argmax(np.abs(np.fft.fft(f))[:int(len(t)/2)])/t[-1])
    # plt.plot(np.arange(int(len(t)/2))/t[-1],np.abs(np.fft.fft(f))[:int(len(t)/2)])
    # # plt.plot(np.fft.fftfreq(1000),np.abs(np.fft.fft(f)))
    # # print(len(np.fft.fftfreq(1000)))
    # plt.show()
    # wert

    # for i in range(Nr):
    #     trans_ref=np.load('Dispersion_S/RK_ref_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx/4)+'_points_'+str(i)+'_transform.npy')
    #     trans=np.load('Dispersion_S/RK_ref_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(Ndt[0])+'_dx_'+str(dx)+'_points_'+str(i)+'_transform.npy')
    #
    #     cuts=np.array([0.75,0.975,1.2])
    #
    #     a=trans_ref/trans
    #
    #     # plt.plot(trans_ref,label='ref')
    #     # plt.plot(trans,label='orig')
    #     # plt.plot(trans*a,label='sol')
    #
    #     plt.plot(np.arange(len(trans))/cuts[i],np.abs(a),label='real')
    #     plt.plot(np.arange(len(trans))/cuts[i],np.angle(a),label='imag')
    #     plt.legend()
    #     plt.show()

        #
        #
        # N=len(trans)
        # m=5
        # A=np.ones((N,m+1))
        # for j in range(N):
        #     A[j,1:]=pow(j,np.arange(1,m+1))
        #     # A[j,:]=np.cos(j*2*np.pi/N*np.arange(m+1))
        # print(np.transpose(A).dot(A))
        # b=np.log(trans_ref)-np.log(trans)
        # sol=np.linalg.solve(np.transpose(A).dot(A),np.transpose(A).dot(b))
        # print(np.max(np.abs(np.transpose(A).dot(A).dot(sol)-np.transpose(A).dot(b))))
        # # print(np.transpose(A).dot(b))
        #
        # # import matplotlib.pyplot as plt
        # # plt.plot(b,label='b')
        # # plt.plot(np.abs(np.log(trans_ref)-A.dot(sol)-np.log(trans)),label='sol')
        # # plt.plot(np.abs(np.log(trans_ref) - np.log(trans)), label='orig')
        #
        # plt.plot(np.abs(trans_ref-np.exp(A.dot(sol))*trans),label='sol')
        # plt.plot(np.abs(trans_ref - trans), label='orig')
        #
        # # plt.plot(trans_ref,label='ref')
        # # plt.plot(trans,label='orig')
        # # plt.plot(np.exp(A.dot(sol))*trans,label='sol')
        # plt.legend()
        #
        # plt.show()
        #
        # # plt.plot((A.dot(sol)).imag,label='imag')
        # # plt.plot((A.dot(sol)).real,label='real')
        # # plt.legend()
        # # plt.show()








    # sol1=np.load('Dispersion_S/RK_ref_equ_scalar_dx2_free_surf_1_ord_8_Ndt_'+str(1)+'_dx_'+str(0.005)+'_points.npy')
    # print(sol[::4,:].shape)
    # print(sol1.shape)
    #
    #
    # a=np.fft.fft(sol[:int(0.75/dt)*2:4,0])
    # a1 = np.fft.fft(sol1[:int(0.75 / dt) * 2, 0])
    # a=a[:round(len(a)/2)]
    # a1 = a1[:round(len(a1) / 2)]
    #
    # plt.plot(a.real,label='a')
    # plt.plot(a1.real,label='a1')
    # plt.legend()
    # plt.show()

    # b=np.log(a)
    # plt.plot(b.real, label='real')
    # plt.plot(b.imag, label='imag')
    # plt.legend()
    # plt.show()

