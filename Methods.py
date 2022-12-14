import numpy as np
import auxiliary_functions as aux_fun
from time import time
import os


def domain_source(dx,T,Ndt,dim,equ,example,ord,delta):
    # return velocity field, source function parameters, time step size, and initial variable

    # INPUT:
    # dx: space discretization step size (float)
    # T: final time until the solution (float)
    # Ndt: Number of time-step sizes used to compute the solution (integer)
    # dim: dimension of the equations (integer: 1, 2)
    # equ: type of equation used (string)
    # example: identificator of an example (string)
    # abc: indentificator of absorbing boundary condition (binary)
    # ord: spatial discretization order (string: 4, 8)
    # delta: PML thickness (float)

    # OUTPUT:
    # nx: number (minus one) of the mesh grid points in the x-direction (integer)
    # ny: number (minus one) of the mesh grid points in the x-direction (integer)
    # X: x-axis of the points positions (float array)
    # Y: y-axis of the points positions (float array)
    # param: velocity values (or elastic constants, for the elastic case) in each mesh point (float array)
    # f: spatial part of the source term function (float array)
    # param_ricker: parameters of the Ricker wavelet (float array)
    # Dt: sizes of the different time teps (float array)
    # NDt: number of time steps for each time-step size (integer array)
    # points: fixed spatial points to save the displacement in the x-direction for all time instants (float array)
    # source_type: type of source used in the simulation (string)
    # var0: initial condition of the system of equations (float array)

    # velocity field and spatial grid definition
    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)

    # initial condition and source related
    var0,f,source_type=aux_fun.source_examples(equ,example,dim,delta,ord,dx,a,b,nx,ny,param,X,Y,x0,y0,T)

    # time steps given a smaller CFL condition
    print('dt: ',dt)
    Dt=np.linspace(dt,Ndt*dt,Ndt)       # different time steps for the solutions
    NDt=np.ceil(T/Dt).astype(int)
    Dt=T/NDt
    print('NDt[0]: ',NDt[0])

    if example[0]=='1':
        # spatial points to save the solution at each time instant
        points=np.array([3.675,6.300,7.875])
        points=(points/dx).astype(int)
    else:
        # spatial points to save the solution at each time instant
        points=np.array([[a/2,7/8*b],[a/2,5/8*b],[a/2,5/12*b]])
        points=np.array([np.argmin(pow(X-points[0,0],2)+pow(Y-points[0,1],2)),np.argmin(pow(X-points[1,0],2)+pow(Y-points[1,1],2)),np.argmin(pow(X-points[2,0],2)+pow(Y-points[2,1],2))])

    # parameters of a Ricker source type (used if the type of souyrce is Ricker)
    f0=25
    t0=1.2/f0+0.1
    T0=0.2
    param_ricker=np.array([f0,t0,T0])

    return nx,ny,X,Y,param,f,param_ricker,Dt,NDt,points,source_type,var0


def sol_RK_7(var0,NDt,Dt,equ,dim,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example):
    # Function to calculate the solution using a 9 stages Runge-Kutta of order 7, RK(9,7).

    # It saves the value of the displacement in the x-direction at the last time instant, for several time-steps  sizes,
    # and the displacement for all time instants in some specific points.

    # initialization of the array to save the solution at the last time instant
    RK_ref=np.zeros((nx*ny,len(NDt)))

    # cycle to compute the solution for each time-step size
    for i in range(len(NDt)):

        print('i',i)  # printing to know were the process is
        start=time()  # variable to compute the required computation time

        # initialization of the array to save the solution in the specific spatial points
        RK_ref_points=np.zeros((NDt[i],len(points)))

        var=var0 # solution inizalitation at time t_0

        # cycle to compute the solution using a specific time step size
        for j in range(NDt[i]):
            # calling the RK(9,7) method to compute the solution in the next time instant
            var=aux_fun.RK_7_source(var=var,dt=Dt[i],equ=equ,dim=dim,delta=delta,beta0=beta0,ord=ord,dx=dx,param=param,nx=nx+1,ny=ny+1,f=f,param_ricker=param_ricker,i=j,source_type=source_type)

            RK_ref_points[j,:]=var[points,0]

        np.save(str(example)+'/RK_ref_equ_'+str(equ)+'_ord_'+ord+'_points_'+str(i),RK_ref_points)

        RK_ref[:,i]=var[:nx*ny,0]

        print('time ',time()-start) # printing to know the amount of computational time taken

    np.save(str(example)+'/RK_ref_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx),RK_ref)

    return RK_ref


def sol_RK_2(var0,NDt,Dt,equ,dim,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example):
    # Function to calculate the solution using a 3 stages Runge-Kutta of order 2, RK(3,2).

    # It saves the value of the displacement in the x-direction at the last time instant, for several time-steps  sizes,
    # and the displacement for all time instants in some specific points.

    # initialization of the array to save the solution at the last time instant
    sol_rk2=np.zeros((nx*ny,len(NDt)))

    # cycle to compute the solution for each time-step size
    for i in range(len(NDt)):

        # initialization of the array to save the solution in the specific spatial points
        sol_rk2_points=np.zeros((NDt[i],len(points)))

        var=var0 # solution inizalitation at time t_0

        # cycle to compute the solution using a specific time step size
        for j in range(NDt[i]):
            # calling the RK(3,2) method to compute the solution in the next time instant
            var=aux_fun.RK_2(var=var,dt=Dt[i],equ=equ,dim=dim,delta=delta,beta0=beta0,ord=ord,dx=dx,param=param,nx=nx+1,ny=ny+1,i=j,f=f,param_ricker=param_ricker,source_type=source_type)

            sol_rk2_points[j,:]=var[points,0]

        np.save(str(example)+'/sol_rk2_equ_'+str(equ)+'_ord_'+ord+'_points_'+str(i),sol_rk2_points)

        sol_rk2[:,i]=var[:nx*ny,0]

    np.save(str(example)+'/sol_rk2_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx),sol_rk2)

    return sol_rk2


def sol_RK_4(var0,NDt,Dt,equ,dim,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example):
    # Function to calculate the solution using a 4 stages Runge-Kutta of order 4, RK(4,4).

    # It saves the value of the displacement in the x-direction at the last time instant, for several time-steps  sizes,
    # and the displacement for all time instants in some specific points.

    # initialization of the array to save the solution at the last time instant
    sol_rk4=np.zeros((nx*ny,len(NDt)))

    # cycle to compute the solution for each time-step size
    for i in range(len(NDt)):

        # initialization of the array to save the solution in the specific spatial points
        sol_rk4_points=np.zeros((NDt[i],len(points)))

        var=var0 # solution inizalitation at time t_0

        # cycle to compute the solution using a specific time step size
        for j in range(NDt[i]):
            # calling the RK(9,7) method to compute the solution in the next time instant
            var=aux_fun.RK_4(var=var,dt=Dt[i],equ=equ,dim=dim,delta=delta,beta0=beta0,ord=ord,dx=dx,param=param,nx=nx+1,ny=ny+1,i=j,f=f,param_ricker=param_ricker,source_type=source_type)

            sol_rk4_points[j,:]=var[points,0]

        np.save(str(example)+'/sol_rk4_equ_'+str(equ)+'_ord_'+ord+'_points_'+str(i),sol_rk4_points)

        sol_rk4[:,i]=var[:nx*ny,0]

    np.save(str(example)+'/sol_rk4_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx),sol_rk4)

    return sol_rk4


def sol_time_2step(var0,NDt,Dt,equ,dim,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example):
    # Function to calculate the solution using a Leapfrog scheme.

    # It saves the value of the displacement in the x-direction at the last time instant, for several time-steps  sizes,
    # and the displacement for all time instants in some specific points.

    # initialization of the array to save the solution at the last time instant
    sol_2time=np.zeros((nx*ny,len(NDt)))

    var0=aux_fun.ini_var0_2MS(var0,nx,ny,dim,delta) # solution inizalitation at time t_0

    # cycle to compute the solution for each time-step size
    for i in range(len(NDt)):
        # calling the Leapfrog method to compute the solution at the final time instant
        var=aux_fun.method_time_2steps(var0=var0,Nt=NDt[i],dt=Dt[i],nx=nx,ny=ny,dx=dx,c2=param,source_type=source_type,f=f[:nx*ny,:],param_ricker=param_ricker,equ=equ,dim=dim,delta=delta,beta0=beta0,ord=ord,points=points,example=example,i=i)

        sol_2time[:,i]=var[:nx*ny,0]

    np.save(str(example)+'/sol_2time_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx),sol_2time)

    return sol_2time


def sol_faber(var0,NDt,Dt,equ,dim,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example,degree,ind_source):
    # Function to calculate the solution using a Faber polynomials scheme.

    # It saves the value of the displacement in the x-direction at the last time instant for several polynoamil degrees,
    # and for several time-steps  sizes. Also saves the displacement for all time instants in some specific points.

    # estimate of the convex envelope of the spectrum of the discretized operator]
    vals=aux_fun.spectral_dist(equ,dim,delta,beta0,ord,dx,param)

    # calculation of the optimal ellipse parameters
    gamma,c,d,a_e=aux_fun.ellipse_properties(vals,1)

    # initialization of the array to save the solution at the last time instant
    sol_faber=np.zeros((len(degree),nx*ny))

    # cycle to compute the solution for each time-step size
    for i in range(len(NDt)):

        # to save computations, watch if the solution was calculated for this time-step size
        if os.path.isfile(str(example)+'/sol_faber_equ_'+str(equ)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(i)+'_dx_'+str(dx)+'.npy'):
            continue

        print('i-------------------------------------------------------',i) # printing to know were the process is
        start=time()  # variable to compute the required computation time

        # computation of Faber polynomials coefficient
        coefficients_faber=np.array(aux_fun.Faber_approx_coeff(degree[-1]+1,gamma*Dt[i],c*Dt[i],d*Dt[i]).tolist(),dtype=np.float_)

        # condition to stop if not all the coefficients were calculated (then, the precision is not good)
        if coefficients_faber[-1]==0:
            print('break')
            break

        # initialization of the array to save the solution at the specific spatial points
        sol_faber_points=np.zeros((NDt[i],len(points)))

        # cycle to compute the solution for each polynomial degree
        for j in range(len(degree)):

            # condition to work with the expanded version of H or the source term Faber expansion
            if ind_source=='H_amplified':
                ext=np.zeros((degree[j]+1,1))
                ext[degree[j],0]=1
                var=np.vstack((var0,ext))
                uk_core=aux_fun.g_core(p=degree[j]+1,f0=param_ricker[0],t0=param_ricker[1],source_type=source_type)
            else:
                uk_core=0
                u_k=0
                var=var0*1

            # cycle to compute the solution using a specific time step size
            for l in range(NDt[i]):

                # condition to work with the expanded version of H or the source term Faber expansion
                if ind_source=='H_amplified':
                    u_k=aux_fun.g_approx(f=f,p=degree[j]+1,f0=param_ricker[0],t0=param_ricker[1],t=l*Dt[i],source_type=source_type,uk_core=uk_core)
                    if np.max(np.abs(u_k))>pow(10,-5):
                        eta=np.max(np.sum(np.abs(u_k),axis=0))
                        u_k=pow(2,-np.log2(eta))*u_k
                        var[-1]=pow(2,np.log2(eta))
                    else:
                        var[-1]=1

                # calculating the solution with Faber polynomials on the next time instant
                var=aux_fun.Faber_approx(var,degree[j]+1,gamma,c,d,equ,dim,delta,beta0,ord,dx,param,nx+1,ny+1,coefficients_faber,ind_source,u_k)

                # condition to work with the expanded version of H or the source term Faber expansion
                if ind_source=='H_amplified':
                    var[-(degree[j]+1):]=0
                    sol_faber_points[l,:]=var[points,0]
                elif ind_source=='FA_ricker' and (pow(np.pi*param_ricker[0]*(l*Dt[i]-param_ricker[1]),2)<45 or ((l+1)*Dt[i]-param_ricker[1])*(param_ricker[1]-l*Dt[i])>0):
                    f_aux=aux_fun.Faber_approx(f,degree[j]+1,gamma,c,d,equ,dim,delta,beta0,ord,dx,param,nx+1,ny+1,np.array(aux_fun.Faber_ricker_coeff(equ,degree[j]+1,l*Dt[i],Dt[i],param_ricker[0],param_ricker[1],gamma*Dt[i],c*Dt[i],d*Dt[i]).tolist(),dtype=np.float_),ind_source,u_k)
                    var=var+f_aux
                    sol_faber_points[l,:]=var[points,0]+f_aux[points,0]

            sol_faber[j,:]=var[:nx*ny,0]

            np.save(str(example)+'/sol_faber_equ_'+str(equ)+'_ord_'+ord+'_'+ind_source+'_points_Ndt_'+str(i)+'_degree_'+str(degree[j]),sol_faber_points)

        np.save(str(example)+'/sol_faber_equ_'+str(equ)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(i)+'_dx_'+str(dx),sol_faber)

        print('time ',time()-start) # printing to know the amount of computational time taken


def sol_rk(var0,NDt,Dt,equ,dim,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example,degree):
    # Function to calculate the solution using arbitrary high order Runge-Kutta (HORK).

    # It saves the value of the displacement in the x-direction at the last time instant for several polynoamil degrees,
    # and for several time-steps  sizes. Also saves the displacement for all time instants in some specific points.

    # initialization of the array to save the solution at the last time instant
    sol_rk=np.zeros((len(degree),nx*ny))

    # cycle to compute the solution for each polynomial degree
    for i in range(len(NDt)):
        print(i) # printing to know were the process is

        # initialization of the array to save the solution at the specific spatial points
        sol_rk_points=np.zeros((NDt[i],len(points)))

        # cycle to compute the solution for each polynomial degree
        for j in range(len(degree)):

            # constructing the amplified matrix
            ext=np.zeros((degree[j]+1,1))
            ext[degree[j],0]=1
            var=np.vstack((var0,ext))
            uk_core=aux_fun.g_core(p=degree[j]+1,f0=param_ricker[0],t0=param_ricker[1],source_type=source_type)

            # cycle to compute the solution using a specific time step size
            for l in range(NDt[i]):

                # updating the amplified matrix
                u_k=aux_fun.g_approx(f=f,p=degree[j]+1,f0=param_ricker[0],t0=param_ricker[1],t=l*Dt[i],source_type=source_type,uk_core=uk_core)
                if np.max(np.abs(u_k))>pow(10,-5):
                    eta=np.max(np.sum(np.abs(u_k),axis=0))
                    u_k=pow(2,-np.log2(eta))*u_k
                    var[-1]=pow(2,np.log2(eta))
                else:
                    var[-1]=1

                # calculating the solution with HORK on the next time instant
                var=aux_fun.RK_op(var,1,Dt[i],equ,dim,delta,beta0,ord,dx,param,nx+1,ny+1,degree[j],Dt[i],u_k)

                sol_rk_points[l,:]=var[points,0]

                # updating the amplified matrix
                var[-(degree[j]+1):]=0

            sol_rk[j,:]=var[:nx*ny,0]

            np.save(str(example)+'/sol_rk_equ_'+str(equ)+'_ord_'+ord+'_points_Ndt_'+str(i)+'_degree_'+str(degree[j]),sol_rk_points)

        np.save(str(example)+'/sol_rk_equ_'+str(equ)+'_ord_'+ord+'_Ndt_'+str(i)+'_dx_'+str(dx),sol_rk)


def sol_krylov(var0,NDt,Dt,equ,dim,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example,degree):
    # Function to calculate the solution using Krylov subspace method.

    # It saves the value of the displacement in the x-direction at the last time instant for several polynoamil degrees,
    # and for several time-steps  sizes. Also saves the displacement for all time instants in some specific points.

    # initialization of the array to save the solution at the last time instant
    sol_krylov=np.zeros((len(degree),nx*ny))

    # cycle to compute the solution for each polynomial degree
    for i in range(len(NDt)):
        print(i) # printing to know were the process is

        # initialization of the array to save the solution at the specific spatial points
        sol_krylov_points=np.zeros((NDt[i],len(points)))

        # cycle to compute the solution for each polynomial degree
        for j in range(len(degree)):

            # constructing the amplified matrix
            ext=np.zeros((degree[j]+1,1))
            ext[degree[j],0]=1
            var=np.vstack((var0,ext))
            uk_core=aux_fun.g_core(p=degree[j]+1,f0=param_ricker[0],t0=param_ricker[1],source_type=source_type)

            # cycle to compute the solution using a specific time step size
            for l in range(NDt[i]):
                start = time()  # to check the amount of time taked to compute the solution with Krylov

                # updating the amplified matrix
                u_k=aux_fun.g_approx(f=f,p=degree[j]+1,f0=param_ricker[0],t0=param_ricker[1],t=l*Dt[i],source_type=source_type,uk_core=uk_core)
                if np.max(np.abs(u_k))>pow(10,-5):
                    eta=np.max(np.sum(np.abs(u_k),axis=0))
                    u_k=pow(2,-np.log2(eta))*u_k
                    var[-1]=pow(2,np.log2(eta))
                else:
                    var[-1]=1

                # calculating the solution with HORK on the next time instant
                var=aux_fun.krylov_op(var,degree[j],Dt[i],equ,dim,delta,beta0,ord,dx,param,nx+1,ny+1,u_k)

                sol_krylov_points[l,:]=var[points,0]

                # updating the amplified matrix
                var[-(degree[j]+1):]=0

                if l%10==0:
                    print('degree, l and time',degree[j],l,time()-start) # checking the time nedded to compute some intervals

            sol_krylov[j,:]=var[:nx*ny,0]

            np.save(str(example)+'/sol_krylov_equ_'+str(equ)+'_ord_'+ord+'_points_Ndt_'+str(i)+'_degree_'+str(degree[j]),sol_krylov_points)

        np.save(str(example)+'/sol_krylov_equ_'+str(equ)+'_ord_'+ord+'_Ndt_'+str(i)+'_dx_'+str(dx),sol_krylov)

