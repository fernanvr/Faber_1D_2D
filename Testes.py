import sys
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import Methods as meth
import numpy as np


def wave_eq(dx,equ,dim,free_surf,delta,beta0,ord,T,T_frac_snapshot,Ndt,degree,example,method,ind_source='H_amplified',save_step=True,replace=0):

    # solution of 1d acoustic equation with font term and constructed solutions with PML to validate accuracy
    # of the methods: RK7, Faber, RK-High order, RK2, Devito

    # INPUT:
    # dx: spatial discretization step size (float)
    # equ: equation formulation (scalar, scalar_dx2, elastic)
    # dim: dimension of the problem (1, 2)
    # delta: thickness of the PML domain (float)
    # beta0: damping parameter of the PML layer (float)
    # ord: spatial discretization order ('4','8')
    # T: end time of the simulation (float)
    # Ndt: amount of time-step sizes used to compute the solutions (int)
    # degree: Faber polynomial degrees used to compute the solution (array int)
    # example: selection of the wave equation parameters (string)
    # method: type of numerical method to solve the equations ('RK7', 'FA')
    # ind_source: indicator of the wave equations' source term treatment ('H_amplified', 'FA_ricker')


    print('method',method,'degree',degree,'Ndt',Ndt)

    # Model parameters
    nx,ny,X,Y,param,f,param_ricker,Dt,NDt,points,source_type,var0=meth.domain_source(dx,T,T_frac_snapshot,Ndt,dim,equ,example,ord,delta)

    meth.method_solver(method,var0,Ndt,NDt,Dt,T_frac_snapshot,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example,degree,ind_source,replace,save_step)


def error_acoustic(Ndt,degree,example,dx,c2_max):

    X=np.load(str(example)+'/X.npy')
    solution=np.load(str(example)+'/solution.npy')
    RK_ref=np.load(str(example)+'/RK_ref_dx_'+str(dx)+'.npy')
    sol_rk2=np.load(str(example)+'/sol_rk2_dx_'+str(dx)+'.npy')
    sol_rk4=np.load(str(example)+'/sol_rk4_dx_'+str(dx)+'.npy')
    time_2step=np.load(str(example)+'/sol_2time_dx_'+str(dx)+'.npy')

    # plt.plot(X,solution)
    # plt.show()
    # asdfasd


    dt=(np.arange(Ndt)+1)*dx/c2_max
    eps=-6
    if example[0]=='2':
        dx=dx*dx


    err_RK_ref=np.log10(np.sqrt(np.sum(pow(np.abs(RK_ref-solution),2),axis=0)*dx))
    err_sol_rk2=np.log10(np.sqrt(np.sum(pow(np.abs(sol_rk2-solution),2),axis=0)*dx))
    err_sol_rk4=np.log10(np.sqrt(np.sum(pow(np.abs(sol_rk4-solution),2),axis=0)*dx))
    err_time_2step=np.log10(np.sqrt(np.sum(pow(np.abs(time_2step-solution),2),axis=0)*dx))

    # err_RK_ref=np.log10(np.max(np.abs(RK_ref-solution),axis=0))
    # err_sol_rk2=np.log10(np.max(np.abs(sol_rk2-solution),axis=0))
    # err_sol_rk4=np.log10(np.max(np.abs(sol_rk4-solution),axis=0))
    # err_time_2step=np.log10(np.max(np.abs(time_2step-solution),axis=0))

    err_RK_ref[(err_RK_ref>1)+(np.isnan(err_RK_ref))>0]=1
    err_sol_rk2[(err_sol_rk2>1)+(np.isnan(err_sol_rk2))>0]=1
    err_sol_rk4[(err_sol_rk4>1)+(np.isnan(err_sol_rk4))>0]=1
    err_time_2step[(err_time_2step>1) + (np.isnan(err_time_2step))>0]=1

    plt.plot(dt,err_RK_ref,label='RK9-7',linewidth=2)
    plt.plot(dt,err_sol_rk2,label='RK3-2',linewidth=2)
    plt.plot(dt,err_sol_rk4,label='RK4-4',linewidth=2)
    plt.plot(dt,err_time_2step,label='2MS',linewidth=2)
    plt.legend()
    plt.xlabel('time-step $\Delta t$',size='16')
    plt.ylabel('$log_{10}(||Err||_2)$',size='16')
    plt.savefig(str(example)+'/low_order_'+str(example)+'.pdf')
    plt.show()

    err_RK_ref[0]=eps-1
    err_sol_rk2[0]=eps-1
    err_sol_rk4[0]=eps-1
    err_time_2step[0]=eps-1

    print('err_RK_ref ',np.max(dt[err_RK_ref<=eps]))
    print('err_sol_rk2 ',np.max(dt[err_sol_rk2<=eps]))
    print('err_sol_rk4 ',np.max(dt[err_sol_rk4<=eps]))
    print('err_time_2step ',np.max(dt[err_time_2step<=eps]))

    degree=np.array([0,1,6,11,16,21,26,31,36])
    err_sol_faber=np.zeros((len(degree),Ndt))
    for i in range(np.minimum(Ndt,42)):
        sol_faber=np.load(str(example)+'/sol_faber_Ndt_'+str(i)+'_dx_'+str(dx)+'.npy')[degree,:]
        print(sol_faber.shape)
        err_sol_faber[:,i]=np.log10(np.sqrt(np.sum(pow(np.abs(sol_faber-np.transpose(solution)),2),axis=1)*dx))
        # err_sol_faber[:,i]=np.log10(np.max(np.abs(sol_faber-np.transpose(solution)),axis=1))
    err_sol_faber[(err_sol_faber>1)+(np.isnan(err_sol_faber))>0]=1

    # plt.plot(np.load(str(example)+'/sol_faber_Ndt_'+str(0)+'.npy')[-1,:])
    # plt.plot(solution)
    # plt.show()
    # sadfdsa
    for i in range(len(degree)):
        plt.plot(dt,err_sol_faber[i,:],label='FA'+str(degree[i]),linewidth=2)
        err_sol_faber[i,0]=eps-1
        print('err_sol_faber_'+str(degree[i])+' ',np.max(dt[err_sol_faber[i,:]<=eps]))

    plt.legend()
    plt.xlabel('time-step $\Delta t$',size='16')
    plt.ylabel('$log_{10}(||Err||_2)$',size='16')
    plt.savefig(str(example)+'/faber_'+str(example)+'.pdf')
    plt.show()

    err_sol_rk=np.zeros((len(degree),Ndt))
    for i in range(Ndt):
        sol_rk=np.load(str(example)+'/sol_rk_Ndt_'+str(i)+'_dx_'+str(dx)+'.npy')[degree,:]
        err_sol_rk[:,i]=np.log10(np.sqrt(np.sum(pow(np.abs(sol_rk-np.transpose(solution)),2),axis=1)*dx))
        # err_sol_rk[:,i]=np.log10(np.max(np.abs(sol_rk-np.transpose(solution)),axis=1))
    err_sol_rk[(err_sol_rk>1)+(np.isnan(err_sol_rk))>0]=1

    for i in range(len(degree)):
        order=degree[i]-2
        if degree[i]<5:
            order=degree[i]
        plt.plot(dt,err_sol_rk[i,:],label='RKHO'+str(degree[i])+'-'+str(order),linewidth=2)
        err_sol_rk[i,0]=eps-1
        print('err_sol_rk_'+str(degree[i])+' ',np.max(dt[err_sol_rk[i,:]<=eps]))

    plt.legend()
    plt.xlabel('time-step $\Delta t$',size='16')
    plt.ylabel('$log_{10}(||Err||_2)$',size='16')
    plt.savefig(str(example)+'/rkn_'+str(example)+'.pdf')
    plt.show()


def eff_graph(Ndt,degree,example,dx,c2_max):

    eff_low_order=np.zeros(4)
    eff_rkn=np.zeros(len(degree))
    eff_faber=np.zeros(len(degree))

    solution=np.load(str(example)+'/solution.npy')
    rk7=np.load(str(example)+'/RK_ref_dx_'+str(dx)+'.npy')
    rk2=np.load(str(example)+'/sol_rk2_dx_'+str(dx)+'.npy')
    rk4=np.load(str(example)+'/sol_rk4_dx_'+str(dx)+'.npy')
    time_2step=np.load(str(example)+'/sol_2time_dx_'+str(dx)+'.npy')

    dt=(np.arange(Ndt)+1)*dx/c2_max
    tol=-6
    if example[0]=='2':
        dx=dx*dx

    err_rk7=np.log10(np.sqrt(np.sum(pow(np.abs(rk7-solution),2),axis=0)*dx))
    err_rk2=np.log10(np.sqrt(np.sum(pow(np.abs(rk2-solution),2),axis=0)*dx))
    err_rk4=np.log10(np.sqrt(np.sum(pow(np.abs(rk4-solution),2),axis=0)*dx))
    err_time_2step=np.log10(np.sqrt(np.sum(pow(np.abs(time_2step-solution),2),axis=0)*dx))

    err_rk7[(err_rk7>1)+(np.isnan(err_rk7))>0]=1
    err_rk2[(err_rk2>1)+(np.isnan(err_rk2))>0]=1
    err_rk4[(err_rk4>1)+(np.isnan(err_rk4))>0]=1
    err_time_2step[(err_time_2step>1) + (np.isnan(err_time_2step))>0]=1

    ind_time_2step=0
    ind_rk2=0
    ind_rk4=0
    ind_rk7=0
    for i in range(len(dt)):
        if ind_time_2step==0 and err_time_2step[i]>tol:
            ind_time_2step=1
            if i==0:
                eff_low_order[0]=0
            else:
                eff_low_order[0]=dt[i-1]
        if ind_rk2==0 and err_rk2[i]>tol:
            ind_rk2=1
            if i==0:
                eff_low_order[1]=0
            else:
                eff_low_order[1]=dt[i-1]/3
        if ind_rk4==0 and err_rk4[i]>tol:
            ind_rk4=1
            eff_low_order[2]=dt[i-1]/4
        if ind_rk7==0 and err_rk7[i]>tol:
            ind_rk7=1
            eff_low_order[3]=dt[i-1]/9
        if ind_time_2step+ind_rk2+ind_rk4+ind_rk7==4:
            break

    err_faber=np.zeros((len(degree),Ndt))
    for i in range(np.minimum(Ndt,42)):
        faber=np.load(str(example)+'/sol_faber_Ndt_'+str(i)+'_dx_'+str(dx)+'.npy')
        err_faber[:,i]=np.log10(np.sqrt(np.sum(pow(np.abs(faber-np.transpose(solution)),2),axis=1)*dx))
    err_faber[(err_faber>1)+(np.isnan(err_faber))>0]=1

    for i in range(len(degree)):
        for j in range(len(dt)):
            if err_faber[i,j]>tol:
                if j==0:
                    eff_faber[i]=0
                else:
                    eff_faber[i]=dt[j-1]/degree[i]
                break
            if j==len(dt)-1:
                eff_faber[i]=dt[j]/degree[i]

    err_rkn=np.zeros((len(degree),Ndt))
    for i in range(Ndt):
        rkn=np.load(str(example)+'/sol_rk_Ndt_'+str(i)+'_dx_'+str(dx)+'.npy')
        err_rkn[:,i]=np.log10(np.sqrt(np.sum(pow(np.abs(rkn-np.transpose(solution)),2),axis=1)*dx))
    err_rkn[(err_rkn>1)+(np.isnan(err_rkn))>0]=1

    for i in range(len(degree)):
        for j in range(len(dt)):
            if err_rkn[i,j]>tol:
                eff_rkn[i]=dt[j-1]/degree[i]
                break
            if j==len(dt)-1:
                eff_rkn[i]=dt[j]/degree[i]

    plt.scatter(1,eff_low_order[0],label='2MS',color='b')
    plt.scatter(3,eff_low_order[1],label='RK3-2',color='g')
    plt.scatter(4,eff_low_order[2],label='RK4-4',color='purple')
    plt.scatter(9,eff_low_order[3],label='RK9-7',color='cyan')
    plt.plot(degree,eff_rkn,label='RKHO',color='lawngreen',linewidth=2)
    plt.plot(degree,eff_faber,label='FA',color='palevioletred',linewidth=2)
    plt.legend()
    plt.ylabel('$E_{ff}$',fontsize=20)
    plt.xlabel('# operations',fontsize=20)
    plt.savefig(str(example)+'/eff.pdf')
    plt.show()


def sources_acoustic_1D(Ndt,T,degree,example,point):

    RK_ref_points=np.load(str(example)+'/RK_ref_points_'+str(0))



    plt.plot(np.linspace(0,T,RK_ref_points.shape[0]),RK_ref_points[:,point])




    sol_rk2_points=np.load(str(example)+'/sol_rk2_points_'+str(0))
    sol_rk4_points=np.load(str(example)+'/sol_rk4_points_'+str(0))
    sol_2time_points=np.load(str(example)+'/sol_2time_points_'+str(0))

    plt.plot(np.linspace(0,T,RK_ref_points.shape[0]),sol_rk2_points[:,point])
    plt.plot(np.linspace(0,T,RK_ref_points.shape[0]),sol_rk4_points[:,point])
    plt.plot(np.linspace(0,T,RK_ref_points.shape[0]),sol_2time_points[:,point])


def acoustic_eigen(dx,equ,dim,abc,delta,beta0,ord,T,Ndt,example):
    # solution of 1d acoustic equation with font term and constructed solutions with PML to validate accuracy
    # of the methods: RK7, Faber, RK-High order, RK2, Devito

    eigen_real=np.zeros(len(dx))
    eigen_imag=np.zeros(len(dx))
    eigen_real_lim=np.zeros(len(dx))
    eigen_imag_lim=np.zeros(len(dx))
    for i in range(len(dx)):
        print('i: ',i)
        # Model parameters
        a,b,nx,ny,X,c2,x0,f,f0,t0,T0,f1,f2,dt,Dt,NDt,points,source_type,var0=domain_source(dx[i],T,Ndt,example,abc)

        if dim==1:
            eigen_real_lim[i]=-beta0
            eigen_imag_lim[i]=2.4*np.max(np.sqrt(c2))/dx[i]
        elif dim==2:
            eigen_real_lim[i]=-beta0*pow((delta-dx[i]/2)/delta,2)
            eigen_imag_lim[i]=3.3*np.max(np.sqrt(c2))/dx[i]

        def lin_op_H(var):
            var=np.expand_dims(var,axis=1)
            return op_H(var,equ,dim,abc,delta,beta0,ord,dx[i],c2,nx+1,ny+1)
        H=LinearOperator(shape=(len(var0),len(var0)),matvec=lin_op_H)

        start=time()
        if dim==1:
            vals,vecs=eigs(H,np.min(np.array([1500,3*nx-2])),which='SR')
            vals1,vecs1=eigs(H,np.min(np.array([1500,3*nx-2])),which='LM')
        elif dim==2:
            # np.min(np.array([3*nx*ny-2,1000]))
            # vals,vecs=eigs(H,5*nx*ny-2,which='LM')
            # vals,vecs=eigs(H,np.min(np.array([2000,5*nx*ny-2])),which='SR')
            # vals1,vecs1=eigs(H,np.min(np.array([2000,5*nx*ny-2])),which='LM')
            vals,vecs=eigs(H,np.min(np.array([50,5*nx*ny-2])),which='SR')
            vals1,vecs1=eigs(H,np.min(np.array([50,5*nx*ny-2])),which='LM')
        end=time()
        print('time: ',end-start)

        eigen_real[i]=np.min(vals.real)
        eigen_imag[i]=np.max(vals1.imag)
        if(end-start>7200):
            break

    np.save('eigenvalues/eigen_real_'+str(dim),eigen_real)
    np.save('eigenvalues/eigen_imag_'+str(dim),eigen_imag)
    np.save('eigenvalues/eigen_real_lim_'+str(dim),eigen_real_lim)
    np.save('eigenvalues/eigen_imag_lim_'+str(dim),eigen_imag_lim)
    eigen_real=np.load('eigenvalues/eigen_real_'+str(dim)+'.npy')
    eigen_imag=np.load('eigenvalues/eigen_imag_'+str(dim)+'.npy')
    eigen_real_lim=np.load('eigenvalues/eigen_real_lim_'+str(dim)+'.npy')
    eigen_imag_lim=np.load('eigenvalues/eigen_imag_lim_'+str(dim)+'.npy')

    plt.plot(np.arange(1,100),eigen_real,color='b',linewidth=2)
    plt.plot(np.arange(1,100),eigen_real_lim,color='r',linewidth=2)
    plt.legend()
    plt.savefig('eigenvalues/real_dim_'+str(dim)+'.pdf')
    plt.show()

    plt.plot(np.arange(1,100),eigen_imag,'b',linewidth=2)
    plt.plot(np.arange(1,100),eigen_imag_lim,'r',linewidth=2)
    plt.legend()
    plt.savefig('eigenvalues/imag_dim_'+str(dim)+'.pdf')
    plt.show()

    # plt.scatter(vals.real,vals.imag,alpha=0.7)
    # plt.axhline(y=0, color='k')
    # plt.axvline(x=0, color='k')
    # #*pow((delta-dx/2)/delta,2)
    # if dim==1:
    #     red_x=np.linspace(-beta0,0,1000)
    #     red_y=np.linspace(-2.4*np.max(np.sqrt(c2))/dx,2.4*np.max(np.sqrt(c2))/dx,1000)
    # elif dim==2:
    #     red_x=np.linspace(-beta0*pow((delta-dx/2)/delta,2),0,1000)
    #     red_y=np.linspace(-3.3*np.max(np.sqrt(c2))/dx,3.3*np.max(np.sqrt(c2))/dx,1000)
    # plt.plot(np.zeros(1000)+red_x[0],red_y,color='red',linewidth=2,linestyle='--')
    # plt.plot(np.zeros(1000)+red_x[-1],red_y,color='red',linewidth=2,linestyle='--')
    # plt.plot(red_x,np.zeros(1000)+red_y[0],color='red',linewidth=2,linestyle='--')
    # plt.plot(red_x,np.zeros(1000)+red_y[-1],color='red',linewidth=2,linestyle='--')
    # plt.savefig('eigenval_dim_'+str(dim)+'_dx'+str(dx)+'.pdf')
    # plt.show()


degree=np.arange(3,31,2)

# wave_eq(0.04,'scalar_dx2',2,1,0.8,30,'8',1.3,1,np.array([1]),degree,'2D_homogeneous_0b','2MS',replace=1)
# wave_eq(0.04,'scalar_dx2',2,1,0.8,30,'8',1.5,1,np.array([1]),degree,'Marmousi_b','2MS')
# wave_eq(0.04,'scalar_dx2',2,1,0.8,30,'8',2,1,np.array([1]),degree,'SEG_EAGE_b','2MS','H_amplified')
# wave_eq(0.04,'scalar_dx2',2,1,0.8,30,'8',3,1,np.array([1]),degree,'piece_GdM_b','2MS','H_amplified')
wave_eq(0.04,'scalar_dx2',2,1,0.8,30,'8',1.1,1,np.array([1]),degree,'2D_heterogeneous_3b','2MS',replace=1)

# wave_eq(0.005,'scalar_dx2',2,1,0.8,30,'8',2.2,1/2,np.array([1]),degree,'2D_heterogeneous_3c','RK7','H_amplified')

