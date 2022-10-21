import matplotlib.pyplot as plt
import numpy as np
import auxiliary_functions as aux_fun
import os
from matplotlib.ticker import FormatStrFormatter


def error_acoustic(Ndt,example,dim,dx,c2_max,equ,ord,ind_source):
    # graphic of Faber polynomials error, for different degrees

    # INPUT:
    # Ndt: amount of time-step sizes used to compute the solutions (int)
    # example: selection of the wave equation parameters (string)
    # dim: dimension of the problem (1, 2)
    # dx: spatial discretization step size (float)
    # c2_max: maximum velocity of the medium (float)
    # equ: equation formulation (scalar, scalar_dx2, elastic)
    # ord: spatial discretization order ('4','8')
    # ind_source: indicator of the wave equations' source term treatment ('H_amplified', 'FA_ricker')

    # loading the reference solution
    RK_ref=np.load(str(example)+'/RK_ref_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')

    # time step sizes used
    dt=(np.arange(Ndt)+1)*dx/(8*c2_max)

    # defining the degrees for which the error will be calculated
    degree=np.arange(5,40,5)
    degree=np.arange(5,30,5)

    # calculating the error with norm ||.||_2
    err_sol_faber=np.zeros((len(degree),Ndt))
    for i in range(Ndt):
        sol_faber=np.load(str(example)+'/sol_faber_equ_'+str(equ)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(i)+'_dx_'+str(dx)+'.npy')[degree-3,:]
        err_sol_faber[:,i]=np.sqrt(np.sum(pow(sol_faber-np.transpose(RK_ref),2),axis=1)*dx)

    # if the dimension is 2, we have to calibrate the norm
    if dim==2:
        err_sol_faber=err_sol_faber*np.sqrt(dx)

    # ploting the error for the polynomial degrees declared before
    ax=plt.gca()
    for i in range(len(degree)):
        # plt.plot(dt,err_sol_faber[i,:],label='FA'+str(degree[i]),linewidth=3,marker='D')
        lin,=ax.plot(dt,err_sol_faber[i,:],linewidth=3)
        ax.scatter(dt,err_sol_faber[i,:], linewidth=3,marker='D',alpha=0.5)
        ax.plot([],[],label='FA'+str(degree[i]),color=lin.get_color(),linewidth=3,marker='D')

    plt.legend(fontsize=19)
    plt.xlabel('time-step $\Delta t$',size='23')
    plt.ylabel('$||Err||_2$',size='22')
    plt.ticklabel_format(style="sci", scilimits=(0,0))
    plt.gca().xaxis.get_offset_text().set_fontsize(17)
    plt.yscale('log')
    # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.yticks(pow(10.0,np.arange(0, -17, step=-2)),fontsize=17)
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.9)
    plt.legend(fontsize=19)
    plt.ylim(pow(10,-16),30)
    plt.savefig(str(example)+'/faber_equ_'+equ+'_'+str(example)+'_ord_'+ord+'_source_'+ind_source+'.pdf')
    plt.show()


def graph_experiments(example,ord,equ,ind_source,max_degree,dx,c2_max,ind,Ndt):
    # graphics of the maximum \Delta t allowed by different approximation degrees of Faber polynomials,
    # using 4 different experiment as most (because of the markers and line types)

    # INPUT:
    # example: selection of the wave equation parameters (string)
    # ord: spatial discretization order ('4','8')
    # equ: equation formulation (scalar, scalar_dx2, elastic)
    # ind_source: indicator of the wave equations' source term treatment ('H_amplified', 'FA_ricker')
    # max_degree: maximum polynomial degree used to graphic (int)
    # dx: spatial discretization step size (float)
    # c2_max: maximum velocity of the medium (float)
    # ind: string used to put in the figure name, for saving
    # Ndt: amount of time-step sizes used to compute the solutions (int)

    # tolerance error such that the maximum \Delta t allowed produce an error which is at most, the tolerance
    eps=1.e-6

    # graphic markers and lines
    graph_marker=np.array(['D','o','s','v'])
    graph_type=np.array(['-','--','-.',':'])
    ax=plt.gca()

    # cycle of the maximum \Delta t of the polynomial degrees for the different experiments
    for i in range(len(equ)):

        # loading the reference solution
        RK_ref=np.load(str(example[i])+'/RK_ref_equ_'+str(equ[i])+'_ord_'+ord[i]+'_dx_'+str(dx)+'.npy')

        # initializing the array with the error of Faber approximation
        err_sol_faber=np.zeros((max_degree,Ndt))

        # initializing the array with the maximum \Delta t
        delta_t=np.zeros(max_degree)

        # cycle to compute the error
        for j in range(Ndt):
            sol_faber=np.load(str(example[i])+'/sol_faber_equ_'+str(equ[i])+'_ord_'+ord[i]+'_'+ind_source+'_Ndt_'+str(j)+'_dx_'+str(dx)+'.npy')[:max_degree,:]
            err_sol_faber[:,j]=np.sqrt(np.sum(pow(sol_faber-np.transpose(RK_ref),2),axis=1)*dx)

        # cycle to compute the maximum \Delta t
        for j in range(max_degree):
            delta_t[j]=np.sum(err_sol_faber[j,:]<eps)
        delta_t=delta_t*dx/(8*c2_max[i])

        # building the labels for the solutions
        label_str=''
        if example[i]=='1D_homogeneous_0':
            label_str=label_str+'TC#1_'
        elif example[i]=='1D_heterogeneous_1a':
            label_str=label_str+'TC#2_'
        elif example[i]=='1D_heterogeneous_2':
            label_str=label_str+'TC#3_'
        elif example[i]=='2D_homogeneous_0a':
            label_str=label_str+'TC#4_'
        elif example[i]=='2D_heterogeneous_3a':
            label_str=label_str+'TC#5_'
        elif example[i]=='2D_heterogeneous_2':
            label_str=label_str+'TC#6_'
        elif example[i]=='2D_heterogeneous_3':
            label_str=label_str+'TC#7_'
        label_str=label_str+'ord'+ord[i]
        if equ[i]=='scalar':
            label_str=label_str+'_1SD'
        elif equ[i]=='scalar_dx2':
            label_str=label_str+'_2SD'

        # plotting the maximum \Delta t by polynomial degree
        # plt.plot(3+np.arange(max_degree),delta_t,label=label_str,linewidth=2,marker=graph_marker[i],linestyle=graph_type[i])
        lin,=ax.plot(3+np.arange(max_degree),delta_t,linewidth=2,linestyle=graph_type[i])
        ax.scatter(3+np.arange(max_degree),delta_t, linewidth=2,marker=graph_marker[i],alpha=0.5)
        ax.plot([],[],label=label_str,color=lin.get_color(),linewidth=2,marker=graph_marker[i],linestyle=graph_type[i])

    # graph parameters and figure saving
    plt.xlabel('Polynomial degree',fontsize=23)
    plt.ylabel(r'$\Delta t_{max}$',fontsize=23)
    plt.gca().yaxis.get_offset_text().set_fontsize(17)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=20)
    plt.ticklabel_format(axis="y",style="sci", scilimits=(0,0))
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.subplots_adjust(left=0.18, bottom=0.15, right=0.9, top=0.95)
    plt.savefig('faber_experiments_'+ind+'.pdf')
    plt.show()


def eff_graph_experiments(example,ord,equ,ind_source,max_degree,dx,c2_max,ind,Ndt):
    # graphics of Faber efficiency for each polynomial degree (number of matrix-vector operations by time length),
    # using 4 different experiment as most (because of the markers and line types)

    # INPUT:
    # example: selection of the wave equation parameters (string)
    # ord: spatial discretization order ('4','8')
    # equ: equation formulation (scalar, scalar_dx2, elastic)
    # ind_source: indicator of the wave equations' source term treatment ('H_amplified', 'FA_ricker')
    # max_degree: maximum polynomial degree used to graphic (int)
    # dx: spatial discretization step size (float)
    # c2_max: maximum velocity of the medium (float)
    # ind: string used to put in the figure name, for saving
    # Ndt: amount of time-step sizes used to compute the solutions (int)

    # tolerance error such that the efficiency is computed with eps as the maximum allowed error
    eps=1.e-6

    # graphic markers and lines
    graph_marker=np.array(['D','o','s','v'])
    graph_type=np.array(['-','--','-.',':'])
    ax=plt.gca()

    # cycle of the maximum \Delta t of the polynomial degrees for the different experiments
    for i in range(len(equ)):

        # loading the reference solution
        RK_ref=np.load(str(example[i])+'/RK_ref_equ_'+str(equ[i])+'_ord_'+ord[i]+'_dx_'+str(dx)+'.npy')

        # initializing the array with the error of Faber approximation
        err_sol_faber=np.zeros((max_degree,Ndt))

        # initializing the array with the maximum \Delta t
        delta_t=np.zeros(max_degree)

        # cycle to compute the error
        for j in range(Ndt):
            sol_faber=np.load(str(example[i])+'/sol_faber_equ_'+str(equ[i])+'_ord_'+ord[i]+'_'+ind_source+'_Ndt_'+str(j)+'_dx_'+str(dx)+'.npy')[:max_degree,:]
            err_sol_faber[:,j]=np.sqrt(np.sum(pow(sol_faber-np.transpose(RK_ref),2),axis=1)*dx)
            # print(err_sol_faber[:,j])

        # cycle to compute the efficiency
        for j in range(max_degree):
            delta_t[j]=np.sum(err_sol_faber[j,:]<eps)/(j+3)
        delta_t=delta_t*dx/c2_max[i]/8

        # building the labels for the solutions
        label_str=''
        if example[i]=='1D_homogeneous_0':
            label_str=label_str+'TC#1_'
        elif example[i]=='1D_heterogeneous_1a':
            label_str=label_str+'TC#2_'
        elif example[i]=='1D_heterogeneous_2':
            label_str=label_str+'TC#3_'
        elif example[i]=='2D_homogeneous_0a':
            label_str=label_str+'TC#4_'
        elif example[i]=='2D_heterogeneous_3a':
            label_str=label_str+'TC#5_'
        elif example[i]=='2D_heterogeneous_2':
            label_str=label_str+'TC#6_'
        elif example[i]=='2D_heterogeneous_3':
            label_str=label_str+'TC#7_'
        label_str=label_str+'ord'+ord[i]
        if equ[i]=='scalar':
            label_str=label_str+'_1SD'
        elif equ[i]=='scalar_dx2':
            label_str=label_str+'_2SD'

        # plotting the maximum \Delta t by polynomial degree
        # plt.plot(3+np.arange(max_degree),1/delta_t,label=label_str,linewidth=2,marker=graph_marker[i],linestyle=graph_type[i])
        lin,=ax.plot(3+np.arange(max_degree),1/delta_t,linewidth=2,linestyle=graph_type[i])
        ax.scatter(3+np.arange(max_degree),1/delta_t, linewidth=2,marker=graph_marker[i],alpha=0.5)
        ax.plot([],[],label=label_str,color=lin.get_color(),linewidth=2,marker=graph_marker[i],linestyle=graph_type[i])

    # graph parameters and figure saving
    plt.xlabel('Polynomial degree',fontsize=23)
    plt.ylabel(r'$N^{\Delta t}_{op}$',fontsize=23)
    plt.ticklabel_format(axis="y",style="sci", scilimits=(0,0))
    plt.gca().yaxis.get_offset_text().set_fontsize(17)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.legend(prop={'size': 20})
    # plt.ylim([1-2*pow(10,-10), 1+1.02*pow(10,-5)])
    # plt.xlim([-0.01, 1])
    plt.subplots_adjust(left=0.18, bottom=0.15, right=0.9, top=0.9)
    plt.savefig('faber_experiments_eff_'+ind+'.pdf')
    plt.show()


def spatial_error(example,equ,ord,ind_source,dx,nx,ny,x0_pos,Ndt_0,dephs_y,degree):
    # computing the spatial error over a line in a 2D propagation problem, for different Faber polynomial degrees,
    # in a fixed time instant. In addition, is save also the snapshoot of the wave at the time instant and the region
    # used for the error calculation.

    # INPUT:
    # example: selection of the wave equation parameters (string)
    # equ: equation formulation (scalar, scalar_dx2, elastic)
    # ord: spatial discretization order ('4','8')
    # ind_source: indicator of the wave equations' source term treatment ('H_amplified', 'FA_ricker')
    # dx: spatial discretization step size (float)
    # nx: number (minus one) of the mesh grid points in the x-direction (integer)
    # ny: number (minus one) of the mesh grid points in the x-direction (integer)
    # x0_pos: x-axis position to compute the error along y-axis (int)
    # Ndt_0: to specify a particular time-step size to compare (int, is a multiplier of the minimum time-step size)
    # dephs_y: profundity of the physical medium
    # degree: Faber polynomial degrees used to compute the solution (array int)


    # loading and reshaping the reference solution
    RK_ref=np.load(str(example)+'/RK_ref_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')
    RK_ref=RK_ref.reshape((ny,nx),order='F')

    # construction of the wave snapshot at the final time instant
    extent = [0, 8, 8,0]
    fig=plt.imshow(RK_ref,extent=extent,alpha=0.9)
    plt.xlabel('X Position [km]',fontsize=20)
    plt.ylabel('Depth [km]',fontsize=20)
    cbar=plt.colorbar(fig)
    cbar.set_label('Displacement [km]',fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.yaxis.get_offset_text().set(size=13)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    # plt.plot(np.array([x0_pos*dx,x0_pos*dx]),np.array([0,8]),'k')
    # plt.gca().set_aspect(1.7, adjustable='box')
    plt.draw()
    plt.tight_layout()
    plt.savefig(example+'/wave_propagation_example0.pdf')
    plt.show()

    sdfgs

    # calculating and plotting the error
    depth=np.linspace(0,dephs_y,ny)
    faber=np.load(str(example)+'/sol_faber_equ_'+str(equ)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(Ndt_0)+'_dx_'+str(dx)+'.npy')
    for i in range(len(degree)):
        aux=faber[degree[i],:]
        aux=aux.reshape((ny,nx),order='F')
        plt.plot(depth,np.abs(RK_ref[:,x0_pos]-aux[:,x0_pos]),linestyle='-',label='FA'+str(degree[i]),linewidth=2)

    # graph parameters and figure saving
    plt.xlabel('Depth [km]',size='22')
    plt.ylabel('Error ($||\cdot||_{\infty}$)',size='22')
    plt.yscale('log')
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=19)
    plt.ylim(pow(10,-20))
    plt.tight_layout()
    # plt.subplots_adjust(left=0.18, bottom=0.15, right=0.9, top=0.9)
    plt.savefig(example+'/error_spatial_cut.pdf')
    plt.show()


def graph_methods_dt_max(example,dim,ord,equ,delta,ind_source,Ndt,degree,dx):

    dt_low_order=np.zeros(4)
    dt_faber=np.zeros(len(degree))
    dt_rkn=np.zeros(len(degree))

    solution=np.load(str(example)+'/RK_ref_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx/2)+'_T23.npy')
    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx/2,delta,equ)
    solution=solution.reshape((ny,nx),order='F')
    solution=solution[1::2,:-1:2]
    c2_max=np.max(param)
    solution=np.expand_dims(solution.flatten('F'),1)

    rk7=np.load(str(example)+'/RK_ref_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')
    rk2=np.load(str(example)+'/sol_rk2_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')
    rk4=np.load(str(example)+'/sol_rk4_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')
    time_2step=np.load(str(example)+'/sol_2time_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')

    tol=4*pow(10,-6)
    dx_factor=dx+0
    if dim==2:
        dx_factor=dx*dx

    err_rk7=np.sqrt(np.sum(pow(rk7-solution,2),axis=0)*dx_factor)
    err_rk2=np.sqrt(np.sum(pow(np.abs(rk2-solution),2),axis=0)*dx_factor)
    err_rk4=np.sqrt(np.sum(pow(np.abs(rk4-solution),2),axis=0)*dx_factor)
    err_time_2step=np.sqrt(np.sum(pow(np.abs(time_2step-solution),2),axis=0)*dx_factor)


    dt_low_order[0]=np.sum(err_time_2step<tol)
    dt_low_order[1]=np.sum(err_rk2<tol)
    dt_low_order[2]=np.sum(err_rk4<tol)
    dt_low_order[3]=np.sum(err_rk7<tol)
    dt_low_order=dt_low_order*dx/(8*c2_max)

    err_faber=np.zeros((len(degree),Ndt))
    for i in range(np.minimum(Ndt,42)):
        faber=np.load(str(example)+'/sol_faber_equ_'+str(equ)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(i)+'_dx_'+str(dx)+'.npy')
        err_faber[:,i]=np.sqrt(np.sum(pow(faber-np.transpose(solution),2),axis=1)*dx_factor)

    for i in range(len(degree)):
        dt_faber[i]=np.sum(err_faber[i,:]<tol)
    dt_faber=dt_faber*dx/(8*c2_max)

    err_rkn=np.zeros((len(degree),Ndt))
    for i in range(Ndt):
        rkn=np.load(str(example)+'/sol_rk_equ_'+str(equ)+'_ord_'+ord+'_Ndt_'+str(i)+'_dx_'+str(dx)+'.npy')
        err_rkn[:,i]=np.sqrt(np.sum(pow(rkn-np.transpose(solution),2),axis=1)*dx_factor)

    for i in range(len(degree)):
        dt_rkn[i]=np.sum(err_rkn[i,:]<tol)
    dt_rkn=dt_rkn*dx/(8*c2_max)

    ax=plt.gca()
    # # 2MS
    # ax.scatter(1,dt_low_order[0], linewidth=2,marker='D',alpha=0.5,color='#9467bd',s=50)
    # ax.scatter(1,dt_low_order[0],marker='D',color='purple',s=20)
    # ax.scatter([],[],label='2MS',color='#9467bd',linewidth=2,marker='D')
    # RK3-2
    ax.scatter(3,dt_low_order[1], linewidth=2,marker='o',alpha=0.5,color='#8c564b',s=50)
    ax.scatter(3,dt_low_order[1],marker='o',color='k',s=20)
    ax.scatter([],[],label='RK3-2',color='#8c564b',linewidth=2,marker='o')
    # RK4-4
    ax.scatter(4,dt_low_order[2], linewidth=2,marker='s',alpha=0.5,color='#2ca02c',s=50)
    ax.scatter(4,dt_low_order[2],marker='s',color='g',s=20)
    ax.scatter([],[],label='RK4-4',color='#2ca02c',linewidth=2,marker='s')
    # RK9-7
    ax.scatter(9,dt_low_order[3], linewidth=2,marker='v',alpha=0.5,color='#d62728',s=50)
    ax.scatter(9,dt_low_order[3],marker='v',color='r',s=20)
    ax.scatter([],[],label='RK9-7',color='#d62728',linewidth=2,marker='v')
    # RKHO
    lin,=ax.plot(degree,dt_rkn,linewidth=2,alpha=0.9,color='#1f77b4')
    ax.scatter(degree,dt_rkn, linewidth=2,marker='P',alpha=0.5,color='#1f77b4')
    ax.scatter(degree,dt_rkn,marker='P',color='b',s=5)
    ax.plot([],[],label='RKHO',color='#1f77b4',linewidth=2,marker='P')
    # FA
    lin,=ax.plot(degree,dt_faber,linewidth=2,linestyle='--',alpha=0.9,color='#ff7f0e')
    ax.scatter(degree,dt_faber, linewidth=2,marker='H',alpha=0.5,color='#ff7f0e')
    ax.scatter(degree,dt_faber,marker='H',color='orange',s=5)
    ax.plot([],[],label='FA',color='#ff7f0e',linewidth=2,marker='H',linestyle='--')

    # plt.scatter(1,dt_low_order[0],label='2MS',color='b')
    # plt.scatter(3,dt_low_order[1],label='RK3-2',color='g')
    # plt.scatter(4,dt_low_order[2],label='RK4-4',color='purple')
    # plt.scatter(9,dt_low_order[3],label='RK9-7',color='cyan')
    # plt.plot(degree,dt_rkn,label='RKHO',color='lawngreen',linewidth=2)
    # plt.plot(degree,dt_faber,label='FA',color='palevioletred',linewidth=2)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=20)
    plt.ylabel('$\Delta t_{max}$',fontsize=20)
    plt.xlabel('Polynomial degree',fontsize=20)
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.95)
    plt.savefig(str(example)+'/methods_dt.pdf')
    plt.show()


def graph_methods_error(example,dim,ord,equ,delta,ind_source,Ndt,degree,dx):

    error_low_order=np.zeros(4)
    error_faber=np.zeros(len(degree))
    error_rkn=np.zeros(len(degree))

    solution=np.load(str(example)+'/RK_ref_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx/2)+'_T23.npy')
    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx/2,delta,equ)
    solution=solution.reshape((ny,nx),order='F')
    solution=solution[1::2,:-1:2]
    c2_max=np.max(param)
    solution=np.expand_dims(solution.flatten('F'),1)

    rk7=np.load(str(example)+'/RK_ref_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')
    rk2=np.load(str(example)+'/sol_rk2_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')
    rk4=np.load(str(example)+'/sol_rk4_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')
    time_2step=np.load(str(example)+'/sol_2time_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')

    tol=4*pow(10,-6)
    dx_factor=dx+0
    if dim==2:
        dx_factor=dx*dx

    print('dt', dx / (8 * c2_max))


    error_low_order[0]=np.sqrt(np.sum(pow(np.abs(time_2step-solution),2),axis=0)*dx_factor)[0]
    error_low_order[1]=np.sqrt(np.sum(pow(np.abs(rk2-solution),2),axis=0)*dx_factor)[0]
    error_low_order[2]=np.sqrt(np.sum(pow(np.abs(rk4-solution),2),axis=0)*dx_factor)[0]
    error_low_order[3]=np.sqrt(np.sum(pow(rk7-solution,2),axis=0)*dx_factor)[0]

    err_faber=np.zeros((len(degree),1))
    faber=np.load(str(example)+'/sol_faber_equ_'+str(equ)+'_ord_'+ord+'_'+ind_source+'_Ndt_0_dx_'+str(dx)+'.npy')
    err_faber[:,0]=np.sqrt(np.sum(pow(faber-np.transpose(solution),2),axis=1)*dx_factor)

    for i in range(len(degree)):
        error_faber[i]=err_faber[i,0]

    err_rkn=np.zeros((len(degree),1))
    rkn=np.load(str(example)+'/sol_rk_equ_'+str(equ)+'_ord_'+ord+'_Ndt_0_dx_'+str(dx)+'.npy')
    err_rkn[:,0]=np.sqrt(np.sum(pow(rkn-np.transpose(solution),2),axis=1)*dx_factor)

    for i in range(len(degree)):
        error_rkn[i]=err_rkn[i,0]

    ax=plt.gca()
    # # 2MS
    # ax.scatter(1,error_low_order[0], linewidth=2,marker='D',alpha=0.5,color='#9467bd',s=50)
    # ax.scatter(1,error_low_order[0],marker='D',color='purple',s=20)
    # ax.scatter([],[],label='2MS',color='#9467bd',linewidth=2,marker='D')
    # RK3-2
    ax.scatter(3,error_low_order[1], linewidth=2,marker='o',alpha=0.5,color='#8c564b',s=50)
    ax.scatter(3,error_low_order[1],marker='o',color='k',s=20)
    ax.scatter([],[],label='RK3-2',color='#8c564b',linewidth=2,marker='o')
    # RK4-4
    ax.scatter(4,error_low_order[2], linewidth=2,marker='s',alpha=0.5,color='#2ca02c',s=50)
    ax.scatter(4,error_low_order[2],marker='s',color='g',s=20)
    ax.scatter([],[],label='RK4-4',color='#2ca02c',linewidth=2,marker='s')
    # RK9-7
    ax.scatter(9,error_low_order[3], linewidth=2,marker='v',alpha=0.5,color='#d62728',s=50)
    ax.scatter(9,error_low_order[3],marker='v',color='r',s=20)
    ax.scatter([],[],label='RK9-7',color='#d62728',linewidth=2,marker='v')
    # RKHO
    lin,=ax.plot(degree,error_rkn,linewidth=2,alpha=0.9,color='#1f77b4')
    ax.scatter(degree,error_rkn, linewidth=2,marker='P',alpha=0.5,color='#1f77b4')
    ax.scatter(degree,error_rkn,marker='P',color='b',s=5)
    ax.plot([],[],label='RKHO',color='#1f77b4',linewidth=2,marker='P')
    # FA
    lin,=ax.plot(degree,error_faber,linewidth=2,linestyle='--',alpha=0.9,color='#ff7f0e')
    ax.scatter(degree,error_faber, linewidth=2,marker='H',alpha=0.5,color='#ff7f0e')
    ax.scatter(degree,error_faber,marker='H',color='orange',s=5)
    ax.plot([],[],label='FA',color='#ff7f0e',linewidth=2,marker='H',linestyle='--')

    # plt.scatter(1,dt_low_order[0],label='2MS',color='b')
    # plt.scatter(3,dt_low_order[1],label='RK3-2',color='g')
    # plt.scatter(4,dt_low_order[2],label='RK4-4',color='purple')
    # plt.scatter(9,dt_low_order[3],label='RK9-7',color='cyan')
    # plt.plot(degree,dt_rkn,label='RKHO',color='lawngreen',linewidth=2)
    # plt.plot(degree,dt_faber,label='FA',color='palevioletred',linewidth=2)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.legend(fontsize=20)
    plt.ylabel('$||E_{rr}||_2$',fontsize=20)
    plt.xlabel('Polynomial degree',fontsize=20)
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.95)
    plt.savefig(str(example)+'/methods_error.pdf')
    plt.show()


def graph_velocities(example,dx,delta,equ):
    # graph of the vvelocity medium with PML boundary

    # INPUT
    # example: identificator of an example (string)
    # dx: space discretization step size (float)
    # delta: PML thickness (float)
    # equ: type of equation used (string)

    # cheking if there exist the paste to save the results, and creating one if there is not
    if not os.path.isdir(example+'/'):
        os.mkdir(example)

    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)

    b= -b
    extent = [0,a, b,0]

    if equ!='elastic':
        param=param.reshape((ny,nx),order='F')
        fig=plt.imshow(param,extent=extent)
        plt.xlabel('X Position [km]',fontsize=20)
        plt.ylabel('Depth [km]',fontsize=20)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        cbar=plt.colorbar(fig)
        cbar.set_label('Velocity [km/s]',fontsize=20)
        cbar.ax.tick_params(labelsize=15)
        plt.grid()

        print(x0)
        print(y0)
        # drawing the point where the source term is located
        plt.plot(x0,-y0,'ok')

        # drawing the limits of the PML domain
        plt.plot(np.array([delta,delta]),np.array([delta,b-delta]),'k')
        plt.plot(np.array([a-delta,a-delta]),np.array([delta,b-delta]),'k')
        plt.plot(np.array([delta,a-delta]),np.array([delta,delta]),'k')
        plt.plot(np.array([delta,a-delta]),np.array([b-delta,b-delta]),'k')

        # saving and showing the picture
        plt.tight_layout()
        plt.savefig(example+'/vel_map.pdf')
        plt.show()
    else:
        # p-wave velocity
        velocity_p=np.sqrt((param[:,4]+2*param[:,2])*np.reciprocal(param[:,0]))
        velocity_p=velocity_p.reshape((ny,nx),order='F')
        fig=plt.imshow(velocity_p,extent=extent)
        plt.xlabel('X Position [km]',fontsize=20)
        plt.ylabel('Depth [km]',fontsize=20)
        cbar=plt.colorbar(fig)
        cbar.set_label('Velocity [km/s]')
        plt.grid()

        print(x0)
        print(y0)
        # drawing the point where the source term is located
        plt.plot(x0,b-y0,'ok')

        # drawing the limits of the PML domain
        plt.plot(np.array([delta,delta]),np.array([delta,b-delta]),'k')
        plt.plot(np.array([a-delta,a-delta]),np.array([delta,b-delta]),'k')
        plt.plot(np.array([delta,a-delta]),np.array([delta,delta]),'k')
        plt.plot(np.array([delta,a-delta]),np.array([b-delta,b-delta]),'k')

        # saving and showing the picture
        plt.savefig(example+'/vel_P_map.pdf')
        plt.show()

        # s-wave velocity
        velocity_s=np.sqrt(param[:,2]*np.reciprocal(param[:,0]))
        velocity_s=velocity_s.reshape((ny,nx),order='F')
        fig=plt.imshow(velocity_s,extent=extent)
        plt.xlabel('X Position [km]',fontsize=20)
        plt.ylabel('Depth [km]',fontsize=20)
        cbar=plt.colorbar(fig)
        cbar.set_label('Velocity [km/s]')
        plt.grid()

        print(x0)
        print(y0)
        # drawing the point where the source term is located
        plt.plot(x0,b-y0,'ok')

        # drawing the limits of the PML domain
        plt.plot(np.array([delta,delta]),np.array([delta,b-delta]),'k')
        plt.plot(np.array([a-delta,a-delta]),np.array([delta,b-delta]),'k')
        plt.plot(np.array([delta,a-delta]),np.array([delta,delta]),'k')
        plt.plot(np.array([delta,a-delta]),np.array([b-delta,b-delta]),'k')

        # saving and showing the picture
        # plt.subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
        plt.savefig(example+'/vel_P_map.pdf')
        plt.show()


def sources_acoustic_1D(Ndt,T,degree,example,point):
    # this is a function draw at the moment

    RK_ref_points=np.load(str(example)+'/RK_ref_points_'+str(0))



    plt.plot(np.linspace(0,T,RK_ref_points.shape[0]),RK_ref_points[:,point])




    sol_rk2_points=np.load(str(example)+'/sol_rk2_points_'+str(0))
    sol_rk4_points=np.load(str(example)+'/sol_rk4_points_'+str(0))
    sol_2time_points=np.load(str(example)+'/sol_2time_points_'+str(0))

    plt.plot(np.linspace(0,T,RK_ref_points.shape[0]),sol_rk2_points[:,point])
    plt.plot(np.linspace(0,T,RK_ref_points.shape[0]),sol_rk4_points[:,point])
    plt.plot(np.linspace(0,T,RK_ref_points.shape[0]),sol_2time_points[:,point])


def snapshot_method(example,dx,delta,equ,ord,ind_source,Ndt_0,degree_index,method):
    # graph of the vvelocity medium with PML boundary

    # INPUT
    # example: identificator of an example (string)
    # dx: space discretization step size (float)
    # delta: PML thickness (float)
    # equ: type of equation used (string)

    # cheking if there exist the paste to save the results, and creating one if there is not
    if not os.path.isdir(example+'/'):
        os.mkdir(example)

    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)

    extent = [0,a, -b,0]

    if equ!='elastic':
        param=param.reshape((ny,nx),order='F')
        plt.imshow(param,extent=extent,cmap='gray')
        plt.xlabel('X Position [km]',fontsize=20)
        plt.ylabel('Depth [km]',fontsize=20)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
    else:
        # p-wave velocity
        velocity_p=np.sqrt((param[:,4]+2*param[:,2])*np.reciprocal(param[:,0]))
        velocity_p=velocity_p.reshape((ny,nx),order='F')
        plt.imshow(velocity_p,extent=extent,cmap='gray')
        plt.xlabel('X Position [km]',fontsize=20)
        plt.ylabel('Depth [km]',fontsize=20)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)

    if method=='RK7':
        sol=np.load(str(example)+'/RK_ref_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'_T23.npy')[:,Ndt_0]
    elif method=='RK2':
        sol=np.load(str(example)+'/sol_rk2_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')[:,Ndt_0]
    elif method=='RK4':
        sol=np.load(str(example)+'/sol_rk4_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')[:,Ndt_0]
    elif method=='2MS':
        sol=np.load(str(example)+'/sol_2time_equ_'+str(equ)+'_ord_'+ord+'_dx_'+str(dx)+'.npy')[:,Ndt_0]
    elif method=='FA':
        sol=np.load(str(example)+'/sol_faber_equ_'+str(equ)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(Ndt_0)+'_dx_'+str(dx)+'.npy')[degree_index,:]
    elif method=='RKHO':
        sol=np.load(str(example)+'/sol_rk_equ_'+str(equ)+'_ord_'+ord+'_Ndt_'+str(Ndt_0)+'_dx_'+str(dx)+'.npy')[degree_index,:]
    sol=sol.reshape((ny,nx),order='F')
    print(np.max(sol))
    print(np.min(sol))


    # construction of the wave snapshot at the final time instant
    fig=plt.imshow(sol,extent=extent,alpha=0.84,vmax=5.436284894142866e-06,vmin=-3.870688779584465e-06)
    plt.xlabel('X Position [km]',fontsize=20)
    plt.ylabel('Depth [km]',fontsize=20)
    cbar=plt.colorbar(fig)
    cbar.set_label('Displacement [km]',fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.yaxis.get_offset_text().set(size=13)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    # plt.plot(np.array([x0_pos*dx,x0_pos*dx]),np.array([0,8]),'k')
    # plt.gca().set_aspect(1.7, adjustable='box')
    plt.draw()
    plt.tight_layout()
    plt.savefig(example+'/snapshot_method_'+method+'_Ndt_'+str(Ndt_0)+'.pdf')
    plt.show()

# --------------------------------------------------------------------------------------------------------
# Graphics of convergence error and its dependence with the time step size, with the examples of the paper
# --------------------------------------------------------------------------------------------------------

# error_acoustic(Ndt=40,example='1D_heterogeneous_1a',dim=1,dx=0.002,c2_max=1.524**2,equ='scalar',ord='4',ind_source='H_amplified')
# error_acoustic(Ndt=29,example='2D_heterogeneous_2',dim=2,dx=0.04,c2_max=6**2,equ='scalar_dx2',ord='4',ind_source='H_amplified')
# error_acoustic(Ndt=40,example='2D_heterogeneous_3',dim=2,dx=0.03125,c2_max=((4.5+18)/0.25),equ='elastic',ord='8',ind_source='H_amplified')


# ------------------------------------------------------------------------
# Comparisson graphics of maximum time step size, \Delta t, and efficiency
# (functions graph_experiments and eff_graph_experiments, with the paper test cases)
# ------------------------------------------------------------------------

# example=np.array(['2D_homogeneous_0a','2D_homogeneous_0a','2D_heterogeneous_3a','2D_heterogeneous_3a'])
# ord=np.array(['4','4','4','4'])
# equ=np.array(['scalar','scalar_dx2','scalar','scalar_dx2'])
# c2_max=np.array([3,3,6,6])
# graph_experiments(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=30-3,dx=0.02,c2_max=c2_max,ind='Tests45_ord4',Ndt=29)
# eff_graph_experiments(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=30-3,dx=0.02,c2_max=c2_max,ind='Tests45_ord4',Ndt=29)
#
#
# example=np.array(['1D_homogeneous_0','1D_homogeneous_0','1D_heterogeneous_2','1D_heterogeneous_2'])
# ord=np.array(['8','8','8','8'])
# equ=np.array(['scalar','scalar_dx2','scalar','scalar_dx2'])
# c2_max=np.array([1.524,1.524,3.048,3.048])
# graph_experiments(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=35-3,dx=0.0025,c2_max=c2_max,ind='Tests13_ord8',Ndt=40)
# eff_graph_experiments(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=35-3,dx=0.0025,c2_max=c2_max,ind='Tests13_ord8',Ndt=40)
#
# example=np.array(['2D_heterogeneous_3','2D_heterogeneous_3'])
# ord=np.array(['4','8'])
# equ=np.array(['elastic','elastic'])
# c2_max=np.array([np.sqrt((4.5+18)/0.25),np.sqrt((4.5+18)/0.25)])
# graph_experiments(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=25-3,dx=0.03125,c2_max=c2_max,ind='Test7_ord48',Ndt=40)
# eff_graph_experiments(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=25-3,dx=0.03125,c2_max=c2_max,ind='Test7_ord48',Ndt=40)


# -----------------------------------------------------------
# Graphics of the examples' velocity field and source position
# -----------------------------------------------------------

# graph_velocities(example='2D_heterogeneous_3',dx=0.02,delta=0.8,equ='scalar')


# --------------------------------------------------
# spatial error and snapshot of the wave propagation
# --------------------------------------------------

# spatial_error(example='2D_heterogeneous_3',equ='scalar',ord='8',ind_source='H_amplified',dx=0.02,nx=400,ny=400,x0_pos=300,Ndt_0=11,dephs_y=8,degree=np.arange(11,19))
