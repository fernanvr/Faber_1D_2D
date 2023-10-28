import matplotlib.pyplot as plt
import numpy as np
import auxiliary_functions as aux_fun
import Methods as meth
import os
from matplotlib.ticker import FormatStrFormatter
import dispersion_tfourier as dis
import math


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def error_acoustic(Ndt,example,dim,dx,c2_max,equ,ord,ind_source):
    # graphic of Faber polynomials error, for different degrees (x-axis time-step, y-axis error, curves polynomial degrees)

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

    # plt.rcParams["figure.figsize"] = [8.5,5]
    ax=plt.gca()
    for i in range(len(degree)):
        # plt.plot(dt,err_sol_faber[i,:],label='FA'+str(degree[i]),linewidth=3,marker='D')
        lin,=ax.plot(dt,err_sol_faber[i,:],linewidth=3)
        ax.scatter(dt,err_sol_faber[i,:], linewidth=3,marker='D',alpha=0.5)
        ax.plot([],[],label='FA'+str(degree[i]),color=lin.get_color(),linewidth=3,marker='D')

    plt.xlabel('time-step $\Delta t$',size='24')
    plt.ylabel('$||Err||_2$',size='24')
    plt.ticklabel_format(style="sci", scilimits=(0,0))
    plt.gca().xaxis.get_offset_text().set_fontsize(17)
    plt.yscale('log')
    # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.yticks(pow(10.0,np.arange(0, -17, step=-2)),fontsize=17)
    plt.subplots_adjust(left=0.22, bottom=0.17, right=0.9, top=0.9)
    # plt.legend(fontsize=20,bbox_to_anchor=(1.1, 0.5), loc="center left", borderaxespad=0)
    plt.ylim(pow(10,-16),30)
    plt.savefig(str(example)+'/faber_equ_'+equ+'_'+str(example)+'_ord_'+ord+'_source_'+ind_source+'.pdf')
    plt.savefig('Convergence_images/faber_equ_'+equ+'_'+str(example)+'_ord_'+ord+'_source_'+ind_source+'.pdf')
    plt.show()


def graph_experiments_dt_max(example,ord,equ,ind_source,max_degree,dx,c2_max,ind,Ndt):
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
    plt.savefig('Convergence_images/faber_experiments_'+ind+'.pdf')
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
    plt.savefig('Convergence_images/faber_experiments_eff_'+ind+'.pdf')
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
    plt.savefig('Convergence_images/wave_propagation_example0.pdf')
    plt.show()

    # calculating and plotting the error
    depth=np.linspace(0,dephs_y,ny)
    faber=np.load(str(example)+'/sol_faber_equ_'+str(equ)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(Ndt_0)+'_dx_'+str(dx)+'.npy')
    for i in range(len(degree)):
        aux=faber[degree[i],:]
        aux=aux.reshape((ny,nx),order='F')
        plt.plot(depth,np.abs(RK_ref[:,x0_pos]-aux[:,x0_pos]),linestyle='-',label='FA'+str(degree[i]),linewidth=2)

    # graph parameters and figure saving
    plt.xlabel('Depth [km]',size='26')
    plt.ylabel('Error ($||\cdot||_{\infty}$)',size='26')
    plt.yscale('log')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(fontsize=20)
    plt.ylim(pow(10,-18))
    plt.tight_layout()
    # plt.subplots_adjust(left=0.18, bottom=0.15, right=0.9, top=0.9)
    plt.savefig(example+'/error_spatial_cut.pdf')
    plt.savefig('Convergence_images/error_spatial_cut.pdf')
    plt.show()


def sol_ref_load(example,equ,free_surf,ord,dx,delta,no_PML,time_space):
    # Function to load the reference solution in dependence of the type of error calculated and the physycal domain restriction

    if time_space=='space':    # reference solutions using the snapshot information at a given time
        sol_ref=np.load(example+'/RK_ref_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_1_dx_'+str(dx/2)+'.npy')
        a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx/2,delta,equ)
        sol_ref=sol_ref.reshape((ny,nx),order='F')
        sol_ref=sol_ref[::2,:-1:2]
        if no_PML:
            a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)
            cut_PML=int(delta/dx)
            if free_surf==1:
                sol_ref=sol_ref[:-cut_PML,cut_PML:-cut_PML]
            else:
                sol_ref=sol_ref[cut_PML:-cut_PML,cut_PML:-cut_PML]
        else:
            sol_ref=sol_ref.flatten('F')
    else:        # reference solution using the seismogram data
        sol_ref=np.load(example+'/RK_ref_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_1_dx_'+str(dx/2)+'_points.npy')
        sol_ref=sol_ref[::2,::2]
        if no_PML:
            cut_PML=int(delta/dx)
            sol_ref=sol_ref[:,cut_PML:-cut_PML]

    return sol_ref


def sol_load(example,meth_ind,meth_label,equ,free_surf,ord,dx,delta,no_PML,time_space,Ndt,nx,ny,degree=0,ind_source="H_amplified"):
    # Function to load the approximated solution in dependence of the type of error calculated and the physycal domain restriction

    if time_space=='space':    # approximated solution using the snapshot information at a given time
        if meth_ind<10:
            sol=np.load(example+'/'+meth_label+'_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_'+str(Ndt)+'_dx_'+str(dx)+'.npy')
        else:
            if meth_label=='sol_faber':
                sol=np.load(example+'/'+meth_label+'_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(Ndt)+'_degree_'+str(degree)+'_dx_'+str(dx)+'.npy')
            else:
                sol=np.load(example+'/'+meth_label+'_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_'+str(Ndt)+'_degree_'+str(degree)+'_dx_'+str(dx)+'.npy')
        if no_PML:
            sol=sol.reshape((ny,nx),order='F')
            cut_PML=int(delta/dx)
            if free_surf==1:
                sol=sol[:-cut_PML,cut_PML:-cut_PML]
            else:
                sol=sol[cut_PML:-cut_PML,cut_PML:-cut_PML]

    else:        # approximated solution using the seismogram data
        if meth_ind<10:
            sol=np.load(example+'/'+meth_label+'_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_'+str(Ndt)+'_dx_'+str(dx)+'_points.npy')
        else:
            if meth_label=='sol_faber':
                sol=np.load(example+'/'+meth_label+'_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_'+ind_source+'_points_Ndt_'+str(Ndt)+'_degree_'+str(degree)+'_dx_'+str(dx)+'.npy')
            else:
                sol=np.load(example+'/'+meth_label+'_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_points_Ndt_'+str(Ndt)+'_degree_'+str(degree)+'_dx_'+str(dx)+'.npy')
        if no_PML:
            cut_PML=int(delta/dx)
            sol=sol[:,cut_PML:-cut_PML]

    return sol


def error_norm(sol,sol_ref,time_space,dim,dx,dt,Ndt=1):
    if time_space=='space':
        return np.sqrt(np.sum(pow(sol-sol_ref,2))*dx**dim)
    else:
        sol_ref=sol_ref[::Ndt,:]
        if len(sol)>len(sol_ref):
            sol=sol[:-1,:]
        return np.sqrt(np.sum(pow(sol-sol_ref,2))*dx*2*dt)


def spatial_discr_error(example,methods,degree,time_space,free_surf=1,equ='scalar_dx2',dx=0.01,ord='8',dim=2,delta=0.8,no_PML=True,ind_source='H_amplified',fig_ind=''):
    # Error of each method using the minimum time-step consider to account for the error produced by the spatial discretization

    sol_ref=sol_ref_load(example,equ,free_surf,ord,dx,delta,no_PML,time_space)
    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)

    color=np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf'])
    names_prov=dis.method_label_graph(methods)
    marker=np.array(['D','o','s','v','P','H','D'])
    line=np.array(['-','--','-.'])
    count_high_order=0
    plt.rcParams["figure.figsize"] = [7,5]
    ax=plt.gca()
    for i in range(len(methods)):
        meth_ind,meth_label=dis.method_label(methods[i])
        if meth_ind<10:
            sol=sol_load(example,meth_ind,meth_label,equ,free_surf,ord,dx,delta,no_PML,time_space,1,nx,ny)
            error=error_norm(sol,sol_ref,time_space,dim,dx,dt)
            ax.scatter(meth_ind,error, linewidth=2,marker=marker[i],alpha=0.5,color=color[i],s=60,zorder=5*i)
            ax.scatter(meth_ind,error,marker=marker[i],color=color[i],s=30,zorder=5*i)
            ax.scatter([],[],label=names_prov[i],color=color[i],linewidth=2,marker=marker[i],zorder=5*i)
        else:
            error=np.zeros(len(degree))+2
            for j in range(len(degree)):
                try:
                    sol=sol_load(example,meth_ind,meth_label,equ,free_surf,ord,dx,delta,no_PML,time_space,1,nx,ny,degree=degree[j],ind_source=ind_source)
                except:
                    continue
                error[j]=error_norm(sol,sol_ref,time_space,dim,dx,dt)
            error[error>1]=float('nan')
            lin,=ax.plot(degree,error,linewidth=2,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
            ax.scatter(degree,error, linewidth=2,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
            ax.plot([],[],label=names_prov[i],color=color[i],linewidth=2,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
            count_high_order+=1

    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    # plt.legend(fontsize=20)
    plt.ylabel('$\Delta t_{max}$',fontsize=20)
    plt.xlabel('# Stages',fontsize=20)
    ax.yaxis.get_offset_text().set(size=16)
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.9, top=0.95)
    plt.savefig(str(example)+'_images/'+example+'_methods_spatial_error'+fig_ind+'_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_dx_'+str(dx)+'_'+time_space+'.pdf')
    # plt.show()
    plt.clf()


def graph_methods_dt_max(example,methods,Ndt,degree,time_space,free_surf=1,equ='scalar_dx2',dx=0.01,ord='8',dim=2,delta=0.8,no_PML=True,ind_source='H_amplified',tol=1e-5,fig_ind=''):
    # graphics of the maximum \Delta t allowed by methods and approximation degrees at a given numerical experiment

    # INPUT:
    # methods: the different methods we will consider (vector string)
    # example: selection of the wave equation parameters (string)
    # ord: spatial discretization order ('4','8')
    # equ: equation formulation (scalar, scalar_dx2, elastic)
    # ind_source: indicator of the wave equations' source term treatment ('H_amplified', 'FA_ricker')
    # max_degree: maximum polynomial degree used to graphic (int)
    # dx: spatial discretization step size (float)
    # ind: string used to put in the figure name, for saving
    # Ndt: amount of time-step sizes used to compute the solutions (int)

    sol_ref=sol_ref_load(example,equ,free_surf,ord,dx,delta,no_PML,time_space)
    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)

    color=np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf'])
    # names_prov=np.array(['FA','HORK','KRY','RK9-7','RK3-2','RK4-4','Leap-frog'])
    names_prov=dis.method_label_graph(methods)
    marker=np.array(['D','o','s','v','P','H','D'])
    line=np.array(['-','--','-.'])
    count_high_order=0
    plt.rcParams["figure.figsize"] = [9,6]
    ax=plt.gca()
    for i in range(len(methods)):
        meth_ind,meth_label=dis.method_label(methods[i])
        if meth_ind<10:
            max_dt=0
            for j in range(len(Ndt)):
                sol=sol_load(example,meth_ind,meth_label,equ,free_surf,ord,dx,delta,no_PML,time_space,Ndt[j],nx,ny)
                error=error_norm(sol,sol_ref,time_space,dim,dx,dt,Ndt=Ndt[j])
                if error>tol or math.isnan(error):
                    break
                else:
                    max_dt=2*dt*Ndt[j]
            ax.scatter(meth_ind,max_dt, linewidth=2,marker=marker[i],alpha=0.5,color=color[i],s=60,zorder=5*i)
            ax.scatter(meth_ind,max_dt,marker=marker[i],color=color[i],s=30,zorder=5*i)
            ax.scatter([],[],label=names_prov[i],color=color[i],linewidth=2,marker=marker[i],zorder=5*i)
        else:
            max_dt=np.zeros(len(degree))
            for j in range(len(degree)):
                cont_miss=0   # counter of missing files
                for k in range(len(Ndt)):
                    try:
                        sol=sol_load(example,meth_ind,meth_label,equ,free_surf,ord,dx,delta,no_PML,time_space,Ndt[k],nx,ny,degree=degree[j],ind_source=ind_source)
                    except:
                        cont_miss+=1
                        if cont_miss>20:
                            max_dt[j]=float('nan')
                            break
                        continue
                    error=error_norm(sol,sol_ref,time_space,dim,dx,dt,Ndt[k])
                    if error>tol or math.isnan(error):
                        break
                    else:
                        max_dt[j]=2*dt*Ndt[k]
            max_dt[max_dt==0]=float('nan')
            lin,=ax.plot(degree,max_dt,linewidth=2,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
            ax.scatter(degree,max_dt, linewidth=2,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
            ax.plot([],[],label=names_prov[i],color=color[i],linewidth=2,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
            count_high_order+=1

    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.legend(fontsize=20)
    plt.ylabel('$\Delta t_{max}$',fontsize=27)
    plt.xlabel('# Stages',fontsize=24)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.subplots_adjust(left=0.2, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(str(example)+'_images/'+example+'_methods_max_dt'+fig_ind+'_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_dx_'+str(dx)+'_'+time_space+'.pdf')
    plt.show()
    # plt.clf()


def graph_methods_eff(example,methods,Ndt,degree,time_space,equ='scalar_dx2',free_surf=1,dx=0.01,ord='8',dim=2,delta=0.8,no_PML=True,ind_source='H_amplified',tol=1e-5,fig_ind=''):
    # graphics of the maximum \Delta t allowed by methods and approximation degrees at a given numerical experiment

    # INPUT:
    # methods: the different methods we will consider (vector string)
    # example: selection of the wave equation parameters (string)
    # ord: spatial discretization order ('4','8')
    # equ: equation formulation (scalar, scalar_dx2, elastic)
    # ind_source: indicator of the wave equations' source term treatment ('H_amplified', 'FA_ricker')
    # max_degree: maximum polynomial degree used to graphic (int)
    # dx: spatial discretization step size (float)
    # ind: string used to put in the figure name, for saving
    # Ndt: amount of time-step sizes used to compute the solutions (int)

    sol_ref=sol_ref_load(example,equ,free_surf,ord,dx,delta,no_PML,time_space)
    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)

    color=np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf'])
    # names_prov=np.array(['FA','HORK','KRY','RK9-7','RK3-2','RK4-4','Leap frog'])
    names_prov=dis.method_label_graph(methods)
    marker=np.array(['D','o','s','v','P','H','D'])
    line=np.array(['-','--','-.'])
    count_high_order=0
    plt.rcParams["figure.figsize"] = [7.5,5]
    ax=plt.gca()
    for i in range(len(methods)):
        meth_ind,meth_label=dis.method_label(methods[i])
        if meth_ind<10:
            max_dt=0
            for j in range(len(Ndt)):
                sol=sol_load(example,meth_ind,meth_label,equ,free_surf,ord,dx,delta,no_PML,time_space,Ndt[j],nx,ny)
                error=error_norm(sol,sol_ref,time_space,dim,dx,dt,Ndt=Ndt[j])
                if error>tol or math.isnan(error):
                    break
                else:
                    max_dt=2*dt*Ndt[j]
            ax.scatter(meth_ind,np.log10(meth_ind/max_dt), linewidth=2,marker=marker[i],alpha=0.5,color=color[i],s=60,zorder=5*i)
            ax.scatter(meth_ind,np.log10(meth_ind/max_dt),marker=marker[i],color=color[i],s=30,zorder=5*i)
            # ax.scatter([],[],label=methods[i],color=color[i],linewidth=2,marker=marker[i],zorder=5*i)
            ax.scatter([],[],label=names_prov[i],color=color[i],linewidth=2,marker=marker[i],zorder=5*i)
        else:
            max_dt=np.zeros(len(degree))
            for j in range(len(degree)):
                cont_miss=0   # counting missing files
                for k in range(len(Ndt)):
                    try:
                        sol=sol_load(example,meth_ind,meth_label,equ,free_surf,ord,dx,delta,no_PML,time_space,Ndt[k],nx,ny,degree=degree[j],ind_source=ind_source)
                    except:
                        cont_miss+=1
                        if cont_miss>20:
                            max_dt[j]=float('nan')
                            break
                        continue
                    error=error_norm(sol,sol_ref,time_space,dim,dx,dt,Ndt=Ndt[k])
                    if error>tol or math.isnan(error):
                        break
                    else:
                        max_dt[j]=2*dt*Ndt[k]
            max_dt[max_dt==0]=float('nan')
            aux=np.log10(degree/max_dt)
            lin,=ax.plot(degree[aux>=0],aux[aux>=0],linewidth=2,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
            ax.scatter(degree[aux>=0],aux[aux>=0], linewidth=2,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
            # ax.scatter(degree,degree/max_dt,marker=marker[i],color=color1[i],s=5)
            # ax.plot([],[],label=methods[i],color=color[i],linewidth=2,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
            ax.plot([],[],label=names_prov[i],color=color[i],linewidth=2,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
            count_high_order+=1

    y_ticks_position=np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],7)
    y_ticks_labels=np.char.add([r"%.1f$\cdot$" % pow(10,math.modf(x)[0]) for x in y_ticks_position],[r"$10^{%.0f}$" % int(math.modf(x)[1]) for x in y_ticks_position])
    ax.set_yticks(y_ticks_position)
    ax.set_yticklabels(y_ticks_labels)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    plt.ylabel(r'$N^{\Delta t}_{op}$',fontsize=24)
    plt.xlabel('# Stages',fontsize=22)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.subplots_adjust(left=0.24, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(str(example)+'_images/'+example+'_methods_eff'+fig_ind+'_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_dx_'+str(dx)+'_'+time_space+'.pdf')
    # plt.show()
    plt.clf()


def graph_methods_mem(example,methods,Ndt,degree,T,time_space,equ='scalar_dx2',free_surf=1,dx=0.01,ord='8',dim=2,delta=0.8,no_PML=True,ind_source='H_amplified',tol=1e-5,fig_ind=''):
    # graphics of the maximum \Delta t allowed by methods and approximation degrees at a given numerical experiment

    # INPUT:
    # methods: the different methods we will consider (vector string)
    # example: selection of the wave equation parameters (string)
    # ord: spatial discretization order ('4','8')
    # equ: equation formulation (scalar, scalar_dx2, elastic)
    # ind_source: indicator of the wave equations' source term treatment ('H_amplified', 'FA_ricker')
    # max_degree: maximum polynomial degree used to graphic (int)
    # dx: spatial discretization step size (float)
    # ind: string used to put in the figure name, for saving
    # Ndt: amount of time-step sizes used to compute the solutions (int)

    sol_ref=sol_ref_load(example,equ,free_surf,ord,dx,delta,no_PML,time_space)
    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)

    color=np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2','#7f7f7f', '#bcbd22', '#17becf'])
    # names_prov=np.array(['FA','HORK','KRY','RK9-7','RK3-2','RK4-4','Leap frog'])
    names_prov=dis.method_label_graph(methods)
    marker=np.array(['D','o','s','v','P','H','D'])
    line=np.array(['-','--','-.'])
    count_high_order=0
    plt.rcParams["figure.figsize"] = [7.5,5]
    ax=plt.gca()
    for i in range(len(methods)):
        meth_ind,meth_label=dis.method_label(methods[i])
        if meth_ind<10:
            max_dt=0
            for j in range(len(Ndt)):
                sol=sol_load(example,meth_ind,meth_label,equ,free_surf,ord,dx,delta,no_PML,time_space,Ndt[j],nx,ny)
                error=error_norm(sol,sol_ref,time_space,dim,dx,dt,Ndt=Ndt[j])
                if error>tol or math.isnan(error):
                    break
                else:
                    max_dt=2*dt*Ndt[j]
            ax.scatter(meth_ind,np.log10(T/max_dt), linewidth=2,marker=marker[i],alpha=0.5,color=color[i],s=60,zorder=5*i)
            ax.scatter(meth_ind,np.log10(T/max_dt),marker=marker[i],color=color[i],s=30,zorder=5*i)
            # ax.scatter([],[],label=methods[i],color=color[i],linewidth=2,marker=marker[i],zorder=5*i)
            ax.scatter([],[],label=names_prov[i],color=color[i],linewidth=2,marker=marker[i],zorder=5*i)
        else:
            max_dt=np.zeros(len(degree))
            for j in range(len(degree)):
                cont_miss=0     # counting missing files
                for k in range(len(Ndt)):
                    try:
                        sol=sol_load(example,meth_ind,meth_label,equ,free_surf,ord,dx,delta,no_PML,time_space,Ndt[k],nx,ny,degree=degree[j],ind_source=ind_source)
                    except:
                        cont_miss+=1
                        if cont_miss>20:
                            max_dt[j]=float('nan')
                            break
                        continue
                    error=error_norm(sol,sol_ref,time_space,dim,dx,dt,Ndt=Ndt[k])
                    if error>tol or math.isnan(error):
                        break
                    else:
                        max_dt[j]=2*dt*Ndt[k]
            max_dt[max_dt==0]=float('nan')
            aux=np.log10(T/max_dt)
            lin,=ax.plot(degree[aux>=0],aux[aux>=0],linewidth=2,linestyle=line[count_high_order],alpha=0.9,color=color[i],zorder=5*i)
            ax.scatter(degree[aux>=0],aux[aux>=0], linewidth=2,marker=marker[i],alpha=0.5,color=color[i],zorder=5*i)
            # ax.scatter(degree,degree/max_dt,marker=marker[i],color=color1[i],s=5)
            # ax.plot([],[],label=methods[i],color=color[i],linewidth=2,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
            ax.plot([],[],label=names_prov[i],color=color[i],linewidth=2,marker=marker[i],linestyle=line[count_high_order],zorder=5*i)
            count_high_order+=1

    y_ticks_position=np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],7)
    y_ticks_labels=np.char.add([r"%.1f$\cdot$" % pow(10,math.modf(x)[0]) for x in y_ticks_position],[r"$10^{%.0f}$" % int(math.modf(x)[1]) for x in y_ticks_position])
    ax.set_yticks(y_ticks_position)
    ax.set_yticklabels(y_ticks_labels)
    plt.xticks(fontsize=21)
    plt.yticks(fontsize=21)
    # plt.legend(fontsize=20)
    # ax.set_yscale('log')
    plt.ylabel(r'$N^{\Delta t}_{mem}$',fontsize=24)
    plt.xlabel('# Stages',fontsize=22)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.subplots_adjust(left=0.24, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(str(example)+'_images/'+example+'_methods_memory'+fig_ind+'_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_dx_'+str(dx)+'_'+time_space+'.pdf')
    # plt.show()
    plt.clf()


def graph_methods_error(example,dim,equ,delta,ind_source,Ndt,degree,dx=0.01,ord='8'):

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


def graph_PML(delta,X,Y,example,free_surf):
    # drawing the limits of the PML domain
    if free_surf==0:
        plt.plot(np.array([delta,delta]),np.array([delta,np.max(Y)-delta]),'k',linestyle='--')
        plt.plot(np.array([np.max(X)-delta,np.max(X)-delta]),np.array([delta,np.max(Y)-delta]),'k',linestyle='--')
        plt.plot(np.array([delta,np.max(X)-delta]),np.array([delta,delta]),'k',linestyle='--')
        plt.plot(np.array([delta,np.max(X)-delta]),np.array([np.max(Y)-delta,np.max(Y)-delta]),'k',linestyle='--')
    else:
        if '2D_heterogeneous_' in example:
            plt.plot(np.array([delta,delta]),np.array([np.min(Y),np.max(Y)-delta]),'k',linestyle='--')
            plt.plot(np.array([np.max(X)-delta,np.max(X)-delta]),np.array([np.min(Y),np.max(Y)-delta]),'k',linestyle='--')
            plt.plot(np.array([delta,np.max(X)-delta]),np.array([np.max(Y)-delta,np.max(Y)-delta]),'k',linestyle='--')
        else:
            plt.plot(np.array([np.min(X)+delta,np.min(X)+delta]),np.array([np.min(Y),np.max(Y)-delta]),'k',linestyle='--')
            plt.plot(np.array([np.max(X)-delta,np.max(X)-delta]),np.array([np.min(Y),np.max(Y)-delta]),'k',linestyle='--')
            plt.plot(np.array([np.min(X)+delta,np.max(X)-delta]),np.array([np.max(Y)-delta,np.max(Y)-delta]),'k',linestyle='--')


def graph_velocities(example,dx=0.01,delta=0.8,equ='scalar',free_surface=1):
    # graph of the velocity medium with PML boundary

    # INPUT
    # example: identificator of an example (string)
    # dx: space discretization step size (float)
    # delta: PML thickness (float)
    # equ: type of equation used (string)

    # cheking if there exist the paste to save the results, and creating one if there is not
    if not os.path.isdir(example+'/'):
        os.mkdir(example)

    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)

    if '2D_heterogeneous_' not in example:
        Y=-Y
    extent = [np.min(X),np.max(X),np.max(Y), np.min(Y)]
    print('extent',extent)

    if equ!='elastic':
        param=param.reshape((ny,nx),order='F')
        fig=plt.imshow(param,extent=extent, aspect='auto')
        plt.xlabel('X Position [km]',fontsize=20)
        plt.ylabel('Depth [km]',fontsize=20)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
        cbar=plt.colorbar(fig)
        cbar.set_label('Velocity [km/s]',fontsize=20)
        cbar.ax.tick_params(labelsize=15)
        plt.grid()

        print('x0',x0)
        print('y0',y0)
        # drawing the point where the source term is located
        if '2D_heterogeneous_' in example:
            plt.plot(x0,np.max(Y)-y0,'ok')
        else:
            plt.plot(x0,-y0,'ok')

        graph_PML(delta,X,Y,example,free_surface)

        # saving and showing the picture
        plt.tight_layout()
        plt.savefig(example+'_images/'+example+'_vel_map.pdf')
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

        print('x0',x0)
        print('y0',y0)
        # drawing the point where the source term is located
        plt.plot(x0,b-y0,'ok')

        # drawing the limits of the PML domain
        plt.plot(np.array([delta,delta]),np.array([delta,b-delta]),'k')
        plt.plot(np.array([a-delta,a-delta]),np.array([delta,b-delta]),'k')
        plt.plot(np.array([delta,a-delta]),np.array([delta,delta]),'k')
        plt.plot(np.array([delta,a-delta]),np.array([b-delta,b-delta]),'k')

        # saving and showing the picture
        plt.savefig(example+'_images/vel_P_map.pdf')
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
        plt.savefig(example+'_images/vel_P_map.pdf')
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


def graph_physical(sol,extent,delta,dx,free_surf,cmap='viridis'):
    pml_thick=int(delta/dx)
    if free_surf==1:
        plt.imshow(sol[pml_thick:-pml_thick,:-pml_thick],extent=extent,alpha=0.85,aspect='auto',cmap=cmap)
    else:
        plt.imshow(sol[pml_thick:-pml_thick,pml_thick:-pml_thick],extent=extent,alpha=0.85,aspect='auto',cmap=cmap)


def snapshot_method(example,equ,method,degree=0,dx=0.01,ord='8',free_surf=1,Ndt_0=1,delta=0.8,ind_source='H_amplified',snapshot_type='pml'):
    # graph of the velocity medium with PML boundary

    # INPUT
    # example: identificator of an example (string)
    # dx: space discretization step size (float)
    # delta: PML thickness (float)
    # equ: type of equation used (string)

    # cheking if there exist the paste to save the results, and creating one if there is not
    if not os.path.isdir(example+'/'):
        os.mkdir(example)

    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)

    if '2D_heterogeneous_' not in example:
        Y=-Y
    if snapshot_type!='physical':
        extent = [np.min(X),np.max(X),np.max(Y), np.min(Y)]
    else:
        if free_surf==0:
            extent = [np.min(X)+delta,np.max(X)-delta,np.max(Y)-delta, np.min(Y)+delta]
        else:
            extent = [np.min(X)+delta,np.max(X)-delta,np.max(Y)-delta, np.min(Y)]

    # background velocity field
    if equ!='elastic':
        param=param.reshape((ny,nx),order='F')
        if snapshot_type=='physical':
            graph_physical(np.sqrt(param),extent,delta,dx,free_surf,cmap='gray')
        else:
            plt.imshow(np.sqrt(param),extent=extent,cmap='gray',aspect='auto')
        plt.xlabel('X Position [km]',fontsize=20)
        plt.ylabel('Depth [km]',fontsize=20)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
    else:
        # p-wave velocity
        velocity_p=np.sqrt((param[:,4]+2*param[:,2])*np.reciprocal(param[:,0]))
        velocity_p=velocity_p.reshape((ny,nx),order='F')
        plt.imshow(velocity_p,extent=extent,cmap='gray',aspect='auto')
        plt.xlabel('X Position [km]',fontsize=20)
        plt.ylabel('Depth [km]',fontsize=20)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)

    # loading the corresponding solution
    meth_ind,meth_label=dis.method_label(method)
    if meth_ind<10:
        sol=np.load(str(example)+'/'+meth_label+'_equ_'+str(equ)+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_'+str(Ndt_0)+'_dx_'+str(dx)+'.npy')
    else:
        if method=='FA':
            sol=np.load(str(example)+'/sol_faber_equ_'+str(equ)+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(Ndt_0)+'_degree_'+str(degree)+'_dx_'+str(dx)+'.npy')
        else:
            sol=np.load(str(example)+'/'+meth_label+'_equ_'+str(equ)+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_'+str(Ndt_0)+'_degree_'+str(degree)+'_dx_'+str(dx)+'.npy')

    sol=sol.reshape((ny,nx),order='F')
    print(np.max(sol))
    print(np.min(sol))

    # construction of the wave snapshot at the final time instant
    # fig=plt.imshow(sol,extent=extent,alpha=0.85,aspect='auto')# ,vmin=-5*pow(10,-6),vmax=5*pow(10,-6)
    if snapshot_type=='physical':
        graph_physical(sol,extent,delta,dx,free_surf,cmap='viridis')
    else:
        plt.imshow(sol,extent=extent,alpha=0.85,aspect='auto')
    if snapshot_type == 'pml':
        graph_PML(delta,X,Y,example,free_surf)

    plt.xlabel('X Position [km]',fontsize=20)
    plt.ylabel('Depth [km]',fontsize=20)
    # cbar=plt.colorbar(fig)
    cbar=plt.colorbar()
    cbar.set_label('Displacement [km]',fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.yaxis.get_offset_text().set(size=15)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    # plt.plot(np.array([x0_pos*dx,x0_pos*dx]),np.array([0,8]),'k')
    # plt.gca().set_aspect(1.7, adjustable='box')
    plt.draw()
    plt.tight_layout()
    plt.savefig(example+'_images/'+example+'_snapshot_method_'+method+'_free_surf'+str(free_surf)+'_Ndt_'+str(Ndt_0)+'_dx_'+str(dx)+'.pdf')
    plt.show()


def snapshot_method_error(example,equ,degree,method,dx=0.01,ord='8',free_surf=1,Ndt_0=1,delta=0.8,ind_source='H_amplified'):
    # graph of the velocity medium with PML boundary

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

    # background velocity field
    if equ!='elastic':
        param=param.reshape((ny,nx),order='F')
        plt.imshow(np.sqrt(param),extent=extent,cmap='gray',aspect='auto')
        plt.xlabel('X Position [km]',fontsize=20)
        plt.ylabel('Depth [km]',fontsize=20)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)
    else:
        # p-wave velocity
        velocity_p=np.sqrt((param[:,4]+2*param[:,2])*np.reciprocal(param[:,0]))
        velocity_p=velocity_p.reshape((ny,nx),order='F')
        plt.imshow(velocity_p,extent=extent,cmap='gray',aspect='auto')
        plt.xlabel('X Position [km]',fontsize=20)
        plt.ylabel('Depth [km]',fontsize=20)
        plt.xticks(fontsize=19)
        plt.yticks(fontsize=19)

    # loading the corresponding solution
    meth_ind,meth_label=dis.method_label(method)
    if meth_ind<10:
        sol=np.load(str(example)+'/'+meth_label+'_equ_'+str(equ)+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_'+str(Ndt_0)+'_dx_'+str(dx)+'.npy')
    else:
        if method=='FA':
            sol=np.load(str(example)+'/sol_faber_equ_'+str(equ)+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_'+ind_source+'_Ndt_'+str(Ndt_0)+'_degree_'+str(degree)+'_dx_'+str(dx)+'.npy')
        else:
            sol=np.load(str(example)+'/'+meth_label+'_equ_'+str(equ)+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_'+str(Ndt_0)+'_degree_'+str(degree)+'_dx_'+str(dx)+'.npy')

    sol=sol.reshape((ny,nx),order='F')

    sol_ref=np.load(example+'/RK_ref_equ_'+equ+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_1_dx_'+str(dx/2)+'.npy')
    sol_ref=sol_ref.reshape((2*ny,2*nx),order='F')
    sol_ref=sol_ref[1::2,:-1:2]

    # construction of the wave snapshot at the final time instant
    fig=plt.imshow(np.abs(sol-sol_ref),extent=extent,alpha=0.84,aspect='auto')# ,vmin=-5*pow(10,-6),vmax=5*pow(10,-6)
    plt.xlabel('X Position [km]',fontsize=20)
    plt.ylabel('Depth [km]',fontsize=20)
    cbar=plt.colorbar(fig)
    cbar.set_label('Displacement [km]',fontsize=20)
    cbar.ax.tick_params(labelsize=15)
    cbar.ax.yaxis.get_offset_text().set(size=15)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    # plt.plot(np.array([x0_pos*dx,x0_pos*dx]),np.array([0,8]),'k')
    # plt.gca().set_aspect(1.7, adjustable='box')
    plt.draw()
    plt.tight_layout()
    plt.savefig(example+'_images/snapshot_method_'+method+'free_surf'+str(free_surf)+'_Ndt_'+str(Ndt_0)+'_dx_'+str(dx)+'.pdf')
    plt.show()


def seismogram(example,delta,T,dim='2',no_PML=True,equ='scalar_dx2',ord='8',ind_source="H_amplified",Ndt_0=1,degree=0,method='RK7',free_surf=1,dx=0.01):
    # graph of the seismogram

    # INPUT
    # example: identification of the example (string)
    # dx: space discretization step size (float)
    # delta: PML thickness (float)
    # equ: type of equation used (string)

    # cheking if there exist the paste to save the results, and creating one if there is not
    if not os.path.isdir(example+'/'):
        os.mkdir(example)

    a,b,nx,ny,X,Y,param,dt,x0,y0=aux_fun.domain_examples(example,dx,delta,equ)

    if no_PML:
        extent = [np.min(X)+delta,np.max(X)-delta, T,0]
    else:
        extent = [np.min(X),np.max(X), T,0]

    meth_ind,meth_label=dis.method_label(method)
    sol=sol_load(example,meth_ind,meth_label,equ,free_surf,ord,dx,delta,no_PML,'time',Ndt_0,nx,ny)

    # construction of the wave snapshot at the final time instant
    fig=plt.imshow(pow(np.abs(sol),1/3)*np.sign(sol),extent=extent,cmap='gray',aspect='auto')# ,vmin=-5*pow(10,-6),vmax=5*pow(10,-6)
    plt.xlabel('X Position [km]',fontsize=20)
    plt.ylabel('Time [s]',fontsize=20)
    # cbar=plt.colorbar(fig)
    # cbar.set_label('Displacement [km]',fontsize=20)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.ax.yaxis.get_offset_text().set(size=13)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.draw()
    plt.tight_layout()
    plt.savefig(example+'_images/'+example+'_seismogram_method_'+method+'_free_surf_'+str(free_surf)+'_Ndt_'+str(Ndt_0)+'.pdf')
    plt.show()
