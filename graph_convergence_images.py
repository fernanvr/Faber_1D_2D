from graph_convergence import *


# --------------------------------------------------------------------------------------------------------
# Figure 12: Graphics of convergence error and its dependence with the time step size, and the appendix for the elastic waves
# --------------------------------------------------------------------------------------------------------

# error_acoustic(Ndt=40,example='1D_heterogeneous_1a',dim=1,dx=0.002,c2_max=1.524**2,equ='scalar',ord='4',ind_source='H_amplified')
# error_acoustic(Ndt=29,example='2D_heterogeneous_2',dim=2,dx=0.04,c2_max=6**2,equ='scalar_dx2',ord='4',ind_source='H_amplified')
# error_acoustic(Ndt=40,example='2D_heterogeneous_3',dim=2,dx=0.03125,c2_max=((4.5+18)/0.25),equ='elastic',ord='8',ind_source='H_amplified')


# --------------------------------------------------------------------------------------------------------
# Figure 13: Comparisson graphics of maximum time step size, \Delta t, and efficiency (functions graph_experiments and eff_graph_experiments, with the paper test cases)
# --------------------------------------------------------------------------------------------------------

# example=np.array(['2D_homogeneous_0a','2D_homogeneous_0a','2D_heterogeneous_3a','2D_heterogeneous_3a'])
# ord=np.array(['4','4','4','4'])
# equ=np.array(['scalar','scalar_dx2','scalar','scalar_dx2'])
# c2_max=np.array([3,3,6,6])
# graph_experiments_dt_max(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=30-3,dx=0.02,c2_max=c2_max,ind='Tests45_ord4',Ndt=29)
# eff_graph_experiments(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=30-3,dx=0.02,c2_max=c2_max,ind='Tests45_ord4',Ndt=29)
#
#
# example=np.array(['1D_homogeneous_0','1D_homogeneous_0','1D_heterogeneous_2','1D_heterogeneous_2'])
# ord=np.array(['8','8','8','8'])
# equ=np.array(['scalar','scalar_dx2','scalar','scalar_dx2'])
# c2_max=np.array([1.524,1.524,3.048,3.048])
# graph_experiments_dt_max(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=35-3,dx=0.0025,c2_max=c2_max,ind='Tests13_ord8',Ndt=40)
# eff_graph_experiments(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=35-3,dx=0.0025,c2_max=c2_max,ind='Tests13_ord8',Ndt=40)
#
# example=np.array(['2D_heterogeneous_3','2D_heterogeneous_3'])
# ord=np.array(['4','8'])
# equ=np.array(['elastic','elastic'])
# c2_max=np.array([np.sqrt((4.5+18)/0.25),np.sqrt((4.5+18)/0.25)])
# graph_experiments_dt_max(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=25-3,dx=0.03125,c2_max=c2_max,ind='Test7_ord48',Ndt=40)
# eff_graph_experiments(example=example,ord=ord,equ=equ,ind_source="H_amplified",max_degree=25-3,dx=0.03125,c2_max=c2_max,ind='Test7_ord48',Ndt=40)


# -----------------------------------------------------------
# Graphics of the examples' velocity field and source position and spatial error and snapshot of the wave propagation
# -----------------------------------------------------------

# graph_velocities(example='2D_heterogeneous_3',dx=0.02,delta=0.8,equ='scalar')

# spatial_error(example='2D_heterogeneous_3',equ='scalar',ord='8',ind_source='H_amplified',dx=0.02,nx=400,ny=400,x0_pos=300,Ndt_0=11,dephs_y=8,degree=np.arange(11,19))