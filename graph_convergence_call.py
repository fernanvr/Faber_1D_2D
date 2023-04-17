import graph_convergence as gra_con
import numpy as np


# gra_con.snapshot_method(example='Marmousi_b',dx=0.04,equ='scalar_dx2',degree=10,method='2MS')
# gra_con.snapshot_method(example='SEG_EAGE_b',dx=0.04,equ='scalar_dx2',degree=10,method='2MS')
# gra_con.snapshot_method(example='piece_GdM_b',dx=0.04,equ='scalar_dx2',degree=10,method='2MS')

# gra_con.snapshot_method(example='2D_heterogeneous_3b',dx=0.01,equ='scalar_dx2',degree=10,method='RK4')
# gra_con.snapshot_method(example='2D_heterogeneous_3b',dx=0.005,equ='scalar_dx2',degree=10,method='RK7')
#
# gra_con.snapshot_method_error(example='2D_heterogeneous_3b',dx=0.01,equ='scalar_dx2',degree=10,method='RK4')


# gra_con.seismogram(example='SEG_EAGE_b',dim=2,dx=0.04,delta=0.8,equ='scalar_dx2',ord='8',ind_source='H_amplified',Ndt_0=1,degree=10,method='2MS',free_surf=1,T=6)


# --------------------------------------------------------------------------------
# Maximum Delta t and efficiency graph for different methods
# --------------------------------------------------------------------------------
example='2D_heterogeneous_3b'
methods=np.array(['FA','HORK','KRY','RK7','RK2','RK4','2MS'])
equ='scalar_dx2'
free_surf=1
ord='8'
Ndt=np.arange(1,51)
degree=np.arange(3,37)
dx=0.01
gra_con.graph_methods_dt_max(example,methods,equ,free_surf,ord,Ndt,degree,dx,tol=1e-7)
gra_con.graph_methods_eff(example,methods,equ,free_surf,ord,Ndt,degree,dx,tol=1e-7)