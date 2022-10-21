import sys
import Methods as meth
import numpy as np


def acoustic(dx,equ,dim,delta,beta0,ord,T,Ndt,degree,example,method,ind_source):
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

    # Model parameters
    nx,ny,X,Y,param,f,param_ricker,Dt,NDt,points,source_type,var0=meth.domain_source(dx,T,Ndt,dim,equ,example,ord,delta)

    # 7th order Runge-Kutta
    if method=='RK7':
      meth.sol_RK_7(var0,NDt,Dt,equ,dim,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example)

    # Faber polynomial approximation scalar
    if method=='FA':
        meth.sol_faber(var0,NDt,Dt,equ,dim,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,source_type,points,example,degree,ind_source)


# ------------------------------------
# Computing the solutions, with the examples of the paper
# ------------------------------------

# polynomial degrees used
degree=np.arange(3,41)

# equation and numerical parameters
parameters=np.array([[0.002,'scalar',1,'4',2,'1D_homogeneous_0'],[0.002,'scalar',1,'8',2,'1D_homogeneous_0'],[0.002,'scalar_dx2',1,'8',2,'1D_homogeneous_0'],
                     [0.002,'scalar',1,'4',2,'1D_heterogeneous_1a'],[0.002,'scalar',1,'8',2,'1D_heterogeneous_2'],[0.002,'scalar_dx2',1,'8',2,'1D_heterogeneous_2'],
                     [0.02,'scalar',2,'4',1.5,'2D_homogeneous_0a'],[0.02,'scalar_dx2',2,'4',1.5,'2D_homogeneous_0a'],[0.02,'scalar_dx2',2,'4',1.5,'2D_heterogeneous_2'],
                     [0.02,'scalar',2,'4',1.5,'2D_heterogeneous_3a'],[0.02,'scalar_dx2',2,'4',1.5,'2D_heterogeneous_3a'],[0.02,'scalar',2,'8',1.2,'2D_heterogeneous_3'],
                     [0.02,'elastic',2,'8',1.5,'2D_heterogeneous_3'],[0.02,'elastic',2,'4',1.5,'2D_heterogeneous_3']])

# cycle to compute the solutions
# for i in range(len(parameters)):
#    acoustic(dx=float(parameters[i,0]),equ=parameters[i,1],dim=int(parameters[i,2]),delta=0.8,beta0=30,ord=parameters[i,3],T=float(parameters[i,4]),Ndt=1,degree=degree,
#             example=parameters[i,5],method='RK7',ind_source='H_amplified') # for the reference solution
#
#    acoustic(dx=float(parameters[i,0]),equ=parameters[i,1],dim=int(parameters[i,2]),delta=0.8,beta0=30,ord=parameters[i,3],T=float(parameters[i,4]),Ndt=70,degree=degree,
#             example=parameters[i,5],method='FA',ind_source='H_amplified') # for Faber approximation


# degree=np.array([9])
# i=11
# acoustic(dx=0.02,equ=parameters[i,1],dim=int(parameters[i,2]),delta=0.8,beta0=30,ord=parameters[i,3],T=0.3,Ndt=1,degree=degree,
#              example=parameters[i,5],method='RK7',ind_source='H_amplified')
# acoustic(dx=0.01,equ=parameters[i,1],dim=int(parameters[i,2]),delta=0.8,beta0=30,ord=parameters[i,3],T=float(parameters[i,4]),Ndt=1,degree=degree,
#              example=parameters[i,5],method='FA',ind_source='H_amplified') # for Faber approximation

#
# # asdfasd
#
# acoustic(dx=0.04,equ="elastic",dim=int(parameters[i,2]),delta=0.8,beta0=30,ord=parameters[i,3],T=float(parameters[i,4]),Ndt=1,degree=degree,
#              example="2D_heterogeneous_3",method='RK7',ind_source='H_amplified')
# acoustic(dx=0.04,equ="elastic",dim=int(parameters[i,2]),delta=0.8,beta0=30,ord=parameters[i,3],T=float(parameters[i,4]),Ndt=1,degree=degree,
#              example="2D_heterogeneous_3",method='FA',ind_source='H_amplified') # for Faber approximation


# ---------------------------------------
# Solving the examples with a bash script
# ---------------------------------------

dx=float(sys.argv[1])
equ=sys.argv[2]
dim=int(sys.argv[3])
ord=sys.argv[4]
Ndt=int(sys.argv[5])
example=sys.argv[6]
method=sys.argv[7]
ind_source=sys.argv[8]
#
if dim==1:
    T=2
elif dim==2 and (example!='2D_heterogeneous3' or equ!='scalar'):
    T=1.5
elif dim==2 and (example=='2D_heterogeneous3' and equ=='scalar'):
    T=1.2

acoustic(dx=dx,equ=equ,dim=dim,delta=0.8,beta0=30,ord=ord,T=T,Ndt=Ndt,degree=degree,example=example,method=method,ind_source=ind_source)