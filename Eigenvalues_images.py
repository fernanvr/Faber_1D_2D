from Eigenvalues import *
# ---------------------------------------------------------------------------------------------------------------------
# Code for the computation of the spectrum of H and the eigenvalues limits, according to the results of the paper
# ---------------------------------------------------------------------------------------------------------------------

# param=np.array([['scalar',1,'4','1D_heterogeneous_2'],['scalar_dx2',1,'4','1D_heterogeneous_2'],['scalar',2,'4','2D_heterogeneous_3'],
#                     ['scalar_dx2',2,'4','2D_heterogeneous_3'],['elastic',2,'4','2D_heterogeneous_3'],['scalar',1,'8','1D_heterogeneous_2'],
#                 ['scalar_dx2',1,'8','1D_heterogeneous_2'],['scalar',2,'8','2D_heterogeneous_3'],['scalar_dx2',2,'8','2D_heterogeneous_3'],
#                 ['elastic',2,'8','2D_heterogeneous_3']])
#
# param=param[np.array([0,1]),:] # selecting only the experiments shown in the article
#
# for i in range(len(param)):
#
#     # condition to adjust dx for the different dimensions
#     if int(param[i,1])==1:
#         # dx=10.5/np.array([100,500,1000,5000])  # for the all spectrum
#         dx=10.5/(10*np.arange(3,500))          # for the limit eigenvalues
#     else:
#         # dx=8/np.array([10,50,100,500])         # for the all spectrum
#         dx=8/(5*np.arange(3,50))             # for the limit eigenvalues
#
#     # computing the spectrum and eigenvalue limits
#     # eigen_full(dx=dx,equ=param[i,0],dim=int(param[i,1]),delta=1.5,beta0=30,ord=param[i,2],T=2,Ndt=1,example=param[i,3])
#     # limit_eigen(dx=dx,equ=param[i,0],dim=int(param[i,1]),delta=1.5,beta0=30,ord=param[i,2],T=2,Ndt=1,example=param[i,3])
#
#     # plotting the spectrum (Figure 6)
#     graph_eigen_full(dx=dx[:4],equ=param[i,0],dim=int(param[i,1]),ord=param[i,2],example=param[i,3])


# ---------------------------------------------------------------------------------------------------------------------
# Figures for the eigenvalues limits, according to the paper experiments
# ---------------------------------------------------------------------------------------------------------------------

# equ=np.array(['scalar','scalar_dx2','scalar','scalar_dx2'])
# ord=np.array(['4','4','8','8'])
# example=np.array(['1D_heterogeneous_2','1D_heterogeneous_2','1D_heterogeneous_2','1D_heterogeneous_2'])
# graph_limit_eigen(dx=10.5/(10*np.arange(3,500)),equ=equ,delta=1.5,beta0=30,ord=ord,example=example,ind='Test3')

# equ=np.array(['scalar','scalar_dx2','scalar','scalar_dx2'])
# ord=np.array(['4','4','8','8'])
# example=np.array(['2D_heterogeneous_3','2D_heterogeneous_3','2D_heterogeneous_3','2D_heterogeneous_3'])
# graph_limit_eigen(dx=8/(10*np.arange(3,500)),equ=equ,delta=1.5,beta0=30,ord=ord,example=example,ind='Test5')

# equ=np.array(['elastic','elastic'])
# ord=np.array(['4','8'])
# example=np.array(['2D_heterogeneous_3','2D_heterogeneous_3'])
# graph_limit_eigen(dx=8/(10*np.arange(3,500)),equ=equ,delta=1.5,beta0=30,ord=ord,example=example,ind='Test7')
#
# equ=np.array(['scalar','scalar','scalar'])
# ord=np.array(['8','8','8'])
# example=np.array(['1D_homogeneous_0','1D_heterogeneous_1a','1D_heterogeneous_2'])
# graph_limit_eigen(dx=10.5/(10*np.arange(3,500)),equ=equ,delta=1.5,beta0=30,ord=ord,example=example,ind='Test_velocity')

# graph_limit_eigen_PML(dx=10.5/(10*np.arange(3,135)),equ='scalar',ord='8')