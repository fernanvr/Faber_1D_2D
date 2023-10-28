from Stability import *


# ---------------------------------------------------------------------------------------------------------------------
# Stability results for Figures 10 and figure for the elastic equations in appendix
# ---------------------------------------------------------------------------------------------------------------------

degree=np.arange(3,41)
c2=1.524**2

# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='4',dim=1,equ='scalar',param=c2)
# stability(N_lamb=150,lamb_max=7,degree=degree,spatial_discr='4',dim=1,equ='scalar_dx2',param=c2)
# stability(N_lamb=150,lamb_max=7,degree=degree,spatial_discr='4',dim=2,equ='scalar',param=c2)
# stability(N_lamb=150,lamb_max=7,degree=degree,spatial_discr='4',dim=2,equ='scalar_dx2',param=c2)

# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='8',dim=1,equ='scalar',param=c2)
# stability(N_lamb=150,lamb_max=7,degree=degree,spatial_discr='8',dim=1,equ='scalar_dx2',param=c2)
# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='8',dim=2,equ='scalar',param=c2)
# stability(N_lamb=150,lamb_max=7,degree=degree,spatial_discr='8',dim=2,equ='scalar_dx2',param=c2)

# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='4',dim=2,equ='elastic',param=np.array([1,2.5,18]))
# stability(N_lamb=350,lamb_max=7,degree=degree,spatial_discr='8',dim=2,equ='elastic',param=np.array([1,2.5,18]))


# ---------------------------------------------------------------------------------------------------------------------
# Stability Figures 10 and figure for the elastic equations in appendix
# ---------------------------------------------------------------------------------------------------------------------

# graph_experiments(spatial_discr=np.array(["4",'8']),dim=np.array(['1']),equ=np.array(['scalar','scalar_dx2']),ind='1D')
# graph_experiments(spatial_discr=np.array(["4",'8']),dim=np.array(['2']),equ=np.array(['scalar','scalar_dx2']),ind='2D')
# graph_experiments(spatial_discr=np.array(["4",'8']),dim=np.array(['2']),equ=np.array(['elastic']),ind='elastic')

# eff_graph(degree=0,spatial_discr=np.array(["4",'8']),dim=np.array(['1']),equ=np.array(['scalar','scalar_dx2']),ind='1D')
# eff_graph(degree=0,spatial_discr=np.array(["4",'8']),dim=np.array(['2']),equ=np.array(['scalar','scalar_dx2']),ind='2D')
# eff_graph(degree=0,spatial_discr=np.array(["4",'8']),dim=np.array(['2']),equ=np.array(['elastic']),ind='elastic')
