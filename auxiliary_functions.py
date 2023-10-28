import os
import numpy as np
import mpmath as mp
import sympy as sym
import operators as op
from scipy import integrate
from scipy import special
from scipy import linalg
from scipy import interpolate
from time import time
try:
    import segyio
except:
    print('np segyio is not installed')
# import segyio


# ---------------------------------------------------------------------------------------------------------
# Construction of mesh, the velocity field, the source parameters, and the time step sizes
# ---------------------------------------------------------------------------------------------------------

def domain_examples(example,dx,delta,equ):
    # return the velocity field, the mesh points, the source position, and the minimal time step size considered

    # cheking if there exist the paste to save the results, and creating one if there is not
    if not os.path.isdir(example + '/'):
        os.mkdir(example)

    if example[0]=='1':  # 1D sintetic examples

        # domain dimensions and amount of points
        a=10.5
        b=0
        nx=int(round(a/dx))
        ny=1
        print('nx: ',nx)

        # spatial grid points
        X=np.expand_dims(np.linspace(dx,a,nx),1)
        np.save(str(example)+'/X',X)
        Y=0*X

        # velocity field depending on the example
        param=np.zeros((nx,1))+pow(1.524,2)
        if example[3:16]=='heterogeneous':
            cut_pos=int(a/2/dx)
            param[cut_pos:]=pow(3.048,2)
            if example[17]=='1':
                cut_pos2=int(2*a/3/dx)
                param[cut_pos2:]=pow(0.1524,2)
        # source term position
        x0=2.6
        y0=0

    elif example[0]=='2': # 2D sintetic examples

        # domain dimensions and amount of points
        # a=4
        # b=4
        a=4+2*delta
        b=4+delta
        nx=int(round(a/dx))
        ny=int(round(b/dx))
        print('nx: ',nx)
        print('ny: ',ny)

        # spatial grid points
        x=np.linspace(dx,a,nx)
        y=np.linspace(b,dx,ny)
        X,Y=np.meshgrid(x,y)
        np.save(str(example)+'/X', X)
        np.save(str(example)+'/Y', Y)
        X=np.expand_dims(X.flatten('F'),1)
        Y=np.expand_dims(Y.flatten('F'),1)

        # velocity field depending on the example
        if equ=='elastic':
            param=np.zeros((5,ny,nx)) # rho, rho1, mu, mu1, lambda
            param[:2,:,:]=0.25
            param[2:4,:,:]=1
            param[4,:,:]=8
        else:
            param=np.zeros((ny,nx))+9
        if example[3:5]=='he':
            # cut_pos=int(b/2/dx)
            cut_pos=int((b-delta)/2/dx)
            if equ=='elastic':
                param[2,cut_pos:,:]=1.5
                param[3,cut_pos:,:]=1.5
                param[4,cut_pos:,:]=12
            else:
                param[cut_pos:,:]=36
            if example[17]=='3':
                # cut_pos_y=int(2*b/3/dx)
                # cut_pos_x=int(3/4*a/dx)
                cut_pos_y=int(3.5*b/6/dx)
                cut_pos_x=int(2/3*a/dx)
                if equ=='elastic':
                    param[2,cut_pos_y:,(cut_pos_x+1):]=2.25
                    param[3,cut_pos_y:,cut_pos_x:]=2.25
                    param[4,cut_pos_y:,(cut_pos_x+1):]=18
                else:
                    param[cut_pos_y:,cut_pos_x:]=1
        if equ=='elastic':
            aux_param=np.zeros((ny*nx,5))
            aux_param[:,0]=param[0,:,:].flatten('F')
            aux_param[:,1]=param[1,:,:].flatten('F')
            aux_param[:,2]=param[2,:,:].flatten('F')
            aux_param[:,3]=param[3,:,:].flatten('F')
            aux_param[:,4]=param[4,:,:].flatten('F')
            param=aux_param
        else:
            param=np.expand_dims(param.flatten('F'),1)

        # source term position
        x0=a/2
        y0=3*b/4

        if example[-1]=='b':
            # x0=np.array([1,2,3])
            # y0=np.array([b-0.2,b-0.2,b-0.2])
            y0=b-2*dx
        elif example[-1]=='c':
            y0=b-4*dx

    elif 'piece_GdM' in example: # 2D piece of Gto do Mato example

        # loading and constructing Gato do Mato piece velocity field with its spatial positions
        param=np.load('velocity_fields/vel.npy')
        x1=0
        x2=11.992
        y1=0
        y2=-6.379
        x_GdM=np.linspace(x1,x2,param.shape[1])
        y_GdM=np.linspace(y1,y2,param.shape[0])
        X_GdM,Y_GdM=np.meshgrid(x_GdM,y_GdM)

        # defining our computational spatial grid with the PML layer
        x1=x1-delta
        x2=x2+delta
        if ('_b' in example) or ('_c' in example):
            y1=-2
        else:
            y1=y1+delta
        y2=y2-delta
        nx=round((x2-x1)/dx)
        ny=round((y1-y2)/dx)
        print('nx: ',nx)
        print('ny: ',ny)
        x2=x1+nx*dx
        y2=y1-ny*dx
        x=np.linspace(x1+dx,x2,nx)
        y=np.linspace(y1,y2+dx,ny)
        X,Y=np.meshgrid(x,y)

        # interpolating the velocities to the computational grid
        points_GdM=np.hstack((np.expand_dims(X_GdM.flatten('F'),1),np.expand_dims(Y_GdM.flatten('F'),1)))
        points=np.hstack((np.expand_dims(X.flatten('F'),1),np.expand_dims(Y.flatten('F'),1)))
        param=interpolate.griddata(points_GdM,np.expand_dims(param.flatten('F'),1),points, method='nearest')

        param=param.reshape((ny,nx),order='F')
        print(param.shape)

        # converting to tring for further calculations
        np.save(str(example)+'/X', X)
        np.save(str(example)+'/Y', Y)
        X=np.expand_dims(X.flatten('F'),1)
        Y=np.expand_dims(Y.flatten('F'),1)
        param=np.expand_dims(param.flatten('F'),1)
        param=param**2

        # source term position
        if '_b' in example:
            y0=y1-2*dx
        elif '_c' in example:
            y0=y1-4*dx
        else:
            y0=y1-(y1-y2)/4
        x0=(x1+x2)/2

        # physical domain dimensions
        a=x2
        b=y1-y2
    elif 'SEG_EAGE' in example: # 2D piece of Gto do Mato example

        # loading and constructing Gato do Mato piece velocity field with its spatial positions
        param=np.transpose(np.load('velocity_fields/seg_eage_xcut_338.npy'))

        x1=0
        x2=13.52
        y1=0
        y2=-4.2
        x_SEG=np.linspace(x1,x2,param.shape[1])
        y_SEG=np.linspace(y1,y2,param.shape[0])
        X_SEG,Y_SEG=np.meshgrid(x_SEG,y_SEG)

        # defining our computational spatial grid with the PML layer
        if ('_b' in example) or ('_c' in example):
            y2=-3.5
            x1=2
            x2=11
        else:
            y1=y1+delta
        x1=x1-delta
        x2=x2+delta
        y2=y2-delta
        nx=round((x2-x1)/dx)
        ny=round((y1-y2)/dx)
        print('nx: ',nx)
        print('ny: ',ny)
        x2=x1+nx*dx
        y2=y1-ny*dx
        x=np.linspace(x1+dx,x2,nx)
        y=np.linspace(y1,y2+dx,ny)
        X,Y=np.meshgrid(x,y)

        # interpolating the velocities to the computational grid
        points_SEG=np.hstack((np.expand_dims(X_SEG.flatten('F'),1),np.expand_dims(Y_SEG.flatten('F'),1)))
        points=np.hstack((np.expand_dims(X.flatten('F'),1),np.expand_dims(Y.flatten('F'),1)))
        param=interpolate.griddata(points_SEG,np.expand_dims(param.flatten('F'),1),points, method='nearest')

        param=param.reshape((ny,nx),order='F')
        print(param.shape)

        # converting to tring for further calculations
        np.save(str(example)+'/X', X)
        np.save(str(example)+'/Y', Y)
        X=np.expand_dims(X.flatten('F'),1)
        Y=np.expand_dims(Y.flatten('F'),1)
        param=np.expand_dims(param.flatten('F'),1)
        param=param**2

        # source term position
        if '_b' in example:
            y0=y1-2*dx
        elif '_c' in example:
            y0=y1-4*dx
        else:
            y0=y1-(y2-y1)/4
        x0=(x1+x2)/2

        # physical domain dimensions
        a=x2-x1
        b=y1-y2
    elif 'Marmousi' in example: # 2D piece of Gto do Mato example

        # loading and constructing Gato do Mato piece velocity field with its spatial positions
        with segyio.open('velocity_fields/marmousi_perfil1.segy') as segyfile:
            param = np.transpose(segyio.tools.cube(segyfile)[0,:,:])
        x1=0
        x2=9.2
        y1=0
        y2=-3.5
        x_Marmousi=np.linspace(x1,x2,param.shape[1])
        y_Marmousi=np.linspace(y1,y2,param.shape[0])
        X_Marmousi,Y_Marmousi=np.meshgrid(x_Marmousi,y_Marmousi)

        # defining our computational spatial grid with the PML layer
        if ('_b' in example) or ('_c' in example):
            x1=2
            x2=8
        else:
            y1=y1+delta
            y2=y2-delta
            x1=x1-delta
            x2=x2+delta
        nx=round((x2-x1)/dx)
        ny=round((y1-y2)/dx)
        print('nx: ',nx)
        print('ny: ',ny)
        x2=x1+nx*dx
        y2=y1-ny*dx
        x=np.linspace(x1+dx,x2,nx)
        y=np.linspace(y1,y2+dx,ny)
        X,Y=np.meshgrid(x,y)

        # interpolating the velocities to the computational grid
        points_Marmousi=np.hstack((np.expand_dims(X_Marmousi.flatten('F'),1),np.expand_dims(Y_Marmousi.flatten('F'),1)))
        points=np.hstack((np.expand_dims(X.flatten('F'),1),np.expand_dims(Y.flatten('F'),1)))
        param=interpolate.griddata(points_Marmousi,np.expand_dims(param.flatten('F'),1),points, method='nearest')

        param=param.reshape((ny,nx),order='F')/1000

        # converting to tring for further calculations
        np.save(str(example)+'/X', X)
        np.save(str(example)+'/Y', Y)
        X=np.expand_dims(X.flatten('F'),1)
        Y=np.expand_dims(Y.flatten('F'),1)
        param=np.expand_dims(param.flatten('F'),1)
        param=param**2

        # source term position
        if '_b' in example:
            y0=y1-2*dx
        elif '_c' in example:
            y0=y1-4*dx
        else:
            y0=y1-(y2-y1)/4
        x0=(x2+x1)/2

        # physical domain dimensions
        a=x2-x1
        b=y1-y2

    # selection of the miminum time step size used for the simulations
    if equ=='elastic':
        dt=dx/np.max(np.sqrt((2*param[:,2]+param[:,4])*np.reciprocal(param[:,0])))/8
    else:
        dt=dx/np.max(np.sqrt(param))/8

    return a,b,nx,ny,X,Y,param,dt,x0,y0


def source_examples(equ,example,dim,delta,ord,dx,a,b,nx,ny,param,X,Y,x0,y0,T):
    # return the source types with its parameters, and the initial condition

    if example[0]=='1': # 1D examples

        if ord=='4':
            points_sol=np.array([-dx,-dx/2,0,a+dx/2,a+dx,a+dx*3/2,a+2*dx])
        elif ord[0]=='8':
            points_sol=np.array([-3*dx,-5*dx/2,-2*dx,-3*dx/2,-dx,-dx/2,0,a+dx/2,a+dx,a+dx*3/2,a+2*dx,a+5*dx/2,a+3*dx,a+7*dx/2])
        if example[3:16]=='heterogeneous':
            cut_pos=int(a/2/dx)
            points_sol=np.hstack((points_sol,cut_pos))
            if example[17]=='1':
                cut_pos2=int(2*a/3/dx)
                points_sol=np.hstack((points_sol,cut_pos2))

        if example=='1D_heterogeneous_0':
            f=example_11_source_x(points=points_sol,x=X,c2=param,delta=delta)
            solution=np.expand_dims(example_11_solution(points=points_sol,x=X,t=T),axis=1)
            np.save(str(example)+'/solution',solution)
            source_type='11'
        elif example[-1]=='1':
            f=source_1D_1(points=points_sol,x=X,c2=param,delta=delta)
            f[:,:3]=10*f[:,:3]
            solution=source_1D_1_solution(points=points_sol,x=X,t=T)
            np.save(str(example)+'/solution',solution)
            source_type=example
        else:
            f=source_x_1D(x0=x0,rad=0.01,X=np.matrix.flatten(X,'F'),nx=nx,equ=equ,delta=delta)
            source_type=example
            if equ=='scalar_dx2':
                source_type=source_type+'S'
            else:
                source_type=source_type+'X'

        if example=='1D_homogeneous_0':
            source_type="8"
            f=f*0
            solution=np.zeros((nx,1))
            solution[:,0]=example_8_solution(X,a,T,np.sqrt(param[0]))[:,0]
            np.save(str(example)+'/solution',solution)

        if example=='1D_heterogeneous_1a':
            var0=f
            source_type='8'
            f=f*0
        else:
            var0=ini_var0(dim,equ,delta,nx,ny,X,source_type,a,points_sol,dx)

    else: # 2D examples

        if ord=='4':
            points_sol_x=np.array([-dx,-dx/2,0,a+dx/2,a+dx,a+dx*3/2,a+2*dx])
            points_sol_y=np.array([-dx,-dx/2,0,b+dx/2,b+dx,b+dx*3/2,b+2*dx])
        elif ord[0]=='8':
            points_sol_x=np.array([-3*dx,-5*dx/2,-2*dx,-3*dx/2,-dx,-dx/2,0,a+dx/2,a+dx,a+dx*3/2,a+2*dx,a+5*dx/2,a+3*dx,a+7*dx/2,a+4*dx])
            points_sol_y=np.array([-3*dx,-5*dx/2,-2*dx,-3*dx/2,-dx,-dx/2,0,b+dx/2,b+dx,b+dx*3/2,b+2*dx,b+5*dx/2,b+3*dx,b+7*dx/2,b+4*dx])

        if example[3:5]=='he':
            cut_pos=int(b/2/dx)
            points_sol_y=np.hstack((points_sol_y,cut_pos))
            if example[17]=='3':
                points_sol_y=np.hstack((points_sol_y,cut_pos))
                points_sol_x=np.hstack((points_sol_x,cut_pos))

        if example[-1]=='0':
            f=example_source_2D(points_x=points_sol_x,points_y=points_sol_y,x=X,y=Y,c2=param,delta=delta)
            f=10*f
            solution=np.expand_dims(f[:nx*ny,0]*(np.exp(T)-1),axis=1)
            np.save(str(example)+'/solution',solution)
            source_type=1
            var0=np.zeros((len(X)*3,1))
        elif example[-1]=='1':
            f=source_2D_1(points_x=points_sol_x,points_y=points_sol_y,x=X,y=Y,c2=param,delta=delta)
            solution=np.expand_dims(f[:len(X),2]*f[:len(X),5]*np.cos(f[:len(X),6]+T)*np.sin(5*f[:len(X),7]-7*T),axis=1)
            np.save(str(example)+'/solution',solution)
            source_type=example
            var0=source_2D_1_ini_var0(points_sol_x,points_sol_y,X,Y,dx)
        else:
            f=source_x_2D(x0=x0,y0=y0,rad=0.02,X=X,Y=Y,nx=nx,ny=ny,equ=equ,delta=delta)
            var0=np.zeros((len(f[:,0]),1))
            source_type=example
            if equ!='scalar':
                source_type=source_type+'S'
            else:
                source_type=source_type+'X'

        if example=='2D_homogeneous_0a' or example=='2D_heterogeneous_3a':
            var0=f
            f=f*0
            source_type='8'


    return var0,f,source_type


# spatial part of source functions for the particular examples

def example_11_source_x(points,x,c2,delta):
    # spatial source term when example==1D_heterogeneous_0

    # INPUTS:
    # x0, y0: spatial coordinates of the source term
    # rad: radius of the source impulse
    # nx-1: number of grid points in the x direction
    # ny-1: number of grid points in the y direction

    # OUTPUT: Spatial term of the source

    S=np.expand_dims(pow(np.prod(np.sin(x-points),axis=1),2),axis=1)
    n=len(points)
    aux=-2*n*S
    for i in range(n-1):
        aux0=0
        for j in range(i+1,n):
            aux0=aux0+np.cos(x-points[j])*np.sin(x-points[j])*np.expand_dims(pow(np.prod(np.sin(x-points[[k for k in range(n) if k not in np.array([i,j])]]),axis=1),2),axis=1)
        aux=aux+8*np.cos(x-points[i])*np.sin(x-points[i])*aux0+2*pow(np.cos(x-points[i]),2)*np.expand_dims(pow(np.prod(np.sin(x-points[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
    aux=aux+2*pow(np.cos(x-points[n-1]),2)*np.expand_dims(pow(np.prod(np.sin(x-points[[k for k in range(n) if k not in np.array([n-1])]]),axis=1),2),axis=1)
    if delta==0:
        source=np.zeros((len(x)*2,2))
    else:
        source=np.zeros((len(x)*3,2))
    source[:len(x),0]=S[:,0]
    source[:len(x),1]=(c2*aux)[:,0]
    return source


def example_11_solution(points,x,t):
    # analitical solution when example==1D_heterogeneous_0

    return pow(np.prod(np.sin(x-points),axis=1),2)*(np.exp(t)-1)


def source_1D_1(points,x,c2,delta):
    # spatial source term for when example==1D*1 (the * is means that anything can be in the middle)

    S=np.expand_dims(pow(np.prod(np.sin(x-points),axis=1),2),axis=1)
    n=len(points)
    aux=-2*n*S
    aux1=0
    for i in range(n-1):
        aux1=aux1+np.cos(x-points[i])*np.sin(x-points[i])*np.expand_dims(pow(np.prod(np.sin(x-points[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
        aux0=0
        for j in range(i+1,n):
            aux0=aux0+np.cos(x-points[j])*np.sin(x-points[j])*np.expand_dims(pow(np.prod(np.sin(x-points[[k for k in range(n) if k not in np.array([i,j])]]),axis=1),2),axis=1)
        aux=aux+8*np.cos(x-points[i])*np.sin(x-points[i])*aux0+2*pow(np.cos(x-points[i]),2)*np.expand_dims(pow(np.prod(np.sin(x-points[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
    aux=aux+2*pow(np.cos(x-points[n-1]),2)*np.expand_dims(pow(np.prod(np.sin(x-points[[k for k in range(n) if k not in np.array([n-1])]]),axis=1),2),axis=1)
    aux1=2*(aux1+np.cos(x-points[n-1])*np.sin(x-points[n-1])*np.expand_dims(pow(np.prod(np.sin(x-points[[k for k in range(n) if k not in np.array([n-1])]]),axis=1),2),axis=1))
    if delta==0:
        source=np.zeros((len(x)*2,4))
    else:
        source=np.zeros((len(x)*3,4))

    source[:len(x),0]=-(2*c2*aux1)[:,0]
    source[:len(x),1]=(c2*(S-aux)-S)[:,0]
    source[:len(x),2]=S[:,0]
    source[:len(x),3]=x[:,0]

    return source


def source_1D_1_solution(points,x,t):
    # analitical solution using the source function for when example==1D*1 (the * is means that anything can be in the middle)

    return 10*np.expand_dims(pow(np.prod(np.sin(x-points),axis=1),2),axis=1)*np.cos(x+t)


def source_x_1D(x0,rad,X,nx,equ,delta):
    # spatial source term of a localized wavelet around a specific point in 1D examples

    # INPUTS:
    # x0, y0: spatial coordinates of the source term
    # rad: radius of the source impulse
    # X: position of the grid points
    # nx-1: number of grid points in the x direction
    # equ: equation formulation
    # abc: absorbing boundary condition indicator

    # OUTPUT: Spatial term of the source

    S=pow(X-x0,2)
    ind=(S>pow(rad,2))
    S=np.exp(S*np.reciprocal(S-pow(rad,2)))
    S[ind]=0
    if delta==0:
        Source=np.zeros((nx*2,1))
    else:
        Source=np.zeros((nx*3,1))

    if equ=='scalar':
        Source[:nx,0]=S
    elif equ=='scalar_dx2':
        Source[nx:2*nx,0]=S

    return Source


def example_8_solution(x,a,t,c):
    # analytical solution of "example_8" using D'Alembert formula when example==1D_homogeneous_0

    sol=np.zeros((len(x),1))
    for i in range(len(x)):
        k=np.floor(((x[i]+c*t)/a+1)/2)
        x_p=x[i]+c*t-2*k*a

        if x_p<0:
            sol[i]=-example_8_ic(-x_p,a)/2
            # sol[i]=-example_15_1(-x_p,a)/2
        else:
            sol[i]=example_8_ic(x_p,a)/2
            # sol[i]=example_15_1(x_p,a)/2

        k=np.floor(((x[i]-c*t)/a+1)/2)

        x_p=x[i]-c*t-2*k*a

        if x_p<0:
            sol[i]=sol[i]-example_8_ic(-x_p,a)/2
            # sol[i]=sol[i]-example_15_1(-x_p,a)/2
        else:
            sol[i]=sol[i]+example_8_ic(x_p,a)/2
            # sol[i]=sol[i]+example_15_1(x_p,a)/2
    return sol


def example_8_ic(x,a):
    # initial condition when example==1D_homogeneous_0

    return np.exp(-10*pow(x-a/2,2))*(1-10*pow(x-a/2,2))


def example_source_2D(points_x,points_y,x,y,c2,delta):
    # spatial part of the source term when example=2*0 (* means any characters in the middle)

    S=np.expand_dims(pow(np.prod(np.sin(x-points_x),axis=1),2),axis=1)
    n=len(points_x)
    aux=-2*n*S
    for i in range(n-1):
        aux0=0
        for j in range(i+1,n):
            aux0=aux0+np.cos(x-points_x[j])*np.sin(x-points_x[j])*np.expand_dims(pow(np.prod(np.sin(x-points_x[[k for k in range(n) if k not in np.array([i,j])]]),axis=1),2),axis=1)
        aux=aux+8*np.cos(x-points_x[i])*np.sin(x-points_x[i])*aux0+2*pow(np.cos(x-points_x[i]),2)*np.expand_dims(pow(np.prod(np.sin(x-points_x[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
    aux=aux+2*pow(np.cos(x-points_x[n-1]),2)*np.expand_dims(pow(np.prod(np.sin(x-points_x[[k for k in range(n) if k not in np.array([n-1])]]),axis=1),2),axis=1)

    n=len(points_y)
    aux1=-2*n*np.expand_dims(pow(np.prod(np.sin(y-points_y),axis=1),2),axis=1)
    for i in range(n-1):
        aux0=0
        for j in range(i+1,n):
            aux0=aux0+np.cos(y-points_y[j])*np.sin(y-points_y[j])*np.expand_dims(pow(np.prod(np.sin(y-points_y[[k for k in range(n) if k not in np.array([i,j])]]),axis=1),2),axis=1)
        aux1=aux1+8*np.cos(y-points_y[i])*np.sin(y-points_y[i])*aux0+2*pow(np.cos(y-points_y[i]),2)*np.expand_dims(pow(np.prod(np.sin(y-points_y[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
    aux1=aux1+2*pow(np.cos(y-points_y[n-1]),2)*np.expand_dims(pow(np.prod(np.sin(y-points_y[[k for k in range(n) if k not in np.array([n-1])]]),axis=1),2),axis=1)

    if delta==0:
        source=np.zeros((len(x)*3,3))
    else:
        source=np.zeros((len(x)*5,3))
    source[:len(x),0]=(S*np.expand_dims(pow(np.prod(np.sin(y-points_y),axis=1),2),axis=1))[:,0]
    source[:len(x),1]=(c2*aux*np.expand_dims(pow(np.prod(np.sin(y-points_y),axis=1),2),axis=1))[:,0]
    source[:len(x),2]=(c2*aux1*S)[:,0]

    return source


def source_2D_1(points_x,points_y,x,y,c2,delta):
    # spatial part of the source term when example=2*1 (* means any characters in the middle)

    S=np.expand_dims(pow(np.prod(np.sin(x-points_x),axis=1),2),axis=1)
    n=len(points_x)
    aux=-2*n*S
    aux0_x=0
    for i in range(n-1):
        aux0_x=aux0_x+np.cos(x-points_x[i])*np.sin(x-points_x[i])*np.expand_dims(pow(np.prod(np.sin(x-points_x[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
        aux0=0
        for j in range(i+1,n):
            aux0=aux0+np.cos(x-points_x[j])*np.sin(x-points_x[j])*np.expand_dims(pow(np.prod(np.sin(x-points_x[[k for k in range(n) if k not in np.array([i,j])]]),axis=1),2),axis=1)
        aux=aux+8*np.cos(x-points_x[i])*np.sin(x-points_x[i])*aux0+2*pow(np.cos(x-points_x[i]),2)*np.expand_dims(pow(np.prod(np.sin(x-points_x[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
    aux=aux+2*pow(np.cos(x-points_x[n-1]),2)*np.expand_dims(pow(np.prod(np.sin(x-points_x[[k for k in range(n) if k not in np.array([n-1])]]),axis=1),2),axis=1)
    aux0_x=2*(aux0_x+np.cos(x-points_x[n-1])*np.sin(x-points_x[n-1])*np.expand_dims(pow(np.prod(np.sin(x-points_x[[k for k in range(n) if k not in np.array([n-1])]]),axis=1),2),axis=1))

    n=len(points_y)
    aux1=-2*n*np.expand_dims(pow(np.prod(np.sin(y-points_y),axis=1),2),axis=1)
    aux0_y=0
    for i in range(n-1):
        aux0_y=aux0_y+np.cos(y-points_y[i])*np.sin(y-points_y[i])*np.expand_dims(pow(np.prod(np.sin(y-points_y[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
        aux0=0
        for j in range(i+1,n):
            aux0=aux0+np.cos(y-points_y[j])*np.sin(y-points_y[j])*np.expand_dims(pow(np.prod(np.sin(y-points_y[[k for k in range(n) if k not in np.array([i,j])]]),axis=1),2),axis=1)
        aux1=aux1+8*np.cos(y-points_y[i])*np.sin(y-points_y[i])*aux0+2*pow(np.cos(y-points_y[i]),2)*np.expand_dims(pow(np.prod(np.sin(y-points_y[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
    aux1=aux1+2*pow(np.cos(y-points_y[n-1]),2)*np.expand_dims(pow(np.prod(np.sin(y-points_y[[k for k in range(n) if k not in np.array([n-1])]]),axis=1),2),axis=1)
    aux0_y=2*(aux0_y+np.cos(y-points_y[n-1])*np.sin(y-points_y[n-1])*np.expand_dims(pow(np.prod(np.sin(y-points_y[[k for k in range(n) if k not in np.array([n-1])]]),axis=1),2),axis=1))
    if delta==0:
        source=np.zeros((len(x)*3,9))
    else:
        source=np.zeros((len(x)*5,9))

    source[:len(x),0]=aux[:,0]
    source[:len(x),1]=aux0_x[:,0]
    source[:len(x),2]=S[:,0]
    source[:len(x),3]=aux1[:,0]
    source[:len(x),4]=aux0_y[:,0]
    source[:len(x),5]=np.expand_dims(pow(np.prod(np.sin(y-points_y),axis=1),2),axis=1)[:,0]
    source[:len(x),6]=x[:,0]
    source[:len(x),7]=y[:,0]
    source[:len(x),8]=c2[:,0]

    return source


def source_2D_1_ini_var0(points_x,points_y,x,y,dx):
    # initial condition when example=2*1 (* means any characters in the middle)

    n=len(points_x)
    aux0_x=0
    for i in range(n):
        aux0_x=aux0_x+np.cos(x-dx/2-points_x[i])*np.sin(x-dx/2-points_x[i])*np.expand_dims(pow(np.prod(np.sin(x-dx/2-points_x[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
    aux0_x=2*aux0_x

    n=len(points_y)
    aux0_y=0
    for i in range(n):
        aux0_y=aux0_y+np.cos(y-dx/2-points_y[i])*np.sin(y-dx/2-points_y[i])*np.expand_dims(pow(np.prod(np.sin(y-dx/2-points_y[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
    aux0_y=2*aux0_y

    var0=np.zeros((len(x)*3,1))

    S1=np.expand_dims(pow(np.prod(np.sin(x-points_x),axis=1),2),axis=1)
    S1x=np.expand_dims(pow(np.prod(np.sin(x-dx/2-points_x),axis=1),2),axis=1)
    S2=np.expand_dims(pow(np.prod(np.sin(y-points_y),axis=1),2),axis=1)
    S2y=np.expand_dims(pow(np.prod(np.sin(y-dx/2-points_y),axis=1),2),axis=1)
    var0[:len(x)]=S1*S2*np.cos(x)*np.sin(5*y)
    var0[len(x):2*len(x)]=(aux0_x*(np.cos(-5*y+x-dx/2)/16+np.cos(-5*y-x+dx/2)/12)-S1x*(np.sin(-5*y+x-dx/2)/16-np.sin(-5*y-x+dx/2)/12))*S2
    var0[2*len(x):3*len(x)]=(aux0_y*(np.cos(-5*(y-dx/2)+x)/16+np.cos(-5*(y-dx/2)-x)/12)+5*S2y*(np.sin(-5*(y-dx/2)+x)/16+np.sin(-5*(y-dx/2)-x)/12))*S1
    return var0


def source_x_2D(x0,y0,rad,X,Y,nx,ny,equ,delta):
    # spatial source term of a localized wavelet around a specific point in 2D examples

    # INPUTS:
    # x0, y0: spatial coordinates of the source term
    # rad: radius of the source impulse
    # X, Y: spatial position of the grid points in the x and y directions, respectively
    # nx-1: number of grid points in the x direction
    # ny-1: number of grid points in the y direction
    # equ: type of equation formulation
    # abc: absorbing boundary condition indicator

    # OUTPUT: Spatial term of the source

    S=pow(X-x0,2)+pow(Y-y0,2)
    ind=(S>pow(rad,2)-10e-10)
    S=np.exp(S*np.reciprocal(S-pow(rad,2)))
    S[ind]=0

    if S.ndim==2:
        S=np.expand_dims(np.sum(S,axis=1),axis=1)

    if delta==0:
        if equ=='scalar':
            Source=np.zeros((nx*ny*3,1))
        elif equ=='scalar_dx2':
            Source=np.zeros((nx*ny*2,1))
        elif equ=='elastic':
            Source=np.zeros((nx*ny*7,1))
    else:
        if equ=='scalar':
            Source=np.zeros((nx*ny*5,1))
        elif equ=='scalar_dx2':
            Source=np.zeros((nx*ny*4,1))
        elif equ=='elastic':
            Source=np.zeros((nx*ny*11,1))

    if equ=='scalar':
        Source[:nx*ny,0]=S[:,0]
    elif equ=='scalar_dx2':
        Source[nx*ny:2*nx*ny,0]=S[:,0]
    elif equ=='elastic':
        Source[2*nx*ny:3*nx*ny,0]=S[:,0]
        Source[3*nx*ny:4*nx*ny,0]=S[:,0]

    return Source


# functions related to the initial condition of the examples

def ini_var0(dim,equ,delta,nx,ny,x,source_type,a,points_sol,dx):
    # initial condition for several scenarios of the examples listed before

    if dim==1:
        if delta!=0:
            var0=np.zeros((nx*3,1))
        else:
            var0=np.zeros((nx*2,1))
        if source_type=="8":
            var0[:nx]=example_8_ic(x,a)
        elif source_type=="15_1":
            var0[:nx]=example_15_1(x,a)
        elif source_type=="15_2":
            var0[:nx]=example_15_2(x,a)
        elif source_type[-1]=='1' and source_type[0]=='1' and source_type!='11':
            var0=source_1D_1_ini_var0(points_sol,x,dx)
    elif dim==2:
        if equ=='scalar':
            if delta!=0:
                var0=np.zeros((nx*ny*5,1))
            elif delta==0:
                var0=np.zeros((nx*ny*3,1))
        elif equ=='scalar_dx2':
            if delta!=0:
                var0=np.zeros((nx*ny*4,1))
            elif delta==0:
                var0=np.zeros((nx*ny*2,1))
        elif equ=='elastic' and delta!=0:
            var0=np.zeros((nx*ny*11,1))

    return var0


def example_15_1(x,a):
    # solution of a variation of example==1D_homogeneous_0

    return np.exp(-10*pow(x-a/2,2))+np.exp(-20*pow(x-a/3,2))-np.exp(-20*pow(x-a/3,2))


def example_15_2(x,a):
    # solution of a variation of example==1D_homogeneous_0

    return -np.exp(-10*pow(x-a/2,2))*(1-10*pow(x-a/2,2))+np.exp(-20*pow(x-a/3,2))*(1-20*pow(x-a/2,2))


def source_1D_1_ini_var0(points,x,dx):
    # initil condition when example==2*1 (* means any characters in the middle)

    S=np.expand_dims(pow(np.prod(np.sin(x-points),axis=1),2),axis=1)
    n=len(points)
    aux1=0
    x=x-dx/2
    for i in range(n-1):
        aux1=aux1+np.cos(x-points[i])*np.sin(x-points[i])*np.expand_dims(pow(np.prod(np.sin(x-points[[k for k in range(n) if k not in np.array([i])]]),axis=1),2),axis=1)
    aux1=2*(aux1+np.cos(x-points[n-1])*np.sin(x-points[n-1])*np.expand_dims(pow(np.prod(np.sin(x-points[[k for k in range(n) if k not in np.array([n-1])]]),axis=1),2),axis=1))

    var0=np.zeros((len(x)*2,1))
    var0[:len(x)]=S*np.cos(x+dx/2)
    var0[len(x):]=aux1*np.sin(x)+np.expand_dims(pow(np.prod(np.sin(x-points),axis=1),2),axis=1)*np.cos(x)

    return var0*10


# ---------------------------------------------------------------------------------------------------------
# Core functions of the different schemes considered to solve the wave equation
# ---------------------------------------------------------------------------------------------------------

# low order Runge-Kutta functions

def RK_7_source(var,dt,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,f,param_ricker,i,source_type):
    # 9 stage 7th order Runge-Kutta

    k1=op.op_H(var,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*i,param_ricker,source_type)
    k2=op.op_H(var+dt*4/63*k1,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+4/63),param_ricker,source_type)
    k3=op.op_H(var+dt*(1/42*k1+1/14*k2),equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+2/21),param_ricker,source_type)
    k4=op.op_H(var+dt*(1/28*k1+3/28*k3),equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+1/7),param_ricker,source_type)
    k5=op.op_H(var+dt*(12551/19652*k1-48363/19652*k3+10976/4913*k4),equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+7/17),param_ricker,source_type)
    k6=op.op_H(var+dt*(-36616931/27869184*k1+2370277/442368*k3-255519173/63700992*k4+226798819/445906944*k5),equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+13/24),param_ricker,source_type)
    k7=op.op_H(var+dt*(-10401401/7164612*k1+47383/8748*k3-4914455/1318761*k4-1498465/7302393*k5+2785280/3739203*k6),equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+7/9),param_ricker,source_type)
    k8=op.op_H(var+dt*(181002080831/17500000000*k1-14827049601/400000000*k3+23296401527134463/857600000000000*k4+2937811552328081/949760000000000*k5-243874470411/69355468750*k6+2857867601589/3200000000000*k7),equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+91/100),param_ricker,source_type)
    k9=op.op_H(var+dt*(-228380759/19257212*k1+4828803/113948*k3-331062132205/10932626912*k4-12727101935/3720174304*k5+22627205314560/4940625496417*k6-268403949/461033608*k7+3600000000000/19176750553961*k8),equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+1),param_ricker,source_type)

    return var+dt*(95/2366*k1+3822231133/16579123200*k4+555164087/2298419200*k5+1279328256/9538891505*k6+5963949/25894400*k7+50000000000/599799373173*k8+28487/712800*k9)


def RK_2(var,dt,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,i,f,param_ricker,source_type):
    # 3 stages 2nd order Runge-Kutta

    k1=op.op_H(var,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*i,param_ricker,source_type)
    k2=op.op_H(var+dt*k1/2,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+1/2),param_ricker,source_type)
    k3=op.op_H(var+dt*k2/2,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+1/2),param_ricker,source_type)

    return var+dt*k3


def RK_4(var,dt,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,i,f,param_ricker,source_type):
    # 4 stages 4th order Runge-Kutta

    k1=op.op_H(var,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*i,param_ricker,source_type)
    k2=op.op_H(var+dt*k1/2,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+1/2),param_ricker,source_type)
    k3=op.op_H(var+dt*k2/2,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+1/2),param_ricker,source_type)
    k4=op.op_H(var+dt*k3,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)+source_xt(f,dt*(i+1),param_ricker,source_type)

    return var+dt*(k1+2*(k2+k3)+k4)/6


# Leapfrog functions

def ini_var0_2MS(var0,nx,ny,dim,delta):
    # initial condition for the Leapfrog scheme, it change because the amount of equatiosn are not the same

    if delta==0:
        return var0[:nx*ny]
    else:
        if dim==1:
            var=np.zeros((2*nx,1))
        else:
            var=np.zeros((3*nx*ny,1))
        var[:nx*ny]=var0[:nx*ny]
        return var


def method_time_2steps(var0,Ndt,Nt,dt,T_frac_snapshot,nx,ny,dx,c2,source_type,f,param_ricker,equ,dim,free_surf,delta,beta0,ord,points,example):
    # computation of the Leapfrog solution at the last time instant

    # declaration of the array saving the solution at fixed points for all time instants
    sol_2time_points=np.zeros((Nt,len(points)))

    # initialization of the solution variable with the initial condition
    var=var0

    # constructing the solution at time instant -1 to use the second order in time scheme
    var0=minus1_step_2MS(var0,dx,nx,dt,c2,f,example)

    # cyle to compute the solution at each time instant
    for j in range(Nt):
        sol_2time_points[j,:]=var[points,0]

        # calculating the solution in the next time instant
        var1=time_step_2MS(var0,var,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx,ny,dt,f,j*dt,param_ricker,source_type)

        var0=var+0
        var=var1+0

        if j==(round(Nt*T_frac_snapshot)-1):
            np.save(str(example)+'/sol_2MS_equ_'+str(equ)+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_'+str(Ndt)+'_dx_'+str(dx),var[:nx*ny,0])

    np.save(str(example)+'/sol_2MS_equ_'+str(equ)+'_free_surf_'+str(free_surf)+'_ord_'+ord+'_Ndt_'+str(Ndt)+'_dx_'+str(dx)+'_points',sol_2time_points[::2,:])

    return var


def minus1_step_2MS(var0,dx,nx,dt,c2,f,example):
    # construction of the solution at time instant -1 according to the different examples considered

    if example=='1D_homogeneous_0':
        var_minus1=example_8_solution(np.expand_dims(np.linspace(dx,10.5,nx),1),10.5,-dt,np.sqrt(c2[0]))
    elif example=='1D_heterogeneous_0':
        var_minus1=np.expand_dims(f[:,0]*(np.exp(-dt)-1),axis=1)
    elif example[0]=='1' and example[-1]=='1':
        var_minus1=np.expand_dims(f[:,2]*np.cos(f[:,3]-dt),axis=1)
    elif example[0]=='2' and example[-1]=='0':
        var_minus1=np.expand_dims(f[:,0]*(np.exp(-dt)-1),axis=1)
    elif example[0]=='2' and example[-1]=='1':
        var_minus1=np.expand_dims(f[:,2]*f[:,5]*np.cos(f[:,6]-dt)*np.sin(5*f[:,7]+7*dt),axis=1)
    elif example=="Dispersion_S" and os.path.isfile('Dispersion_S/2MS_minus1.npy'):
        var_minus1=np.load('Dispersion_S/2MS_minus1.npy')
        os.remove('Dispersion_S/2MS_minus1.npy')
    else:
        var_minus1=var0*0

    return var_minus1


def time_step_2MS(var0,var,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx,ny,dt,f,t,param_ricker,source_type):
    # using the Leapfrog second order in time approximation to compute the solution.
    # The conditionals are product of difference in the formulations regarding in the problem dimensions,
    # and the use of absorbing boundary conditions.

    if delta==0:
        var1=2*var-var0+dt**2*(op.op_H_2ord(var,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx+1,ny+1)+source_xt(f,t,param_ricker,str(source_type)+'_2MS'))
    else:
        aux=op.op_H_2ord(var,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx+1,ny+1)
        var1=var0*0
        if dim==1:
            var1[:nx]=2*var[:nx]-var0[:nx]+dt*op.beta_i(dim,dx,nx+1,ny+1,delta,beta0,1,0,free_surf)*(var[:nx]-var0[:nx])+dt**2*(aux[:nx]+source_xt(f,t,param_ricker,str(source_type)+'_2MS'))
            var1[nx:]=4*var[nx:]-3*var0[nx:]-2*dt*(aux[nx:])
        else:
            var1[:nx*ny]=2*var[:nx*ny]-var0[:nx*ny]-dt*(op.beta_i(dim,dx,nx+1,ny+1,delta,beta0,1,0,free_surf)+op.beta_i(dim,dx,nx+1,ny+1,delta,beta0,2,0,free_surf))*(var[:nx*ny]-var0[:nx*ny])\
                         +dt**2*(aux[:nx*ny]+source_xt(f,t,param_ricker,str(source_type)+'_2MS'))
            # First order variation
            # var1[nx*ny:2*nx*ny]=var[nx*ny:2*nx*ny]+dt*aux[nx*ny:2*nx*ny]
            # var1[2*nx*ny:]=var[2*nx*ny:]+dt*aux[2*nx*ny:]

            # Leap-frog first order variation
            var1[nx*ny:2*nx*ny]=var0[nx*ny:2*nx*ny]+2*dt*aux[nx*ny:2*nx*ny]
            var1[2*nx*ny:]=var0[2*nx*ny:]+2*dt*aux[2*nx*ny:]

            # this 2nd order approximation is unstable
            # var1[nx*ny:2*nx*ny]=4*var[nx*ny:2*nx*ny]-3*var0[nx*ny:2*nx*ny]-2*dt*(aux[nx*ny:2*nx*ny])
            # var1[2*nx*ny:]=4*var[2*nx*ny:]-3*var0[2*nx*ny:]-2*dt*(aux[2*nx*ny:])


    return var1


# Faber functions

def ellipse_parameters(a12,a22,b1,b2,c):

    a11=1
    th=0
    if np.abs(a12)>pow(10,-14):
        if np.abs(a22-1)<pow(10,-14):
            th=np.pi/4
        else:
            th=np.arctan(a12/(1-a22))/2

        a11=pow(np.cos(th),2)+a12*np.sin(th)*np.cos(th)+a22*pow(np.sin(th),2)
        a22=pow(np.sin(th),2)-a12*np.sin(th)*np.cos(th)+a22*pow(np.cos(th),2)
        aux=b1
        b1=b1*np.cos(th)+b2*np.sin(th)
        b2=b2*np.cos(th)-aux*np.sin(th)
    # print('th: ',th)

    x0=b1/(2*a11)
    y0=b2/(2*a22)
    a=np.sqrt((pow(x0,2)*a11+pow(y0,2)*a22-c)/a11)
    b=np.sqrt((pow(x0,2)*a11+pow(y0,2)*a22-c)/a22)

    # print('a:',a)
    # print('b:', b)
    # print('(a+b)^2: ',pow(a+b,2))

    return (a+b)/2,np.sqrt(np.abs(a**2-b**2)),-np.array([x0*np.cos(th)-y0*np.sin(th),x0*np.sin(th)+y0*np.cos(th)]),a


def spectral_dist(equ,dim,delta,beta0,ord,dx,param):
    # return the vertex of the polygon containing the convex envelope of H spectrum

    if equ=='elastic':
        c2=np.sqrt(np.max((2*param[2]+param[4])*np.reciprocal(param[0])))
        if ord=='4':
            c2=c2*7.95
        elif ord=='8':
            c2=c2*7.37
    elif equ=='scalar':
        c2=np.sqrt(np.max(param))
        if ord=='4':
            if dim==1:
                c2=c2*2.34
            elif dim==2:
                c2=c2*3.31
        elif ord=='8':
            if dim==1:
                c2=c2*2.58
            elif dim==2:
                c2=c2*3.65
    elif equ=='scalar_dx2':
        c2=np.sqrt(np.max(param))
        if ord=='4':
            if dim==1:
                c2=c2*2.31
            elif dim==2:
                c2=c2*3.27
        elif ord=='8':
            if dim==1:
                c2=c2*2.55
            elif dim==2:
                c2=c2*3.62
    if dim==1:
        if delta==0:
            vals=np.array([-dx+1j*c2/dx,-dx-1j*c2/dx,1j*c2/dx,-1j*c2/dx])
        else:
            vals=np.array([-beta0*pow((delta-dx/2)/delta,2)+1j*c2/dx,-beta0*pow((delta-dx/2)/delta,2)-1j*c2/dx,1j*c2/dx,-1j*c2/dx])
    elif dim==2:
        # vals=np.array([-beta0*pow((delta-dx/2)/delta,2)+1j*c2/dx,-beta0*pow((delta-dx/2)/delta,2)-1j*c2/dx,1j*c2/dx,-1j*c2/dx])
        # vals=np.array([-beta0,-beta0*pow((delta-dx/2)/delta,2)/2+1j*c2/dx,-beta0*pow((delta-dx/2)/delta,2)/2-1j*c2/dx,1j*c2/dx,-1j*c2/dx])
        if beta0*0.75<-c2/dx/10:
            if delta==0:
                vals=np.array([-dx/2+1j*c2/dx,-dx/2-1j*c2/dx,1j*c2/dx,-1j*c2/dx])
            else:
                vals=np.array([-beta0*pow((delta-dx/2)/delta,2)/2+1j*c2/dx,-beta0*pow((delta-dx/2)/delta,2)/2-1j*c2/dx,1j*c2/dx,-1j*c2/dx])
        else:
            if delta==0:
                vals=np.array([-dx,-dx/2+1j*c2/dx,-dx/2-1j*c2/dx,1j*c2/dx,-1j*c2/dx])
            else:
                vals=np.array([-beta0,-beta0*pow((delta-dx/2)/delta,2)/2+1j*c2/dx,-beta0*pow((delta-dx/2)/delta,2)/2-1j*c2/dx,1j*c2/dx,-1j*c2/dx])

    return vals


def newton_b(x2,y1):
    # Newton's method taking the semi-axis b as the independent variable to find the optimal solution
    # we now that the function is convex and so, it only have a zero, which will be the optimal solution

    x2=np.abs(x2)
    b=1.5*y1
    for i in range(20):
        f_b=-x2*pow(y1,2)/(2*pow(b,3)*pow(1-pow(y1/b,2),1.5))+1
        df_b=3*x2*pow(y1,2)/2*(1/(pow(b,4)*pow(1-pow(y1/b,2),1.5))+pow(y1,2)/(pow(b,6)*pow(1-pow(y1/b,2),2.5)))

        b=b-f_b/df_b
        if b<=y1:
            b=y1*2

    return pow(x2,2)/(4*(pow(b,2)-pow(y1,2))),x2,pow(x2/2,2)-pow(x2,2)/(4*(1-pow(y1/b,2)))


def ellipse_properties(w,scale):
    # construct the allipse with minimum capacity based on the polygon points given by the function "spectral_dist"
    # ellipse equation: x^2+a12xy+a22y^2+b1x+b2y+c=0

    # points on the ellipse boundary
    Q=np.zeros((len(w),2))
    Q[:,0]=w.real
    Q[:,1]=w.imag
    # print(Q)
    # calculating the ellipse coefficients
    a12=0
    b2=0
    if Q.shape[0]==4:
        a22,b1,c=newton_b(Q[0,0],Q[0,1])
    elif Q.shape[0]==5:
        b1=-Q[1,0]
        c=-pow(Q[0,0],2)-b1*Q[0,0]
        a22=-c/pow(Q[1,1],2)

    # converting ellipse in coefficients in the desire parameters
    gamma,c,d,a=ellipse_parameters(a12,a22,b1,b2,c)
    # print('gamma: ',gamma,', c: ',c,', d: ',d,', a: ',a)

    if isinstance(gamma,float):
        return gamma*scale,c*scale,d[0]*scale,a*scale
        # return gamma,c,d[0],a
    return gamma[0]*scale,c[0]*scale,d[0][0],a*scale
    # return gamma[0],c[0],d[0][0],a


def Faber_approx_coeff(m_max,gamma,c,d):
    coeff=np.zeros(m_max)*mp.exp(0)
    accurate_coeff=1
    for i in range(m_max):
        start=time()
        if accurate_coeff==1:
            coeff[i]=mp.quad(lambda theta: coeff_faber(gamma,theta,c,d,i),[0,1],method='tanh-sinh',error=False,verbose=False,maxdegree=20)
        else:
            coeff[i]=integrate.quad(lambda theta: coeff_faber(gamma,theta,c,d,i),0,1,limit=200,epsrel=pow(10,-16))[0]
        end=time()
        # print('coef[',i,'] time ',end-start)
        if end-start>60: # to avoid coefficients computation take a prohibited amount of time
            accurate_coeff=0

    return coeff
    # return np.array(coeff.tolist(), dtype=float)


def coeff_faber(gamma,theta,c,d,j):
    return (mp.exp((gamma+c*c/(4*gamma))*mp.cos(2*mp.pi*theta)+d+1j*(gamma-c*c/(4*gamma))*mp.sin(2*mp.pi*theta))*mp.exp(-1j*2*mp.pi*j*theta)).real


def Faber_approx(var,m_max,gamma,c,d,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,coeff,ind_source,u_k):

    mat_dim=len(var)
    result=np.zeros((mat_dim,1))#*mp.exp(0)

    c0=d/gamma
    c1=c**2/(4*gamma**2)
    result[:,0]=coeff[0]*var[:,0]

    if ind_source=='H_amplified':
        F1=op.op_H_extended(var,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,u_k)/gamma-c0*var
        def F1_fix(var):
            return op.op_H_extended(var,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,u_k)/gamma-c0*var
    else:
        F1=op.op_H(var,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)/gamma-c0*var
        def F1_fix(var):
            return op.op_H(var,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)/gamma-c0*var

    result[:,0]=result[:,0]+F1[:,0]*coeff[1]

    F2=F1_fix(F1)-2*c1*var
    result[:,0]=result[:,0]+F2[:,0]*coeff[2]

    for i in range(3,m_max):
        F0=F1
        F1=F2
        F2=F1_fix(F1)-c1*F0
        result[:,0]=result[:,0]+F2[:,0]*coeff[i]

    return result


def Faber_ricker_coeff(equ,m_max,t,dt,f0,t0,gamma,c,d):
    coeff=np.zeros(m_max)*mp.exp(0)
    for i in range(m_max):
        start=time()
        if i<10:
            # coeff[i]=mp.quad(lambda theta: (integral_faber_ricker_coeff(equ,f0,t,dt,t0,theta,gamma,c,d)*mp.exp(-1j*i*2*mp.pi*theta)).real,[0,1],method='tanh-sinh',error=False,verbose=False,maxdegree=20)
            coeff[i]=integrate.quad(lambda theta: integral_faber_ricker_coeff(equ,f0,t,dt,t0,theta,gamma,c,d,i),0,1,limit=200,epsrel=pow(10,-16))[0]
            # coeff[i]=integrate.quad(lambda theta: (integral_faber_ricker_coeff(equ,f0,t,dt,t0,theta,gamma,c,d)*np.exp(-1j*i*2*np.pi*theta)).real,0,1/(i+1))[0]
            # for j in range(1,i+1):
            #     coeff[i]=coeff[i]+integrate.quad(lambda theta: (integral_faber_ricker_coeff(equ,f0,t,dt,t0,theta,gamma,c,d)*np.exp(-1j*i*2*np.pi*theta)).real,j/(i+1),(j+1)/(i+1))[0]
        else:
            # coeff[i]=mp.quad(lambda theta: (integral_faber_ricker_coeff(equ,f0,t,dt,t0,theta,gamma,c,d)*mp.exp(-1j*i*2*mp.pi*theta)).real,[0,1],method='tanh-sinh',error=False,verbose=False,maxdegree=20)
            coeff[i]=integrate.quad(lambda theta: integral_faber_ricker_coeff(equ,f0,t,dt,t0,theta,gamma,c,d,i),0,1,limit=200,epsrel=pow(10,-16))[0]
            # coeff[i]=integrate.quad(lambda theta: (integral_faber_ricker_coeff(equ,f0,t,dt,t0,theta,gamma,c,d)*np.exp(-1j*i*2*np.pi*theta)).real,0,1/(i+1))[0]
            # for j in range(1,i+1):
            #     coeff[i]=coeff[i]+integrate.quad(lambda theta: (integral_faber_ricker_coeff(equ,f0,t,dt,t0,theta,gamma,c,d)*np.exp(-1j*i*2*np.pi*theta)).real,j/(i+1),(j+1)/(i+1))[0]
        end=time()
        # print('coef[',i,'] time ',end-start)
        if end-start>60: # to avoid coefficients computation take a prohibited amount of time
            coeff[i]=0
            break

    return coeff


def integral_faber_ricker_coeff(equ,f0,t,dt,t0,theta,gamma,c,d,i):

    theta1=theta*1
    theta=gamma*np.exp(theta*1j*2*np.pi)+d+c**2/(4*gamma)*np.exp(-theta*1j*2*np.pi)

    def first_integral(f0,t,dt,t0,theta):
        # return mp.sqrt(1/(mp.pi*f0**2))/2*mp.exp((theta/(2*mp.pi*f0))**2+(t+dt-t0)*theta)*(mp.erf(theta/(2*mp.pi*f0)+mp.pi*f0*(t+dt-t0))+mp.erf(-theta/(2*mp.pi*f0)+mp.pi*f0*(t0-t)))
        return np.sqrt(1/(np.pi*f0**2))/2*np.exp((theta/(2*np.pi*f0))**2+(t+dt-t0)*theta)*(special.erf(theta/(2*np.pi*f0)+np.pi*f0*(t+dt-t0))+special.erf(-theta/(2*np.pi*f0)+np.pi*f0*(t0-t)))

    if equ=='scalar':
        return 1/(2*(np.pi*f0)**2)*(np.exp(dt*theta-(np.pi*f0*(t-t0))**2)-np.exp(-(np.pi*f0*(t+dt-t0))**2))-theta/(2*(np.pi*f0)**2)*first_integral(f0,t,dt,t0,theta)
    else:
        return (((t+dt-t0-theta/(2*(np.pi*f0)**2))*np.exp(-(np.pi*f0*(t+dt-t0))**2)+(t0-t+theta/(2*(np.pi*f0)**2))*np.exp(dt*theta-(np.pi*f0*(t-t0))**2)-(theta/(np.pi*f0))**2/2*first_integral(f0,t,dt,t0,theta))*np.exp(-1j*i*2*np.pi*theta1)).real
        # return Float((((t+dt-t0-theta/(2*(mp.pi*f0)**2))*mp.exp(-(mp.pi*f0*(t+dt-t0))**2)+(t0-t+theta/(2*(mp.pi*f0)**2))*mp.exp(dt*theta-(mp.pi*f0*(t-t0))**2)-(theta/(mp.pi*f0))**2/2*first_integral(f0,t,dt,t0,theta))*np.exp(-1j*i*2*np.pi*theta1)).real,50)


# High order Runge-Kutta functions

def ssprk_alpha(mu,grau):

    # INPUTS:
    # mu: inverse of the CFL condition
    # grau: degree of the polynomial used to approximate the solution

    # OUTPUT: array with the coefficients \alpha of the SSPRK method

    alpha=np.zeros(grau)*mp.exp(0)
    alpha[0] = mp.exp(0)

    for i in range(1,grau):
        alpha[i]=1/(mu*(i+1))*alpha[i-1]
        alpha[1:i]=1/(mu*np.array(range(1,i)))*alpha[:(i - 1)]
        alpha[0]=1-np.sum(alpha[1:(i+1)])

    return alpha


def RK_op(var,mu,dt,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx,ny,grau,u_k,ind_source='H_amplified'):

    alpha=np.array(ssprk_alpha(mu,grau).tolist(),dtype=np.float_)
    approx = var*alpha[0]

    aux=var#*mp.exp(0)

    for i in range(1,grau-1):
        if ind_source=='H_amplified':
            aux=aux+mu*dt*op.op_H_extended(aux,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx,ny,u_k)
        else:
            aux=aux+mu*dt*op.op_H(aux,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx,ny)
        approx=approx+aux*alpha[i]

    if ind_source=='H_amplified':
        aux=aux+mu*dt*op.op_H_extended(aux,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx,ny,u_k)
        aux=aux+mu*dt*op.op_H_extended(aux,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx,ny,u_k)
    else:
        aux=aux+mu*dt*op.op_H(aux,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx,ny)
        aux=aux+mu*dt*op.op_H(aux,equ,dim,free_surf,delta,beta0,ord,dx,c2,nx,ny)
    approx=approx+aux*alpha[grau-1]

    return approx


# Krylov function

# def krylov_op(var,degree,dt,equ,dim,delta,beta0,ord,dx,param,nx,ny,u_k):
#
#     Vm=np.zeros((len(var),degree+1))
#     Vm[:,0]=var[:,0]/np.linalg.norm(var[:,0],2)
#
#     Hm=np.zeros((degree+1,degree+1))
#
#     for i in range(degree+1):
#         start=time()
#         w=op.op_H_extended(np.expand_dims(Vm[:,i],axis=1),equ,dim,delta,beta0,ord,dx,param,nx,ny,u_k)*dt
#         print("time 1:",time()-start)
#         start=time()
#         for j in range(i+1):
#             Hm[j,i]=sum(w*np.expand_dims(Vm[:,i],axis=1))
#             w=w-Hm[j,i]*np.expand_dims(Vm[:,i],axis=1)
#         if i<degree:
#             h_aux=np.linalg.norm(w,2)
#             Vm[:,i+1]=w[:,0]/h_aux
#         print("time 2:",time()-start)
#
#     sdafasd
#     return np.linalg.norm(var[:,0],2)*Vm.dot(np.expand_dims(linalg.expm(Hm)[:,0],axis=1))


def krylov_op(var,degree,dt,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,u_k):

    Vm=np.zeros((len(var),degree+1))
    Vm[:,0]=var[:,0]/np.linalg.norm(var[:,0],2)

    Hm=np.zeros((degree+1,degree+1))

    for i in range(degree+1):
        w=op.op_H_extended(np.expand_dims(Vm[:,i],axis=1),equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,u_k)[:,0]*dt
        for j in range(i+1):
            Hm[j,i]=np.dot(w,Vm[:,j])
            w=w-Hm[j,i]*Vm[:,j]
        if i<degree:
            Hm[i+1,i]=np.linalg.norm(w,2)
            Vm[:,i+1]=w/Hm[i+1,i]

    try:
        return np.linalg.norm(var[:,0],2)*Vm.dot(np.expand_dims(linalg.expm(Hm)[:,0],axis=1))
    except:
        # return np.linalg.norm(var[:,0],2)*Vm.dot(np.expand_dims(linalg.expm(Hm*0)[:,0],axis=1))
        return var*float("nan")


# ---------------------------------------------------------------------------
# auxiliary function of the high-order methods to construct the amplified matrix
# ---------------------------------------------------------------------------

def g_approx(f,p,f0,t0,t,source_type,uk_core):
    # INPUT:
    # f: spatial part of the source function
    # p-1: degree of approximation of the source (p==2 only, for the exact representation of u_k)
    # f0, t0: parameters of the source function
    # t: time variable for the time part of the sourc funtion

    # OUTPUT: first p+1 vectors u_k approximating ethe source term

    uk=np.zeros((len(f[:,0]),p))

    if source_type=='11':
        uk[:,0]=np.exp(t)*f[:,0]+(1+t-np.exp(t))*f[:,1]
        uk[:,1]=np.exp(t)*f[:,0]+(1-np.exp(t))*f[:,1]
        for i in range(2,p):
            uk[:,i]=np.exp(t)*(f[:,0]-f[:,1])
    elif source_type==1:
        uk[:,0]=np.exp(t)*f[:,0]+(1+t-np.exp(t))*(f[:,1]+f[:,2])
        uk[:,1]=np.exp(t)*f[:,0]+(1-np.exp(t))*(f[:,1]+f[:,2])
        for i in range(2,p):
            uk[:,i]=np.exp(t)*(f[:,0]-(f[:,1]+f[:,2]))
    elif source_type[0]=='1' and source_type[-1]=='1':
        for i in range(p):
            if i%2==0:
                if i%4==0:
                    uk[:,i]=f[:,0]*np.cos(f[:,3]+t)+f[:,1]*np.sin(f[:,3]+t)
                else:
                    uk[:,i]=-(f[:,0]*np.cos(f[:,3]+t)+f[:,1]*np.sin(f[:,3]+t))
            else:
                if i%4==1:
                    uk[:,i]=-f[:,0]*np.sin(f[:,3]+t)+f[:,1]*np.cos(f[:,3]+t)
                else:
                    uk[:,i]=f[:,0]*np.sin(f[:,3]+t)-f[:,1]*np.cos(f[:,3]+t)
    elif source_type[0]=='2' and source_type[-1]=='1':
        aux_trig=np.zeros((len(f[:,0]),4))
        aux_trig[:,0]=np.cos(8*t-5*f[:,7]+f[:,6])
        aux_trig[:,1]=np.cos(6*t-5*f[:,7]-f[:,6])
        aux_trig[:,2]=np.sin(8*t-5*f[:,7]+f[:,6])
        aux_trig[:,3]=np.sin(6*t-5*f[:,7]-f[:,6])
        for i in range(p):
            if i%2==0:
                if i%4==0:
                    uk[:,i]=-f[:,2]*f[:,5]*(3*pow(6,i)*aux_trig[:,1]+4*pow(8,i)*aux_trig[:,0])\
                            -f[:,8]*((f[:,0]*(pow(8,i)*aux_trig[:,0]/16+pow(6,i)*aux_trig[:,1]/12)-2*f[:,1]*(pow(8,i)*aux_trig[:,2]/16-pow(6,i)*aux_trig[:,3]/12)-f[:,2]*(pow(8,i)*aux_trig[:,0]/16+pow(6,i)*aux_trig[:,1]/12))*f[:,5]
                                     +(f[:,3]*(pow(8,i)*aux_trig[:,0]/16+pow(6,i)*aux_trig[:,1]/12)+10*f[:,4]*(pow(8,i)*aux_trig[:,2]/16+pow(6,i)*aux_trig[:,3]/12)-25*f[:,5]*(pow(8,i)*aux_trig[:,0]/16+pow(6,i)*aux_trig[:,1]/12))*f[:,2])
                else:
                    uk[:,i]=f[:,2]*f[:,5]*(3*pow(-6,i)*aux_trig[:,1]+4*pow(-8,i)*aux_trig[:,0])\
                            -f[:,8]*((-f[:,0]*(pow(8,i)*aux_trig[:,0]/16+pow(6,i)*aux_trig[:,1]/12)+2*f[:,1]*(pow(8,i)*aux_trig[:,2]/16-pow(6,i)*aux_trig[:,3]/12)+f[:,2]*(pow(8,i)*aux_trig[:,0]/16+pow(6,i)*aux_trig[:,1]/12))*f[:,5]
                                     +(-f[:,3]*(pow(8,i)*aux_trig[:,0]/16+pow(6,i)*aux_trig[:,1]/12)-10*f[:,4]*(pow(8,i)*aux_trig[:,2]/16+pow(6,i)*aux_trig[:,3]/12)+25*f[:,5]*(pow(8,i)*aux_trig[:,0]/16+pow(6,i)*aux_trig[:,1]/12))*f[:,2])
            else:
                if i%4==1:
                    uk[:,i]=f[:,2]*f[:,5]*(3*pow(-6,i)*aux_trig[:,3]+4*pow(-8,i)*aux_trig[:,2])\
                            -f[:,8]*((-f[:,0]*(pow(8,i)*aux_trig[:,2]/16+pow(6,i)*aux_trig[:,3]/12)-2*f[:,1]*(pow(8,i)*aux_trig[:,0]/16-pow(6,i)*aux_trig[:,1]/12)+f[:,2]*(pow(8,i)*aux_trig[:,2]/16+pow(6,i)*aux_trig[:,3]/12))*f[:,5]
                                     +(-f[:,3]*(pow(8,i)*aux_trig[:,2]/16+pow(6,i)*aux_trig[:,3]/12)+10*f[:,4]*(pow(8,i)*aux_trig[:,0]/16+pow(6,i)*aux_trig[:,1]/12)+25*f[:,5]*(pow(8,i)*aux_trig[:,2]/16+pow(6,i)*aux_trig[:,3]/12))*f[:,2])
                else:
                    uk[:,i]=-f[:,2]*f[:,5]*(3*pow(-6,i)*aux_trig[:,3]+4*pow(-8,i)*aux_trig[:,2])\
                            -f[:,8]*((f[:,0]*(pow(8,i)*aux_trig[:,2]/16+pow(6,i)*aux_trig[:,3]/12)-2*f[:,1]*(pow(8,i)*aux_trig[:,0]/16-pow(6,i)*aux_trig[:,1]/12)-f[:,2]*(pow(8,i)*aux_trig[:,2]/16+pow(6,i)*aux_trig[:,3]/12))*f[:,5]
                                     +(f[:,3]*(pow(8,i)*aux_trig[:,2]/16+pow(6,i)*aux_trig[:,3]/12)-10*f[:,4]*(pow(8,i)*aux_trig[:,0]/16+pow(6,i)*aux_trig[:,1]/12)-25*f[:,5]*(pow(8,i)*aux_trig[:,2]/16+pow(6,i)*aux_trig[:,3]/12))*f[:,2])
    elif source_type!="8" and source_type[-1]!="S":
        uk[:,0]=(t-t0)*np.exp(-pow(np.pi*f0*(t-t0),2))*f[:,0]
        uk[:,1]=(1-2*pow(np.pi*f0*(t-t0),2))*np.exp(-pow(np.pi*f0*(t-t0),2))*f[:,0]
        uk[:,2]=(-6*pow(np.pi*f0,2)*(t-t0)+4*pow(np.pi*f0*(t-t0),3)*np.pi*f0)*np.exp(-pow(np.pi*f0*(t-t0),2))*f[:,0]
        uk[:,3]=(-6*pow(np.pi*f0,2)+24*pow(np.pi*f0,4)*pow(t-t0,2)-8*pow(np.pi*f0,6)*pow(t-t0,4))*np.exp(-pow(np.pi*f0*(t-t0),2))*f[:,0]
        for i in range(4,p):
            uk[:,i]=uk_core[i-4](t)*f[:,0]
    elif source_type!="8" and source_type[-1]=="S":
        uk[:,0]=(1-2*pow(np.pi*f0*(t-t0),2))*np.exp(-pow(np.pi*f0*(t-t0),2))*f[:,0]
        uk[:,1]=(-6*pow(np.pi*f0,2)*(t-t0)+4*pow(np.pi*f0*(t-t0),3)*np.pi*f0)*np.exp(-pow(np.pi*f0*(t-t0),2))*f[:,0]
        uk[:,2]=(-6*pow(np.pi*f0,2)+24*pow(np.pi*f0,4)*pow(t-t0,2)-8*pow(np.pi*f0,6)*pow(t-t0,4))*np.exp(-pow(np.pi*f0*(t-t0),2))*f[:,0]
        for i in range(3,p):
            uk[:,i]=uk_core[i-3](t)*f[:,0]

    return uk


def g_core(p,f0,t0,source_type):
    if len(str(source_type))>1:  # necessary condition to avoid source_type=1
        if source_type[-1]=='X':
            uk_core=np.zeros(p-4,dtype=sym.Symbol)
            t_var=sym.Symbol('t_var')
            aux=(-6*pow(sym.pi*f0,2)+24*pow(sym.pi*f0,4)*pow(t_var-t0,2)-8*pow(sym.pi*f0,6)*pow(t_var-t0,4))*sym.exp(-pow(sym.pi*f0*(t_var-t0),2))
            for i in range(p-4):
                aux=aux.diff(t_var)
                uk_core[i]=sym.lambdify(t_var,aux)
            return uk_core
        elif source_type[-1]=='S':
            uk_core=np.zeros(p-3,dtype=sym.Symbol)
            t_var=sym.Symbol('t_var')
            aux=(-6*pow(sym.pi*f0,2)+24*pow(sym.pi*f0,4)*pow(t_var-t0,2)-8*pow(sym.pi*f0,6)*pow(t_var-t0,4))*sym.exp(-pow(sym.pi*f0*(t_var-t0),2))
            for i in range(p-3):
                aux=aux.diff(t_var)
                uk_core[i]=sym.lambdify(t_var,aux)
            return uk_core
    return 0


# time part of the source term function

def source_xt(f,t,param_ricker,source_type):
    # INPUTS:
    # t: time to evaluate the source term
    # f0: mean frequency of the Ricker's source
    # t0: time delay of the source

    # OUTPUT: evaluation of the temporal term of the source at time t
    if source_type=='11':
        return np.expand_dims(f[:,0]*np.exp(t)+f[:,1]*(1+t-np.exp(t)),axis=1)
    elif source_type=='11_2MS':
        return np.expand_dims(f[:,0]*np.exp(t)+f[:,1]*(1-np.exp(t)),axis=1)
    elif source_type==1:
        return np.expand_dims(f[:,0]*np.exp(t)+(f[:,1]+f[:,2])*(1+t-np.exp(t)),axis=1)
    elif source_type=='1_2MS':
        return np.expand_dims(f[:,0]*np.exp(t)+(f[:,1]+f[:,2])*(1-np.exp(t)),axis=1)
    elif source_type[0]=='1' and source_type[-1]=='1':
        return np.expand_dims(f[:,0]*np.cos(f[:,3]+t)+f[:,1]*np.sin(f[:,3]+t),axis=1)
    elif source_type[0]=='1' and source_type[-5:]=='1_2MS':
        return np.expand_dims(-f[:,0]*np.sin(f[:,3]+t)+f[:,1]*np.cos(f[:,3]+t),axis=1)
    elif source_type[0]=='2' and source_type[-1]=='1':
        int1=np.cos(8*t-5*f[:,7]+f[:,6])/16+np.cos(6*t-5*f[:,7]-f[:,6])/12
        int2=np.sin(8*t-5*f[:,7]+f[:,6])/16-np.sin(6*t-5*f[:,7]-f[:,6])/12
        int3=np.sin(8*t-5*f[:,7]+f[:,6])/16+np.sin(6*t-5*f[:,7]-f[:,6])/12
        return np.expand_dims(f[:,2]*f[:,5]*(-np.sin(f[:,6]+t)*np.sin(5*f[:,7]-7*t)-7*np.cos(f[:,6]+t)*np.cos(5*f[:,7]-7*t))-
                              f[:,8]*((f[:,0]*int1-2*f[:,1]*int2-f[:,2]*int1)*f[:,5]+(f[:,3]*int1+10*f[:,4]*int3-25*f[:,5]*int1)*f[:,2]),axis=1)
    elif source_type[0]=='2' and source_type[-5:]=='1_2MS':
        deriv1=np.cos(f[:,6]+t)*np.sin(5*f[:,7]-7*t)
        deriv2=np.sin(f[:,6]+t)*np.sin(5*f[:,7]-7*t)
        deriv3=np.cos(f[:,6]+t)*np.cos(5*f[:,7]-7*t)
        return np.expand_dims(f[:,2]*f[:,5]*(-50*np.cos(f[:,6]+t)*np.sin(5*f[:,7]-7*t)+14*np.sin(f[:,6]+t)*np.cos(5*f[:,7]-7*t))-
                              f[:,8]*((f[:,0]*deriv1-2*f[:,1]*deriv2-f[:,2]*deriv1)*f[:,5]+(f[:,3]*deriv1+10*f[:,4]*deriv3-25*f[:,5]*deriv1)*f[:,2]),axis=1)
    elif source_type[-1]!='S':
        return f*(t-param_ricker[1])*np.exp(-pow(np.pi*param_ricker[0]*(t-param_ricker[1]),2))

    return f*(1-2*pow(np.pi*param_ricker[0]*(t-param_ricker[1]),2))*np.exp(-pow(np.pi*param_ricker[0]*(t-param_ricker[1]),2))

