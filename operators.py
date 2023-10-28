import numpy as np


def op_H(var,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny):

    result=np.zeros((len(var),1))
    if equ=='scalar':
        if dim==1:
            aux=(nx-1)

            dpv1=dpx_1D(var[aux:2*aux],dx,pos=1,ord=ord,nx=nx)

            # scalar wave field equations u
            result[:aux]=param*dpv1

            # velocity equation v_1
            result[aux:2*aux]=dpx_1D(var[:aux],dx,pos=2,ord=ord,nx=nx)

            if delta!=0:

                # scalar wave field equations u
                result[:aux]=result[:aux]-param*var[2*aux:]

                # velocity equations
                result[aux:2*aux]=result[aux:2*aux]-beta_i(dim,dx,nx,ny,delta,beta0,1,1)*var[aux:2*aux]

                # auxiliari variables equations
                result[2*aux:]=beta_i(dim,dx,nx,ny,delta,beta0,1,2)*(dpv1-var[2*aux:])

        elif dim==2:
            aux=(nx-1)*(ny-1)

            dpv1=dpx_2D(var[aux:2*aux],dx,1,1,ord,nx,ny)
            dpv2=dpx_2D(var[2*aux:3*aux],dx,2,1,ord,nx,ny)

            # scalar wave field equations u
            result[:aux]=param*(dpv1+dpv2)

            # velocity equations v_i
            result[aux:2*aux]=dpx_2D(var[:aux],dx,1,2,ord,nx,ny)
            result[2*aux:3*aux]=dpx_2D(var[:aux],dx,2,2,ord,nx,ny)

            if delta!=0:

                # scalar wave field equations u
                result[:aux]=result[:aux]-param*(var[3*aux:4*aux]+var[4*aux:])

                # velocity equations
                result[aux:2*aux]=result[aux:2*aux]-beta_i(dim,dx,nx,ny,delta,beta0,1,1)*var[aux:2*aux]
                result[2*aux:3*aux]=result[2*aux:3*aux]-beta_i(dim,dx,nx,ny,delta,beta0,2,1)*var[2*aux:3*aux]

                # auxiliari variables equations
                result[3*aux:4*aux]=beta_i(dim,dx,nx,ny,delta,beta0,1,2)*(dpv1-var[3*aux:4*aux])
                result[4*aux:]=beta_i(dim,dx,nx,ny,delta,beta0,2,2)*(dpv2-var[4*aux:])
    elif equ=='scalar_dx2':
        if dim==1:
            aux=(nx-1)

            # scalar wave field equations u
            result[:aux]=var[aux:2*aux]

            # velocity equation v_1
            result[aux:2*aux]=param*dp2x_1D(var[:aux],dx,ord)

            if delta!=0:

                # velocity equations
                result[aux:2*aux]=result[aux:2*aux]-beta_i(dim,dx,nx,ny,delta,beta0,1,0)*var[aux:2*aux]+param*dpx_1D(var[2*aux:3*aux],dx,1,ord,nx)

                # auxiliari variables equations
                result[2*aux:]=-beta_i(dim,dx,nx,ny,delta,beta0,1,1)*(dpx_1D(var[:aux],dx,pos=2,ord=ord,nx=nx)+var[2*aux:])

        elif dim==2:
            aux=(nx-1)*(ny-1)

            # scalar wave field equations u
            result[:aux]=var[aux:2*aux]

            # velocity equations v_i
            result[aux:2*aux]=param*(dp2x_2D(var[:aux],dx,1,ord,nx,ny,free_surf)+dp2x_2D(var[:aux],dx,2,ord,nx,ny,free_surf))

            if delta!=0:

                beta10=beta_i(dim,dx,nx,ny,delta,beta0,1,2,free_surf)
                beta11=beta_i(dim,dx,nx,ny,delta,beta0,1,1,free_surf)
                beta20=beta_i(dim,dx,nx,ny,delta,beta0,2,2,free_surf)
                beta21=beta_i(dim,dx,nx,ny,delta,beta0,2,1,free_surf)

                # velocity equations v_i
                result[aux:2*aux]=result[aux:2*aux]+param*(dpx_2D(var[2*aux:3*aux],dx,1,1,ord,nx,ny,free_surf)+dpx_2D(var[3*aux:4*aux],dx,2,1,ord,nx,ny,free_surf))-(beta10+beta20)*var[aux:2*aux]-beta10*beta20*var[:aux]

                # auxiliari variables equations
                result[2*aux:3*aux]=-beta11*var[2*aux:3*aux]+(beta20-beta11)*dpx_2D(var[:aux],dx,1,2,ord,nx,ny,free_surf)
                result[3*aux:]=-beta21*var[3*aux:4*aux]+(beta10-beta21)*dpx_2D(var[:aux],dx,2,2,ord,nx,ny,free_surf)
    elif equ=='elastic':
        # in the elastic case, 1D is equal to 1D acoustic, so we don't consider 1D elastic

        if dim==2:
            aux=(nx-1)*(ny-1)

            # relations of displacement and velocity
            result[:aux]=var[2*aux:3*aux]
            result[aux:2*aux]=var[3*aux:4*aux]

            # elastic wave field equations of velocity-stress
            result[2*aux:3*aux]=np.expand_dims(np.reciprocal(param[:,0]),axis=1)*(dpx_2D(var[4*aux:5*aux],dx,1,1,ord,nx,ny)+dpx_2D(var[5*aux:6*aux],dx,2,1,ord,nx,ny))
            result[3*aux:4*aux]=np.expand_dims(np.reciprocal(param[:,1]),axis=1)*(dpx_2D(var[5*aux:6*aux],dx,1,2,ord,nx,ny)+dpx_2D(var[6*aux:7*aux],dx,2,2,ord,nx,ny))

            # stress equations T_{ij}
            result[4*aux:5*aux]=np.expand_dims(2*param[:,2]+param[:,4],axis=1)*dpx_2D(var[2*aux:3*aux],dx,1,2,ord,nx,ny)+np.expand_dims(param[:,4],axis=1)*dpx_2D(var[3*aux:4*aux],dx,2,1,ord,nx,ny)
            result[5*aux:6*aux]=np.expand_dims(param[:,3],axis=1)*(dpx_2D(var[2*aux:3*aux],dx,2,2,ord,nx,ny)+dpx_2D(var[3*aux:4*aux],dx,1,1,ord,nx,ny))
            result[6*aux:7*aux]=np.expand_dims(param[:,4],axis=1)*dpx_2D(var[2*aux:3*aux],dx,1,2,ord,nx,ny)+np.expand_dims(2*param[:,2]+param[:,4],axis=1)*dpx_2D(var[3*aux:4*aux],dx,2,1,ord,nx,ny)

            if delta!=0:

                beta10=beta_i(dim,dx,nx,ny,delta,beta0,1,0)
                beta11=beta_i(dim,dx,nx,ny,delta,beta0,1,1)
                beta20=beta_i(dim,dx,nx,ny,delta,beta0,2,0)
                beta21=beta_i(dim,dx,nx,ny,delta,beta0,2,1)

                # elastic wave field equations of velocity-stress
                result[2*aux:3*aux]=result[2*aux:3*aux]-(beta10+beta20)*var[2*aux:3*aux]-beta10*beta20*var[:aux]
                result[3*aux:4*aux]=result[3*aux:4*aux]-(beta11+beta21)*var[3*aux:4*aux]-beta11*beta21*var[aux:2*aux]

                # auxiliari variables equations
                result[7*aux:8*aux]=-beta11*var[7*aux:8*aux]+(beta20-beta11)*np.expand_dims(2*param[:,2]+param[:,4],axis=1)*dpx_2D(var[:aux],dx,1,2,ord,nx,ny)
                result[8*aux:9*aux]=-beta21*var[8*aux:9*aux]+(beta10-beta21)*np.expand_dims(param[:,3],axis=1)*dpx_2D(var[:aux],dx,2,2,ord,nx,ny)
                result[9*aux:10*aux]=-beta10*var[9*aux:10*aux]+(beta21-beta10)*np.expand_dims(param[:,3],axis=1)*dpx_2D(var[aux:2*aux],dx,1,1,ord,nx,ny)
                result[10*aux:11*aux]=-beta20*var[10*aux:11*aux]+(beta11-beta20)*np.expand_dims(2*param[:,2]+param[:,4],axis=1)*dpx_2D(var[aux:2*aux],dx,2,1,ord,nx,ny)

    return result


def op_H_extended(var,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny,u_k):

    result=np.zeros((len(var),1))

    p=u_k.shape[1]
    result[:-p]=op_H(var[:-p],equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny)
    for i in range(p):
        if var[len(var)-1-i]!=0:
            result[:(len(var)-p),0]=result[:(len(var)-p),0]+u_k[:,i]*var[len(var)-1-i]
        if i>0:
            result[len(var)-1-i]=var[len(var)-i]
    return result


def op_H_2ord(var,equ,dim,free_surf,delta,beta0,ord,dx,param,nx,ny):
    result=np.zeros((len(var),1))
    if equ=='scalar_dx2':
        if dim==1:
            aux=(nx-1)

            # scalar wave field equations u
            result[:aux]=param*dp2x_1D(var[:aux],dx,ord)

            if delta!=0:

                # scalar wave field equations u
                result[:aux]=result[:aux]+param*dpx_1D(var[aux:2*aux],dx,1,ord,nx)

                # auxiliar variable
                result[aux:2*aux]=-beta_i(dim,dx,nx,ny,delta,beta0,1,1)*(var[aux:2*aux]+dpx_1D(var[:aux],dx,2,ord,nx))

        elif dim==2:
            aux=(nx-1)*(ny-1)

            # scalar wave field equations u
            result[:aux]=param*(dp2x_2D(var[:aux],dx,1,ord,nx,ny,free_surf)+dp2x_2D(var[:aux],dx,2,ord,nx,ny,free_surf))

            if delta!=0:

                beta10=beta_i(dim,dx,nx,ny,delta,beta0,1,2,free_surf)
                beta11=beta_i(dim,dx,nx,ny,delta,beta0,1,1,free_surf)
                beta20=beta_i(dim,dx,nx,ny,delta,beta0,2,2,free_surf)
                beta21=beta_i(dim,dx,nx,ny,delta,beta0,2,1,free_surf)

                # scalar wave field equations u
                result[:aux]=result[:aux]-beta10*beta20*var[:aux]+param*(dpx_2D(var[aux:2*aux],dx,1,1,ord,nx,ny,free_surf)+dpx_2D(var[2*aux:3*aux],dx,2,1,ord,nx,ny,free_surf))

                # auxiliar variables
                result[aux:2*aux]=-beta11*var[aux:2*aux]+(beta20-beta11)*dpx_2D(var[:aux],dx,1,2,ord,nx,ny,free_surf)
                result[2*aux:3*aux]=-beta21*var[2*aux:3*aux]+(beta10-beta21)*dpx_2D(var[:aux],dx,2,2,ord,nx,ny,free_surf)

    return result


def dp2x_1D(var,dx,ord):
    result=var*0
    if ord=='4':
        result[0]=-5/2*var[0]+4/3*var[1]-1/12*var[2]
        result[1]=-5/2*var[1]+4/3*(var[2]+var[0])-1/12*var[3]
        result[2:-2]=-5/2*var[2:-2]+4/3*(var[3:-1]+var[1:-3])-1/12*(var[4:]+var[:-4])
        result[-2]=-5/2*var[-2]+4/3*(var[-3]+var[-1])-1/12*var[-4]
        result[-1]=-5/2*var[-1]+4/3*var[-2]-1/12*var[-3]
    elif ord[0]=='8':
        if ord[-1]=='0' or ord[-1]=='8':
            coeff=np.array([-205/72,8/5,-1/5,8/315,-1/560])
        elif ord[-1]=='1':
            coeff=np.array([-2.9681109,1.70000993,-0.25546155,0.04445392,-0.00494685])
        elif ord[-1]=='2':
            coeff=np.array([-2.94455921,1.67982573,-0.24303362,0.03936426,-0.00387677])

        result[0]=coeff[0]*var[0]+coeff[1]*var[1]+coeff[2]*var[2]+coeff[3]*var[3]+coeff[4]*var[4]
        result[1]=coeff[0]*var[1]+coeff[1]*(var[2]+var[0])+coeff[2]*var[3]+coeff[3]*var[4]+coeff[4]*var[5]
        result[2]=coeff[0]*var[2]+coeff[1]*(var[3]+var[1])+coeff[2]*(var[4]+var[0])+coeff[3]*var[5]+coeff[4]*var[6]
        result[3]=coeff[0]*var[3]+coeff[1]*(var[4]+var[2])+coeff[2]*(var[5]+var[1])+coeff[3]*(var[6]+var[0])+coeff[4]*var[7]
        result[4:-4]=coeff[0]*var[4:-4]+coeff[1]*(var[5:-3]+var[3:-5])+coeff[2]*(var[6:-2]+var[2:-6])+coeff[3]*(var[7:-1]+var[1:-7])+coeff[4]*(var[8:]+var[:-8])
        result[-4]=coeff[0]*var[-4]+coeff[1]*(var[-3]+var[-5])+coeff[2]*(var[-2]+var[-6])+coeff[3]*(var[-1]+var[-7])+coeff[4]*(var[-8])
        result[-3]=coeff[0]*var[-3]+coeff[1]*(var[-2]+var[-4])+coeff[2]*(var[-1]+var[-5])+coeff[3]*(var[-6])+coeff[4]*(var[-7])
        result[-2]=coeff[0]*var[-2]+coeff[1]*(var[-1]+var[-3])+coeff[2]*(var[-4])+coeff[3]*(var[-5])+coeff[4]*(var[-6])
        result[-1]=coeff[0]*var[-1]+coeff[1]*(var[-2])+coeff[2]*(var[-3])+coeff[3]*(var[-4])+coeff[4]*(var[-5])
    return result/dx**2


def dp2x_2D(var,dx,dir,ord,nx,ny,free_surf):
    result=var*0
    if ord=='4':
        if dir==1:
            result[:(ny-1)]=-5/2*var[:(ny-1)]+4/3*var[(ny-1):2*(ny-1)]-1/12*var[2*(ny-1):3*(ny-1)]
            result[(ny-1):2*(ny-1)]=-5/2*var[(ny-1):2*(ny-1)]+4/3*(var[2*(ny-1):3*(ny-1)]+var[:(ny-1)])-1/12*var[3*(ny-1):4*(ny-1)]
            result[2*(ny-1):(nx-3)*(ny-1)]=-5/2*var[2*(ny-1):(nx-3)*(ny-1)]+4/3*(var[3*(ny-1):(nx-2)*(ny-1)]+var[(ny-1):(nx-4)*(ny-1)])-1/12*(var[4*(ny-1):(nx-1)*(ny-1)]+var[:(nx-5)*(ny-1)])
            result[(nx-3)*(ny-1):(nx-2)*(ny-1)]=-5/2*var[(nx-3)*(ny-1):(nx-2)*(ny-1)]+4/3*(var[(nx-2)*(ny-1):(nx-1)*(ny-1)]+var[(nx-4)*(ny-1):(nx-3)*(ny-1)])-1/12*(var[(nx-5)*(ny-1):(nx-4)*(ny-1)])
            result[(nx-2)*(ny-1):(nx-1)*(ny-1)]=-5/2*var[(nx-2)*(ny-1):(nx-1)*(ny-1)]+4/3*(var[(nx-3)*(ny-1):(nx-2)*(ny-1)])-1/12*(var[(nx-4)*(ny-1):(nx-3)*(ny-1)])
        else:
            result[2:(ny-1)*(nx-1)-2]=-5/2*var[2:(ny-1)*(nx-1)-2]+4/3*(var[1:(ny-1)*(nx-1)-3]+var[3:(ny-1)*(nx-1)-1])-1/12*(var[:(ny-1)*(nx-1)-4]+var[4:(ny-1)*(nx-1)])
            result[range(0,(nx-1)*(ny-1),ny-1)]=-5/2*var[range(0,(nx-1)*(ny-1),ny-1)]+4/3*(var[range(1,(nx-1)*(ny-1),ny-1)])-1/12*(var[range(2,(nx-1)*(ny-1),ny-1)])
            result[range(1,(nx-1)*(ny-1),ny-1)]=-5/2*var[range(1,(nx-1)*(ny-1),ny-1)]+4/3*(var[range(2,(nx-1)*(ny-1),ny-1)]+var[range(0,(nx-1)*(ny-1),ny-1)])-1/12*(var[range(3,(nx-1)*(ny-1),ny-1)])
            result[range(ny-3,(nx-1)*(ny-1),ny-1)]=-5/2*var[range(ny-3,(nx-1)*(ny-1),ny-1)]+4/3*(var[range(ny-2,(nx-1)*(ny-1),ny-1)]+var[range(ny-4,(nx-1)*(ny-1),ny-1)])-1/12*var[range(ny-5,(nx-1)*(ny-1),ny-1)]
            result[range(ny-2,(nx-1)*(ny-1),ny-1)]=-5/2*var[range(ny-2,(nx-1)*(ny-1),ny-1)]+4/3*(var[range(ny-3,(nx-1)*(ny-1),ny-1)])-1/12*var[range(ny-4,(nx-1)*(ny-1),ny-1)]
    elif ord[0]=='8':
        if ord[-1]=='0' or ord[-1]=='8':
            coeff=np.array([-205/72,8/5,-1/5,8/315,-1/560])
        elif ord[-1]=='1':
            coeff=np.array([-2.9681109,1.70000993,-0.25546155,0.04445392,-0.00494685])
        elif ord[-1]=='2':
            coeff=np.array([-2.94455921,1.67982573,-0.24303362,0.03936426,-0.00387677])
        if dir==1:
            result[:(ny-1)]=coeff[0]*var[:(ny-1)]+coeff[1]*var[(ny-1):2*(ny-1)]+coeff[2]*var[2*(ny-1):3*(ny-1)]+coeff[3]*var[3*(ny-1):4*(ny-1)]+coeff[4]*var[4*(ny-1):5*(ny-1)]
            result[(ny-1):2*(ny-1)]=coeff[0]*var[(ny-1):2*(ny-1)]+coeff[1]*(var[2*(ny-1):3*(ny-1)]+var[:(ny-1)])+coeff[2]*var[3*(ny-1):4*(ny-1)]+coeff[3]*var[4*(ny-1):5*(ny-1)]+coeff[4]*var[5*(ny-1):6*(ny-1)]
            result[2*(ny-1):3*(ny-1)]=coeff[0]*var[2*(ny-1):3*(ny-1)]+coeff[1]*(var[3*(ny-1):4*(ny-1)]+var[(ny-1):2*(ny-1)])+coeff[2]*(var[4*(ny-1):5*(ny-1)]+var[:(ny-1)])+coeff[3]*var[5*(ny-1):6*(ny-1)]+coeff[4]*var[6*(ny-1):7*(ny-1)]
            result[3*(ny-1):4*(ny-1)]=coeff[0]*var[3*(ny-1):4*(ny-1)]+coeff[1]*(var[4*(ny-1):5*(ny-1)]+var[2*(ny-1):3*(ny-1)])+coeff[2]*(var[5*(ny-1):6*(ny-1)]+var[(ny-1):2*(ny-1)])+coeff[3]*(var[6*(ny-1):7*(ny-1)]+var[:(ny-1)])+coeff[4]*var[7*(ny-1):8*(ny-1)]
            result[4*(ny-1):(nx-5)*(ny-1)]=coeff[0]*var[4*(ny-1):(nx-5)*(ny-1)]+coeff[1]*(var[5*(ny-1):(nx-4)*(ny-1)]+var[3*(ny-1):(nx-6)*(ny-1)])+coeff[2]*(var[6*(ny-1):(nx-3)*(ny-1)]+var[2*(ny-1):(nx-7)*(ny-1)])+coeff[3]*(var[7*(ny-1):(nx-2)*(ny-1)]+var[(ny-1):(nx-8)*(ny-1)])+coeff[4]*(var[8*(ny-1):(nx-1)*(ny-1)]+var[:(nx-9)*(ny-1)])
            result[(nx-5)*(ny-1):(nx-4)*(ny-1)]=coeff[0]*var[(nx-5)*(ny-1):(nx-4)*(ny-1)]+coeff[1]*(var[(nx-4)*(ny-1):(nx-3)*(ny-1)]+var[(nx-6)*(ny-1):(nx-5)*(ny-1)])+coeff[2]*(var[(nx-3)*(ny-1):(nx-2)*(ny-1)]+var[(nx-7)*(ny-1):(nx-6)*(ny-1)])+coeff[3]*(var[(nx-2)*(ny-1):(nx-1)*(ny-1)]+var[(nx-8)*(ny-1):(nx-7)*(ny-1)])+coeff[4]*(var[(nx-9)*(ny-1):(nx-8)*(ny-1)])
            result[(nx-4)*(ny-1):(nx-3)*(ny-1)]=coeff[0]*var[(nx-4)*(ny-1):(nx-3)*(ny-1)]+coeff[1]*(var[(nx-3)*(ny-1):(nx-2)*(ny-1)]+var[(nx-5)*(ny-1):(nx-4)*(ny-1)])+coeff[2]*(var[(nx-2)*(ny-1):(nx-1)*(ny-1)]+var[(nx-6)*(ny-1):(nx-5)*(ny-1)])+coeff[3]*(var[(nx-7)*(ny-1):(nx-6)*(ny-1)])+coeff[4]*(var[(nx-8)*(ny-1):(nx-7)*(ny-1)])
            result[(nx-3)*(ny-1):(nx-2)*(ny-1)]=coeff[0]*var[(nx-3)*(ny-1):(nx-2)*(ny-1)]+coeff[1]*(var[(nx-2)*(ny-1):(nx-1)*(ny-1)]+var[(nx-4)*(ny-1):(nx-3)*(ny-1)])+coeff[2]*(var[(nx-5)*(ny-1):(nx-4)*(ny-1)])+coeff[3]*(var[(nx-6)*(ny-1):(nx-5)*(ny-1)])+coeff[4]*(var[(nx-7)*(ny-1):(nx-6)*(ny-1)])
            result[(nx-2)*(ny-1):(nx-1)*(ny-1)]=coeff[0]*var[(nx-2)*(ny-1):(nx-1)*(ny-1)]+coeff[1]*(var[(nx-3)*(ny-1):(nx-2)*(ny-1)])+coeff[2]*(var[(nx-4)*(ny-1):(nx-3)*(ny-1)])+coeff[3]*(var[(nx-5)*(ny-1):(nx-4)*(ny-1)])+coeff[4]*(var[(nx-6)*(ny-1):(nx-5)*(ny-1)])
        else:
            result[4:(ny-1)*(nx-1)-4]=coeff[0]*var[4:(ny-1)*(nx-1)-4]+coeff[1]*(var[5:(ny-1)*(nx-1)-3]+var[3:(ny-1)*(nx-1)-5])+coeff[2]*(var[6:(ny-1)*(nx-1)-2]+var[2:(ny-1)*(nx-1)-6])+coeff[3]*(var[7:(ny-1)*(nx-1)-1]+var[1:(ny-1)*(nx-1)-7])+coeff[4]*(var[8:(ny-1)*(nx-1)]+var[:(ny-1)*(nx-1)-8])
            if free_surf==0:
                result[range(0,(nx-1)*(ny-1),ny-1)]=coeff[0]*var[range(0,(nx-1)*(ny-1),ny-1)]+coeff[1]*(var[range(1,(nx-1)*(ny-1),ny-1)])+coeff[2]*(var[range(2,(nx-1)*(ny-1),ny-1)])+coeff[3]*(var[range(3,(nx-1)*(ny-1),ny-1)])+coeff[4]*(var[range(4,(nx-1)*(ny-1),ny-1)])
                result[range(1,(nx-1)*(ny-1),ny-1)]=coeff[0]*var[range(1,(nx-1)*(ny-1),ny-1)]+coeff[1]*(var[range(2,(nx-1)*(ny-1),ny-1)]+var[range(0,(nx-1)*(ny-1),ny-1)])+coeff[2]*(var[range(3,(nx-1)*(ny-1),ny-1)])+coeff[3]*(var[range(4,(nx-1)*(ny-1),ny-1)])+coeff[4]*(var[range(5,(nx-1)*(ny-1),ny-1)])
                result[range(2,(nx-1)*(ny-1),ny-1)]=coeff[0]*var[range(2,(nx-1)*(ny-1),ny-1)]+coeff[1]*(var[range(3,(nx-1)*(ny-1),ny-1)]+var[range(1,(nx-1)*(ny-1),ny-1)])+coeff[2]*(var[range(4,(nx-1)*(ny-1),ny-1)]+var[range(0,(nx-1)*(ny-1),ny-1)])+coeff[3]*(var[range(5,(nx-1)*(ny-1),ny-1)])+coeff[4]*(var[range(6,(nx-1)*(ny-1),ny-1)])
                result[range(3,(nx-1)*(ny-1),ny-1)]=coeff[0]*var[range(3,(nx-1)*(ny-1),ny-1)]+coeff[1]*(var[range(4,(nx-1)*(ny-1),ny-1)]+var[range(2,(nx-1)*(ny-1),ny-1)])+coeff[2]*(var[range(5,(nx-1)*(ny-1),ny-1)]+var[range(1,(nx-1)*(ny-1),ny-1)])+coeff[3]*(var[range(6,(nx-1)*(ny-1),ny-1)]+var[range(0,(nx-1)*(ny-1),ny-1)])+coeff[4]*(var[range(7,(nx-1)*(ny-1),ny-1)])
            else:
                result[range(0,(nx-1)*(ny-1),ny-1)]=(-3144919/352800)*var[range(0,(nx-1)*(ny-1),ny-1)]+16*(var[range(1,(nx-1)*(ny-1),ny-1)])-14*(var[range(2,(nx-1)*(ny-1),ny-1)])+(112/9)*(var[range(3,(nx-1)*(ny-1),ny-1)])-(35/4)*(var[range(4,(nx-1)*(ny-1),ny-1)])+(112/25)*(var[range(5,(nx-1)*(ny-1),ny-1)])-(14/9)*(var[range(6,(nx-1)*(ny-1),ny-1)])+(16/49)*(var[range(7,(nx-1)*(ny-1),ny-1)])-(1/32)*(var[range(8,(nx-1)*(ny-1),ny-1)])
                result[range(1,(nx-1)*(ny-1),ny-1)]=(271343/156800)*var[range(0,(nx-1)*(ny-1),ny-1)]-(1991/630)*(var[range(1,(nx-1)*(ny-1),ny-1)])+(57/40)*(var[range(2,(nx-1)*(ny-1),ny-1)])+(13/60)*(var[range(3,(nx-1)*(ny-1),ny-1)])-(109/288)*(var[range(4,(nx-1)*(ny-1),ny-1)])+(6/25)*(var[range(5,(nx-1)*(ny-1),ny-1)])-(11/120)*(var[range(6,(nx-1)*(ny-1),ny-1)])+(179/8820)*(var[range(7,(nx-1)*(ny-1),ny-1)])-(9/4480)*(var[range(8,(nx-1)*(ny-1),ny-1)])
                result[range(2,(nx-1)*(ny-1),ny-1)]=(-18519/78400)*var[range(0,(nx-1)*(ny-1),ny-1)]+(58/35)*(var[range(1,(nx-1)*(ny-1),ny-1)])-(251/90)*(var[range(2,(nx-1)*(ny-1),ny-1)])+(22/15)*(var[range(3,(nx-1)*(ny-1),ny-1)])-(1/16)*(var[range(4,(nx-1)*(ny-1),ny-1)])-(14/225)*(var[range(5,(nx-1)*(ny-1),ny-1)])+(1/30)*(var[range(6,(nx-1)*(ny-1),ny-1)])-(2/245)*(var[range(7,(nx-1)*(ny-1),ny-1)])+(17/20160)*(var[range(8,(nx-1)*(ny-1),ny-1)])
                result[range(3,(nx-1)*(ny-1),ny-1)]=(74801/1411200)*var[range(0,(nx-1)*(ny-1),ny-1)]-(37/140)*(var[range(1,(nx-1)*(ny-1),ny-1)])+(67/40)*(var[range(2,(nx-1)*(ny-1),ny-1)])-(263/90)*(var[range(3,(nx-1)*(ny-1),ny-1)])+(53/32)*(var[range(4,(nx-1)*(ny-1),ny-1)])-(23/100)*(var[range(5,(nx-1)*(ny-1),ny-1)])+(13/360)*(var[range(6,(nx-1)*(ny-1),ny-1)])-(1/245)*(var[range(7,(nx-1)*(ny-1),ny-1)])+(1/4480)*(var[range(8,(nx-1)*(ny-1),ny-1)])
            result[range(ny-5,(nx-1)*(ny-1),ny-1)]=coeff[0]*var[range(ny-5,(nx-1)*(ny-1),ny-1)]+coeff[1]*(var[range(ny-4,(nx-1)*(ny-1),ny-1)]+var[range(ny-6,(nx-1)*(ny-1),ny-1)])+coeff[2]*(var[range(ny-3,(nx-1)*(ny-1),ny-1)]+var[range(ny-7,(nx-1)*(ny-1),ny-1)])+coeff[3]*(var[range(ny-2,(nx-1)*(ny-1),ny-1)]+var[range(ny-8,(nx-1)*(ny-1),ny-1)])+coeff[4]*(var[range(ny-9,(nx-1)*(ny-1),ny-1)])
            result[range(ny-4,(nx-1)*(ny-1),ny-1)]=coeff[0]*var[range(ny-4,(nx-1)*(ny-1),ny-1)]+coeff[1]*(var[range(ny-3,(nx-1)*(ny-1),ny-1)]+var[range(ny-5,(nx-1)*(ny-1),ny-1)])+coeff[2]*(var[range(ny-2,(nx-1)*(ny-1),ny-1)]+var[range(ny-6,(nx-1)*(ny-1),ny-1)])+coeff[3]*(var[range(ny-7,(nx-1)*(ny-1),ny-1)])+coeff[4]*(var[range(ny-8,(nx-1)*(ny-1),ny-1)])
            result[range(ny-3,(nx-1)*(ny-1),ny-1)]=coeff[0]*var[range(ny-3,(nx-1)*(ny-1),ny-1)]+coeff[1]*(var[range(ny-2,(nx-1)*(ny-1),ny-1)]+var[range(ny-4,(nx-1)*(ny-1),ny-1)])+coeff[2]*(var[range(ny-5,(nx-1)*(ny-1),ny-1)])+coeff[3]*(var[range(ny-6,(nx-1)*(ny-1),ny-1)])+coeff[4]*(var[range(ny-7,(nx-1)*(ny-1),ny-1)])
            result[range(ny-2,(nx-1)*(ny-1),ny-1)]=coeff[0]*var[range(ny-2,(nx-1)*(ny-1),ny-1)]+coeff[1]*(var[range(ny-3,(nx-1)*(ny-1),ny-1)])+coeff[2]*(var[range(ny-4,(nx-1)*(ny-1),ny-1)])+coeff[3]*(var[range(ny-5,(nx-1)*(ny-1),ny-1)])+coeff[4]*(var[range(ny-6,(nx-1)*(ny-1),ny-1)])
    return result/dx**2


def beta_i(dim,dx,nx,ny,delta,beta0,i,j,free_surf):
    n_beta=int(delta/dx)
    aux=0
    if j==1:  # j==1 vx,vy, j!=1 u, wx, wy (for 1SD)
        aux=1/2

    if dim==1:
        result=np.zeros((nx-1,1))
        for k in range(n_beta):
            result[k]=beta0*pow((delta-(k+1-aux)*dx)/delta,2)
            result[-(1+k)]=beta0*pow((delta-(k+1/2+aux)*dx)/delta,2)
    elif dim==2:
        result=np.zeros(((nx-1)*(ny-1),1))
        if i==1:
            for k in range(n_beta):
                result[np.arange((ny-1)*k,(ny-1)*(k+1))]=beta0*pow((delta-(k+1-aux)*dx)/delta,2)
                result[np.arange((ny-1)*(nx-2-k),(ny-1)*(nx-1-k))]=beta0*pow((delta-(k+1/2+aux)*dx)/delta,2)
        else:
            if free_surf==0: # condition of the free surface for the functions Beta for the 'y' direction (Beta_20 and Beta_21)
                for k in range(n_beta):
                    result[np.arange(k,(ny-1)*(nx-1),ny-1)]=beta0*pow((delta-(k+1/2+aux)*dx)/delta,2)
                    result[np.arange(ny-2-k,(ny-1)*(nx-1),ny-1)]=beta0*pow((delta-(k+1-aux)*dx)/delta,2)
            else:
                for k in range(n_beta):
                    result[np.arange(ny-2-k,(ny-1)*(nx-1),ny-1)]=beta0*pow((delta-(k+1-aux)*dx)/delta,2)

    return result


def dpx_1D(vec,dx,pos,ord,nx):
    # Function to calculate the partial derivative of a variable

    # INPUTS:
    # vec: vector of (m-1)x(n-1) elements, i.e., all the interior points of the selected variable
    # dx: spatial step
    # pos: position of the grid point (to calculate the derivative)
    #       1 - derivatives of the variables with the nearest points to the left or the lower boundary of the effective domain (vx)
    #       2 - for the other variables (u, wx)
    # ord: order of the finite difference scheme used in the derivative
    # nx-1: number of points in the spatial grid on the x axis in the effective numerical domain

    result=np.zeros((nx-1,1))
    if ord=='2':
        # using centered differences of second order
        if pos==1:
            result[:(nx-2)]=vec[1:(nx-1)]-vec[:(nx-2)]
            result[nx-2]=-vec[nx-2]
        elif pos==2:
            result[0]=vec[0]
            result[1:(nx-1)]=vec[1:(nx-1)]-vec[:(nx-2)]
        result=result/dx
    elif ord=='4':
        # using centered differences of fourth order
        if pos==1:
            result[0]=27*(-vec[0]+vec[1])-vec[2]
            result[1:(nx-3)]=27*(-vec[1:(nx-3)]+vec[2:(nx-2)])+vec[:(nx-4)]-vec[3:]
            result[nx-3]=vec[nx-4]+27*(-vec[nx-3]+vec[nx-2])
            result[nx-2]=-27*vec[nx-2]+vec[nx-3]
        elif pos==2:
            result[0]=27*vec[0]-vec[1]
            result[1]=27*(-vec[0]+vec[1])-vec[2]
            result[2:(nx-2)]=27*(-vec[1:(nx-3)]+vec[2:(nx-2)])+vec[:(nx-4)]-vec[3:]
            result[nx-2]=27*(-vec[nx-3]+vec[nx-2])+vec[nx-4]
        result=result/(24*dx)
    elif ord[0]=='8':
        # using centered differences of eigth order
        if pos==1:
            result[0]=vec[1]-vec[0]-1/15*vec[2]+1/125*vec[3]-1/1715*vec[4]
            result[1]=vec[2]-vec[1]-1/15*(vec[3]-vec[0])+1/125*vec[4]-1/1715*vec[5]
            result[2]=vec[3]-vec[2]-1/15*(vec[4]-vec[1])+1/125*(vec[5]-vec[0])-1/1715*vec[6]
            result[3:(nx-5)]=vec[4:(nx-4)]-vec[3:(nx-5)]-1/15*(vec[5:(nx-3)]-vec[2:(nx-6)])+1/125*(vec[6:(nx-2)]-vec[1:(nx-7)])-1/1715*(vec[7:]-vec[:(nx-8)])
            result[nx-5]=vec[nx-4]-vec[nx-5]-1/15*(vec[nx-3]-vec[nx-6])+1/125*(vec[nx-2]-vec[nx-7])-1/1715*(-vec[nx-8])
            result[nx-4]=vec[nx-3]-vec[nx-4]-1/15*(vec[nx-2]-vec[nx-5])+1/125*(-vec[nx-6])-1/1715*(-vec[nx-7])
            result[nx-3]=vec[nx-2]-vec[nx-3]-1/15*(-vec[nx-4])+1/125*(-vec[nx-5])-1/1715*(-vec[nx-6])
            result[nx-2]=-vec[nx-2]-1/15*(-vec[nx-3])+1/125*(-vec[nx-4])-1/1715*(-vec[nx-5])
        elif pos==2:
            result[0]=vec[0]-1/15*vec[1]+1/125*vec[2]-1/1715*vec[3]
            result[1]=vec[1]-vec[0]-1/15*vec[2]+1/125*vec[3]-1/1715*vec[4]
            result[2]=vec[2]-vec[1]-1/15*(vec[3]-vec[0])+1/125*vec[4]-1/1715*vec[5]
            result[3]=vec[3]-vec[2]-1/15*(vec[4]-vec[1])+1/125*(vec[5]-vec[0])-1/1715*vec[6]
            result[4:(nx-4)]=vec[4:(nx-4)]-vec[3:(nx-5)]-1/15*(vec[5:(nx-3)]-vec[2:(nx-6)])+1/125*(vec[6:(nx-2)]-vec[1:(nx-7)])-1/1715*(vec[7:]-vec[:(nx-8)])
            result[nx-4]=vec[nx-4]-vec[nx-5]-1/15*(vec[nx-3]-vec[nx-6])+1/125*(vec[nx-2]-vec[nx-7])-1/1715*(-vec[nx-8])
            result[nx-3]=vec[nx-3]-vec[nx-4]-1/15*(vec[nx-2]-vec[nx-5])+1/125*(-vec[nx-6])-1/1715*(-vec[nx-7])
            result[nx-2]=vec[nx-2]-vec[nx-3]-1/15*(-vec[nx-4])+1/125*(-vec[nx-5])-1/1715*(-vec[nx-6])
        result=result*1225/(1024*dx)

    return result


def dpx_2D(vec,dx,dir,pos,ord,nx,ny,free_surf):
    # Function to calculate the partial derivative of a variable

    # INPUTS:
    # vec: vector of (m-1)x(n-1) elements, i.e., all the interior points of the selected variable
    # dx: spatial step
    # dir: x direction of the derivation
    #       1 - x direction
    #       2 - y direction
    # pos: position of the grid point (to calculate the derivative)
    #       1 - derivatives of the variables with the nearest points to the left or the lower boundary of the effective domain (vx, vy, for 1SD)
    #       2 - are the other variables (u, wx, wy, for 1SD)
    # ord: order of the finite difference scheme used in the derivative
    # nx-1: number of points in the spatial grid on the x axis in the effective numerical domain
    # ny-1: number of points in the spatial grid on the y axis in the effective numerical domain

    result=np.zeros(((nx-1)*(ny-1),1))
    if ord=='2':
        # using centered differences of second order
        if dir==1:
            if pos==1:
                result[:((nx-2)*(ny-1))]=vec[(ny-1):((nx-1)*(ny-1))]-vec[0:((nx-2)*(ny-1))]
                result[((nx-2)*(ny-1)):((nx-1)*(ny-1))]=-vec[((nx-2)*(ny-1)):((nx-1)*(ny-1))]
            elif pos==2:
                result[:(ny-1)]=vec[:(ny-1)]
                result[(ny-1):((nx-1)*(ny-1))]=vec[(ny-1):((nx-1)*(ny-1))]-vec[:((nx-2)*(ny-1))]
        else:
            if pos==1:
                result[1:((nx-1)*(ny-1))] = vec[:((nx-1)*(ny-1)-1)] - vec[1:((nx-1)*(ny-1))]
                result[range(0,(nx-1)*(ny-1),ny-1)]=-vec[range(0,(nx-1)*(ny-1),ny-1)]
            elif pos==2:
                result[:((nx-1)*(ny-1)-1)] = vec[:((nx-1)*(ny-1)-1)] - vec[1:((nx-1)*(ny-1))]
                result[range(ny-2,(nx-1)*(ny-1),ny-1)]=vec[range(ny-2,(nx-1)*(ny-1),ny-1)]
        result=result/dx
    elif ord=='4':
        # using centered differences of fourth order
        if dir==1:
            if pos==1:
                result[:(ny-1)]=27*(-vec[:(ny-1)]+vec[(ny-1):2*(ny-1)])-vec[2*(ny-1):3*(ny-1)]
                result[(ny-1):(nx-3)*(ny-1)]=27*(-vec[(ny-1):(nx-3)*(ny-1)]+vec[2*(ny-1):(nx-2)*(ny-1)])+vec[:(ny-1)*(nx-4)]-vec[3*(ny-1):(nx-1)*(ny-1)]
                result[(ny-1)*(nx-3):(ny-1)*(nx-2)]=vec[(ny-1)*(nx-4):(ny-1)*(nx-3)]+27*(-vec[(ny-1)*(nx-3):(ny-1)*(nx-2)]+vec[(ny-1)*(nx-2):(ny-1)*(nx-1)])
                result[(ny-1)*(nx-2):(ny-1)*(nx-1)]=-27*vec[(ny-1)*(nx-2):(ny-1)*(nx-1)]+vec[(ny-1)*(nx-3):(ny-1)*(nx-2)]
            elif pos==2:
                result[:(ny-1)]=27*vec[:(ny-1)]-vec[(ny-1):2*(ny-1)]
                result[(ny-1):2*(ny-1)]=27*(-vec[:(ny-1)]+vec[(ny-1):2*(ny-1)])-vec[2*(ny-1):3*(ny-1)]
                result[2*(ny-1):(ny-1)*(nx-2)]=27*(-vec[(ny-1):(ny-1)*(nx-3)]+vec[2*(ny-1):(ny-1)*(nx-2)])+vec[:(ny-1)*(nx-4)]-vec[3*(ny-1):(ny-1)*(nx-1)]
                result[(ny-1)*(nx-2):(ny-1)*(nx-1)]=27*(-vec[(ny-1)*(nx-3):(ny-1)*(nx-2)]+vec[(ny-1)*(nx-2):(ny-1)*(nx-1)])+vec[(ny-1)*(nx-4):(ny-1)*(nx-3)]
        else:
            if pos==1:
                result[2:(ny-1)*(nx-1)-1]=27*(vec[1:(ny-1)*(nx-1)-2]-vec[2:(ny-1)*(nx-1)-1])-vec[:(ny-1)*(nx-1)-3]+vec[3:(ny-1)*(nx-1)]
                result[range(0,(nx-1)*(ny-1),ny-1)]=-27*vec[range(0,(nx-1)*(ny-1),ny-1)]+vec[range(1,(nx-1)*(ny-1),ny-1)]
                result[range(1,(nx-1)*(ny-1),ny-1)]=27*(vec[range(0,(nx-1)*(ny-1),ny-1)]-vec[range(1,(nx-1)*(ny-1),ny-1)])+vec[range(2,(nx-1)*(ny-1),ny-1)]
                result[range(ny-2,(nx-1)*(ny-1),ny-1)]=27*(vec[range(ny-3,(nx-1)*(ny-1),ny-1)]-vec[range(ny-2,(nx-1)*(ny-1),ny-1)])-vec[range(ny-4,(nx-1)*(ny-1),ny-1)]
            elif pos==2:
                result[1:(ny-1)*(nx-1)-2]=27*(vec[1:(ny-1)*(nx-1)-2]-vec[2:(ny-1)*(nx-1)-1])-vec[0:(ny-1)*(nx-1)-3]+vec[3:(ny-1)*(nx-1)]
                result[range(0,(nx-1)*(ny-1),ny-1)]=27*(vec[range(0,(nx-1)*(ny-1),ny-1)]-vec[range(1,(nx-1)*(ny-1),ny-1)])+vec[range(2,(nx-1)*(ny-1),ny-1)]
                result[range(ny-3,(nx-1)*(ny-1),ny-1)]=27*(vec[range(ny-3,(nx-1)*(ny-1),ny-1)]-vec[range(ny-2,(nx-1)*(ny-1),ny-1)])-vec[range(ny-4,(nx-1)*(ny-1),ny-1)]
                result[range(ny-2,(nx-1)*(ny-1),ny-1)]=27*vec[range(ny-2,(nx-1)*(ny-1),ny-1)]-vec[range(ny-3,(nx-1)*(ny-1),ny-1)]
        result=result/(24*dx)
    elif ord=='8':
        # using centered differences of fourth order
        if dir==1:
            if pos==1:
                result[:(ny-1)]=vec[(ny-1):2*(ny-1)]-vec[:(ny-1)]-1/15*vec[2*(ny-1):3*(ny-1)]+1/125*vec[3*(ny-1):4*(ny-1)]-1/1715*vec[4*(ny-1):5*(ny-1)]
                result[(ny-1):2*(ny-1)]=vec[2*(ny-1):3*(ny-1)]-vec[(ny-1):2*(ny-1)]-1/15*(vec[3*(ny-1):4*(ny-1)]-vec[:(ny-1)])+1/125*vec[4*(ny-1):5*(ny-1)]-1/1715*vec[5*(ny-1):6*(ny-1)]
                result[2*(ny-1):3*(ny-1)]=vec[3*(ny-1):4*(ny-1)]-vec[2*(ny-1):3*(ny-1)]-1/15*(vec[4*(ny-1):5*(ny-1)]-vec[(ny-1):2*(ny-1)])+1/125*(vec[5*(ny-1):6*(ny-1)]-vec[:(ny-1)])-1/1715*vec[6*(ny-1):7*(ny-1)]
                result[3*(ny-1):(nx-5)*(ny-1)]=vec[4*(ny-1):(nx-4)*(ny-1)]-vec[3*(ny-1):(nx-5)*(ny-1)]-1/15*(vec[5*(ny-1):(nx-3)*(ny-1)]-vec[2*(ny-1):(nx-6)*(ny-1)])+1/125*(vec[6*(ny-1):(nx-2)*(ny-1)]-vec[(ny-1):(nx-7)*(ny-1)])-1/1715*(vec[7*(ny-1):]-vec[:(nx-8)*(ny-1)])
                result[(nx-5)*(ny-1):(nx-4)*(ny-1)]=vec[(nx-4)*(ny-1):(nx-3)*(ny-1)]-vec[(nx-5)*(ny-1):(nx-4)*(ny-1)]-1/15*(vec[(nx-3)*(ny-1):(nx-2)*(ny-1)]-vec[(nx-6)*(ny-1):(nx-5)*(ny-1)])+1/125*(vec[(nx-2)*(ny-1):]-vec[(nx-7)*(ny-1):(nx-6)*(ny-1)])-1/1715*(-vec[(nx-8)*(ny-1):(nx-7)*(ny-1)])
                result[(nx-4)*(ny-1):(nx-3)*(ny-1)]=vec[(nx-3)*(ny-1):(nx-2)*(ny-1)]-vec[(nx-4)*(ny-1):(nx-3)*(ny-1)]-1/15*(vec[(nx-2)*(ny-1):]-vec[(nx-5)*(ny-1):(nx-4)*(ny-1)])+1/125*(-vec[(nx-6)*(ny-1):(nx-5)*(ny-1)])-1/1715*(-vec[(nx-7)*(ny-1):(nx-6)*(ny-1)])
                result[(nx-3)*(ny-1):(nx-2)*(ny-1)]=vec[(nx-2)*(ny-1):]-vec[(nx-3)*(ny-1):(nx-2)*(ny-1)]-1/15*(-vec[(nx-4)*(ny-1):(nx-3)*(ny-1)])+1/125*(-vec[(nx-5)*(ny-1):(nx-4)*(ny-1)])-1/1715*(-vec[(nx-6)*(ny-1):(nx-5)*(ny-1)])
                result[(nx-2)*(ny-1):]=-vec[(nx-2)*(ny-1):]-1/15*(-vec[(nx-3)*(ny-1):(nx-2)*(ny-1)])+1/125*(-vec[(nx-4)*(ny-1):(nx-3)*(ny-1)])-1/1715*(-vec[(nx-5)*(ny-1):(nx-4)*(ny-1)])
            elif pos==2:
                result[:(ny-1)]=vec[:(ny-1)]-1/15*vec[(ny-1):2*(ny-1)]+1/125*vec[2*(ny-1):3*(ny-1)]-1/1715*vec[3*(ny-1):4*(ny-1)]
                result[(ny-1):2*(ny-1)]=vec[(ny-1):2*(ny-1)]-vec[:(ny-1)]-1/15*vec[2*(ny-1):3*(ny-1)]+1/125*vec[3*(ny-1):4*(ny-1)]-1/1715*vec[4*(ny-1):5*(ny-1)]
                result[2*(ny-1):3*(ny-1)]=vec[2*(ny-1):3*(ny-1)]-vec[(ny-1):2*(ny-1)]-1/15*(vec[3*(ny-1):4*(ny-1)]-vec[:(ny-1)])+1/125*vec[4*(ny-1):5*(ny-1)]-1/1715*vec[5*(ny-1):6*(ny-1)]
                result[3*(ny-1):4*(ny-1)]=vec[3*(ny-1):4*(ny-1)]-vec[2*(ny-1):3*(ny-1)]-1/15*(vec[4*(ny-1):5*(ny-1)]-vec[(ny-1):2*(ny-1)])+1/125*(vec[5*(ny-1):6*(ny-1)]-vec[:(ny-1)])-1/1715*vec[6*(ny-1):7*(ny-1)]
                result[4*(ny-1):(nx-4)*(ny-1)]=vec[4*(ny-1):(nx-4)*(ny-1)]-vec[3*(ny-1):(nx-5)*(ny-1)]-1/15*(vec[5*(ny-1):(nx-3)*(ny-1)]-vec[2*(ny-1):(nx-6)*(ny-1)])+1/125*(vec[6*(ny-1):(nx-2)*(ny-1)]-vec[(ny-1):(nx-7)*(ny-1)])-1/1715*(vec[7*(ny-1):]-vec[:(nx-8)*(ny-1)])
                result[(nx-4)*(ny-1):(nx-3)*(ny-1)]=vec[(nx-4)*(ny-1):(nx-3)*(ny-1)]-vec[(nx-5)*(ny-1):(nx-4)*(ny-1)]-1/15*(vec[(nx-3)*(ny-1):(nx-2)*(ny-1)]-vec[(nx-6)*(ny-1):(nx-5)*(ny-1)])+1/125*(vec[(nx-2)*(ny-1):]-vec[(nx-7)*(ny-1):(nx-6)*(ny-1)])-1/1715*(-vec[(nx-8)*(ny-1):(nx-7)*(ny-1)])
                result[(nx-3)*(ny-1):(nx-2)*(ny-1)]=vec[(nx-3)*(ny-1):(nx-2)*(ny-1)]-vec[(nx-4)*(ny-1):(nx-3)*(ny-1)]-1/15*(vec[(nx-2)*(ny-1):]-vec[(nx-5)*(ny-1):(nx-4)*(ny-1)])+1/125*(-vec[(nx-6)*(ny-1):(nx-5)*(ny-1)])-1/1715*(-vec[(nx-7)*(ny-1):(nx-6)*(ny-1)])
                result[(nx-2)*(ny-1):]=vec[(nx-2)*(ny-1):]-vec[(nx-3)*(ny-1):(nx-2)*(ny-1)]-1/15*(-vec[(nx-4)*(ny-1):(nx-3)*(ny-1)])+1/125*(-vec[(nx-5)*(ny-1):(nx-4)*(ny-1)])-1/1715*(-vec[(nx-6)*(ny-1):(nx-5)*(ny-1)])
        else:
            if pos==1:
                result[4:(ny-1)*(nx-1)-3]=vec[3:(ny-1)*(nx-1)-4]-vec[4:(ny-1)*(nx-1)-3]-1/15*(vec[2:(ny-1)*(nx-1)-5]-vec[5:(ny-1)*(nx-1)-2])+1/125*(vec[1:(ny-1)*(nx-1)-6]-vec[6:(ny-1)*(nx-1)-1])-1/1715*(vec[:(ny-1)*(nx-1)-7]-vec[7:(ny-1)*(nx-1)])
                result[range(ny-4,(nx-1)*(ny-1),ny-1)]=vec[range(ny-5,(nx-1)*(ny-1),ny-1)]-vec[range(ny-4,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(ny-6,(nx-1)*(ny-1),ny-1)]-vec[range(ny-3,(nx-1)*(ny-1),ny-1)])+1/125*(vec[range(ny-7,(nx-1)*(ny-1),ny-1)]-vec[range(ny-2,(nx-1)*(ny-1),ny-1)])-1/1715*(vec[range(ny-8,(nx-1)*(ny-1),ny-1)])
                result[range(ny-3,(nx-1)*(ny-1),ny-1)]=vec[range(ny-4,(nx-1)*(ny-1),ny-1)]-vec[range(ny-3,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(ny-5,(nx-1)*(ny-1),ny-1)]-vec[range(ny-2,(nx-1)*(ny-1),ny-1)])+1/125*(vec[range(ny-6,(nx-1)*(ny-1),ny-1)])-1/1715*(vec[range(ny-7,(nx-1)*(ny-1),ny-1)])
                result[range(ny-2,(nx-1)*(ny-1),ny-1)]=vec[range(ny-3,(nx-1)*(ny-1),ny-1)]-vec[range(ny-2,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(ny-4,(nx-1)*(ny-1),ny-1)])+1/125*(vec[range(ny-5,(nx-1)*(ny-1),ny-1)])-1/1715*(vec[range(ny-6,(nx-1)*(ny-1),ny-1)])
                if free_surf==0:
                    result[range(0,(nx-1)*(ny-1),ny-1)]=-vec[range(0,(nx-1)*(ny-1),ny-1)]-1/15*(-vec[range(1,(nx-1)*(ny-1),ny-1)])+1/125*(-vec[range(2,(nx-1)*(ny-1),ny-1)])-1/1715*(-vec[range(3,(nx-1)*(ny-1),ny-1)])
                    result[range(1,(nx-1)*(ny-1),ny-1)]=vec[range(0,(nx-1)*(ny-1),ny-1)]-vec[range(1,(nx-1)*(ny-1),ny-1)]-1/15*(-vec[range(2,(nx-1)*(ny-1),ny-1)])+1/125*(-vec[range(3,(nx-1)*(ny-1),ny-1)])-1/1715*(-vec[range(4,(nx-1)*(ny-1),ny-1)])
                    result[range(2,(nx-1)*(ny-1),ny-1)]=vec[range(1,(nx-1)*(ny-1),ny-1)]-vec[range(2,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(0,(nx-1)*(ny-1),ny-1)]-vec[range(3,(nx-1)*(ny-1),ny-1)])+1/125*(-vec[range(4,(nx-1)*(ny-1),ny-1)])-1/1715*(-vec[range(5,(nx-1)*(ny-1),ny-1)])
                    result[range(3,(nx-1)*(ny-1),ny-1)]=vec[range(2,(nx-1)*(ny-1),ny-1)]-vec[range(3,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(1,(nx-1)*(ny-1),ny-1)]-vec[range(4,(nx-1)*(ny-1),ny-1)])+1/125*(vec[range(0,(nx-1)*(ny-1),ny-1)]-vec[range(5,(nx-1)*(ny-1),ny-1)])-1/1715*(-vec[range(6,(nx-1)*(ny-1),ny-1)])
                    result=result*1225/(1024*dx)
                else:
                    result=result*1225/(1024*dx)
                    result[range(0,(nx-1)*(ny-1),ny-1)]=(-6435/1024)*vec[range(1,(nx-1)*(ny-1),ny-1)]+(5005/1024)*vec[range(2,(nx-1)*(ny-1),ny-1)]-(27027/5120)*vec[range(3,(nx-1)*(ny-1),ny-1)]+(32175/7168)*vec[range(4,(nx-1)*(ny-1),ny-1)]-(25025/9216)*vec[range(5,(nx-1)*(ny-1),ny-1)]+(12285/11264)*vec[range(6,(nx-1)*(ny-1),ny-1)]-(3465/13312)*vec[range(7,(nx-1)*(ny-1),ny-1)]+(143/5120)*vec[range(8,(nx-1)*(ny-1),ny-1)]
                    result[range(1,(nx-1)*(ny-1),ny-1)]=(131093/107520)*vec[range(1,(nx-1)*(ny-1),ny-1)]-(49087/46080)*vec[range(2,(nx-1)*(ny-1),ny-1)]-(10973/76800)*vec[range(3,(nx-1)*(ny-1),ny-1)]+(4597/21504)*vec[range(4,(nx-1)*(ny-1),ny-1)]-(4019/27648)*vec[range(5,(nx-1)*(ny-1),ny-1)]+(10331/168960)*vec[range(6,(nx-1)*(ny-1),ny-1)]-(2983/199680)*vec[range(7,(nx-1)*(ny-1),ny-1)]+(2621/1612800)*vec[range(8,(nx-1)*(ny-1),ny-1)]
                    result[range(2,(nx-1)*(ny-1),ny-1)]=(-8707/107520)*vec[range(1,(nx-1)*(ny-1),ny-1)]+(17947/15360)*vec[range(2,(nx-1)*(ny-1),ny-1)]-(29319/25600)*vec[range(3,(nx-1)*(ny-1),ny-1)]+(533/21504)*vec[range(4,(nx-1)*(ny-1),ny-1)]+(263/9216)*vec[range(5,(nx-1)*(ny-1),ny-1)]-(903/56320)*vec[range(6,(nx-1)*(ny-1),ny-1)]+(283/66560)*vec[range(7,(nx-1)*(ny-1),ny-1)]-(257/537600)*vec[range(8,(nx-1)*(ny-1),ny-1)]
                    result[range(3,(nx-1)*(ny-1),ny-1)]=(543/35840)*vec[range(1,(nx-1)*(ny-1),ny-1)]-(265/3072)*vec[range(2,(nx-1)*(ny-1),ny-1)]+(1233/1024)*vec[range(3,(nx-1)*(ny-1),ny-1)]-(8625/7168)*vec[range(4,(nx-1)*(ny-1),ny-1)]+(775/9216)*vec[range(5,(nx-1)*(ny-1),ny-1)]-(639/56320)*vec[range(6,(nx-1)*(ny-1),ny-1)]+(15/13312)*vec[range(7,(nx-1)*(ny-1),ny-1)]-(1/21504)*vec[range(8,(nx-1)*(ny-1),ny-1)]
            elif pos==2:
                result[3:(ny-1)*(nx-1)-4]=vec[3:(ny-1)*(nx-1)-4]-vec[4:(ny-1)*(nx-1)-3]-1/15*(vec[2:(ny-1)*(nx-1)-5]-vec[5:(ny-1)*(nx-1)-2])+1/125*(vec[1:(ny-1)*(nx-1)-6]-vec[6:(ny-1)*(nx-1)-1])-1/1715*(vec[:(ny-1)*(nx-1)-7]-vec[7:(ny-1)*(nx-1)])
                result[range(ny-5,(nx-1)*(ny-1),ny-1)]=vec[range(ny-5,(nx-1)*(ny-1),ny-1)]-vec[range(ny-4,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(ny-6,(nx-1)*(ny-1),ny-1)]-vec[range(ny-3,(nx-1)*(ny-1),ny-1)])+1/125*(vec[range(ny-7,(nx-1)*(ny-1),ny-1)]-vec[range(ny-2,(nx-1)*(ny-1),ny-1)])-1/1715*(vec[range(ny-8,(nx-1)*(ny-1),ny-1)])
                result[range(ny-4,(nx-1)*(ny-1),ny-1)]=vec[range(ny-4,(nx-1)*(ny-1),ny-1)]-vec[range(ny-3,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(ny-5,(nx-1)*(ny-1),ny-1)]-vec[range(ny-2,(nx-1)*(ny-1),ny-1)])+1/125*(vec[range(ny-6,(nx-1)*(ny-1),ny-1)])-1/1715*(vec[range(ny-7,(nx-1)*(ny-1),ny-1)])
                result[range(ny-3,(nx-1)*(ny-1),ny-1)]=vec[range(ny-3,(nx-1)*(ny-1),ny-1)]-vec[range(ny-2,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(ny-4,(nx-1)*(ny-1),ny-1)])+1/125*(vec[range(ny-5,(nx-1)*(ny-1),ny-1)])-1/1715*(vec[range(ny-6,(nx-1)*(ny-1),ny-1)])
                result[range(ny-2,(nx-1)*(ny-1),ny-1)]=vec[range(ny-2,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(ny-3,(nx-1)*(ny-1),ny-1)])+1/125*(vec[range(ny-4,(nx-1)*(ny-1),ny-1)])-1/1715*(vec[range(ny-5,(nx-1)*(ny-1),ny-1)])
                if free_surf==0:
                    result[range(0,(nx-1)*(ny-1),ny-1)]=vec[range(0,(nx-1)*(ny-1),ny-1)]-vec[range(1,(nx-1)*(ny-1),ny-1)]-1/15*(-vec[range(2,(nx-1)*(ny-1),ny-1)])+1/125*(-vec[range(3,(nx-1)*(ny-1),ny-1)])-1/1715*(-vec[range(4,(nx-1)*(ny-1),ny-1)])
                    result[range(1,(nx-1)*(ny-1),ny-1)]=vec[range(1,(nx-1)*(ny-1),ny-1)]-vec[range(2,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(0,(nx-1)*(ny-1),ny-1)]-vec[range(3,(nx-1)*(ny-1),ny-1)])+1/125*(-vec[range(4,(nx-1)*(ny-1),ny-1)])-1/1715*(-vec[range(5,(nx-1)*(ny-1),ny-1)])
                    result[range(2,(nx-1)*(ny-1),ny-1)]=vec[range(2,(nx-1)*(ny-1),ny-1)]-vec[range(3,(nx-1)*(ny-1),ny-1)]-1/15*(vec[range(1,(nx-1)*(ny-1),ny-1)]-vec[range(4,(nx-1)*(ny-1),ny-1)])+1/125*(vec[range(0,(nx-1)*(ny-1),ny-1)]-vec[range(5,(nx-1)*(ny-1),ny-1)])-1/1715*(-vec[range(6,(nx-1)*(ny-1),ny-1)])
                    result=result*1225/(1024*dx)
                else:
                    result=result*1225/(1024*dx)
                    result[range(0,(nx-1)*(ny-1),ny-1)]=(5034629/3763200)*vec[range(0,(nx-1)*(ny-1),ny-1)]-(23533/15360)*vec[range(1,(nx-1)*(ny-1),ny-1)]+(4259/15360)*vec[range(2,(nx-1)*(ny-1),ny-1)]-(1103/9216)*vec[range(3,(nx-1)*(ny-1),ny-1)]+(151/3072)*vec[range(4,(nx-1)*(ny-1),ny-1)]-(1171/76800)*vec[range(5,(nx-1)*(ny-1),ny-1)]+(139/46080)*vec[range(6,(nx-1)*(ny-1),ny-1)]-(211/752640)*vec[range(7,(nx-1)*(ny-1),ny-1)]
                    result[range(1,(nx-1)*(ny-1),ny-1)]=(-363509/3763200)*vec[range(0,(nx-1)*(ny-1),ny-1)]+(6297/5120)*vec[range(1,(nx-1)*(ny-1),ny-1)]-(6147/5120)*vec[range(2,(nx-1)*(ny-1),ny-1)]+(211/3072)*vec[range(3,(nx-1)*(ny-1),ny-1)]+(3/1024)*vec[range(4,(nx-1)*(ny-1),ny-1)]-(153/25600)*vec[range(5,(nx-1)*(ny-1),ny-1)]+(29/15360)*vec[range(6,(nx-1)*(ny-1),ny-1)]-(57/250880)*vec[range(7,(nx-1)*(ny-1),ny-1)]
                    result[range(2,(nx-1)*(ny-1),ny-1)]=(4631/250880)*vec[range(0,(nx-1)*(ny-1),ny-1)]-(305/3072)*vec[range(1,(nx-1)*(ny-1),ny-1)]+(1245/1024)*vec[range(2,(nx-1)*(ny-1),ny-1)]-(3725/3072)*vec[range(3,(nx-1)*(ny-1),ny-1)]+(275/3072)*vec[range(4,(nx-1)*(ny-1),ny-1)]-(69/5120)*vec[range(5,(nx-1)*(ny-1),ny-1)]+(5/3072)*vec[range(6,(nx-1)*(ny-1),ny-1)]-(5/50176)*vec[range(7,(nx-1)*(ny-1),ny-1)]

    return result
