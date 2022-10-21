from graph_q_ellip import *
from min_ellip import *
import numpy as np
from random import randrange


def wezlz(Q,R,e,max_it):
    # print('Q',Q)
    # print('R',R)
    if len(Q)==0 or len(R)==5:
        # print('min_el',min_el(R,e,max_it))
        return min_el(R,e,max_it)
    else:
        i=randrange(len(Q))
        # i=0
        q=Q[i,:]
        R1,a12,a22,b1,b2,c=wezlz(np.delete(Q,(i),axis=0),R,e,max_it)
        # print('isnan: ',np.isnan(a12)+np.isnan(a22)+np.isnan(b1)+np.isnan(b2)+np.isnan(c))
        # if (np.isnan(a12)+np.isnan(a22)+np.isnan(b1)+np.isnan(b2)+np.isnan(c))==True:
            # print(a12,a22,b1,b2,c)
            # print('R1',R1)
            # print('R',R)
            # print('Q',np.delete(Q,(i),axis=0))
            # asfdadsfas
        # print('in_el: ',in_el(q,R1,a12,a22,b1,b2,c))
        if in_el(q,R1,a12,a22,b1,b2,c)==1:
            return R1,a12,a22,b1,b2,c
        else:
            return wezlz(np.delete(Q,(i),axis=0),np.vstack((R,q)),e,max_it)

# Q=np.array([[-2.19735341e-12,  4.51847341e+02],
#  [-2.19735341e-12, -4.51847341e+02]])
# Q=np.array([[-2.98801200e+01 , 0.00000000e+00],
#  [-2.98801200e+01, -0.00000000e+00],
#  [ 0.00000000e+00 , 0.00000000e+00],
#  [ 0.00000000e+00 , 0.00000000e+00],
#  [-1.49400600e+01 , 4.51847341e+02],
#  [-1.49400600e+01 ,-4.51847341e+02],
#  [ 0.00000000e+00 , 0.00000000e+00],
#  [ 0.00000000e+00 , 0.00000000e+00],
#  [-2.19735341e-12 , 4.51847341e+02],
#  [-2.19735341e-12 ,-4.51847341e+02]])
# R=np.array([[ -14.94006 ,   -451.84734123]])
# R=np.zeros((0,2))
# e=pow(10,-12)
# max_it=20
# R,a12,a22,b1,b2,c=wezlz(Q,R,e,max_it)
# print(a12,a22,b1,b2,c)



# min_el(array([[-1.49400600e+01, -4.51847341e+02],
#        [-2.19735341e-12,  4.51847341e+02],
#        [-2.19735341e-12, -4.51847341e+02]]), array([nan]), array([nan]), array([nan]), array([nan]), array([nan]))
# e 1e-12
# max_it 20