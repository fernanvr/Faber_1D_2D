import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin


def graph_q_ellip(Q,a12,a22,b1,b2,c,n,type,scale):
    plt.plot(Q[:,0],Q[:,1],'ob',alpha=0.8)

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
    print('th: ',th)

    x0=b1/(2*a11)
    y0=b2/(2*a22)
    a=np.sqrt((pow(x0,2)*a11+pow(y0,2)*a22-c)/a11)*scale
    b=np.sqrt((pow(x0,2)*a11+pow(y0,2)*a22-c)/a22)*scale


    print('a:',a)
    print('b:', b)
    print('(a+b)^2: ',pow(a+b,2))

    theta=np.linspace(0,2*np.pi,n)
    x=a*np.cos(theta)-x0
    y=b*np.sin(theta)-y0
    plt.plot(x*np.cos(th)-y*np.sin(th),x*np.sin(th)+y*np.cos(th), linewidth=2,dashes=type)
    plt.axhline(0,color='k')
    plt.axvline(0,color='k')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.ylim([-5,5])
    # plt.xlim(([-21,6]))
    return (a+b)/2,np.sqrt(np.abs(a**2-b**2)),-np.array([x0*np.cos(th)-y0*np.sin(th),x0*np.sin(th)+y0*np.cos(th)]),a


def graph_q_circles1(Q,a12,a22,b1,b2,c,n,type,scale):
    plt.plot(Q[:,0],Q[:,1],'ob',alpha=0.8)

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
    print('th: ',th)

    x0=b1/(2*a11)
    y0=b2/(2*a22)
    a=np.sqrt((pow(x0,2)*a11+pow(y0,2)*a22-c)/a11)
    b=np.sqrt((pow(x0,2)*a11+pow(y0,2)*a22-c)/a22)


    if a>b:
        b=b+scale*(a-b)
    else:
        a=a+scale*(b-a)

    c=np.sqrt(np.abs(a**2-b**2))

    if scale!=0:
        def f2(x):
            return np.abs(np.max(np.sqrt(pow(Q[:,0]-(-x[0]-c*x[1]),2)+pow(Q[:,1],2))+np.sqrt(pow(Q[:,0]-(-x[0]+c*x[1]),2)+pow(Q[:,1],2)))-2*a*x[1])+a*x[1]/2
        scale1=fmin(f2,[x0,1])
        x0=scale1[0]
        scale1=scale1[1]
        print('a b c',a,b,c,np.max(np.sqrt(pow(Q[:,0]-(-x0-c*scale1),2)+pow(Q[:,1],2))+np.sqrt(pow(Q[:,0]-(-x0+c*scale1),2)+pow(Q[:,1],2))),x0,y0)
        b=b*scale1
        a=a*scale1

    print('a:',a)
    print('b:', b)
    print('(a+b)^2: ',pow(a+b,2))

    theta=np.linspace(0,2*np.pi,n)
    x=a*np.cos(theta)-x0
    y=b*np.sin(theta)-y0
    plt.plot(x*np.cos(th)-y*np.sin(th),x*np.sin(th)+y*np.cos(th), linewidth=2,dashes=type)
    plt.axhline(0,color='k')
    plt.axvline(0,color='k')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.ylim([-5,5])
    # plt.xlim(([-21,6]))
    return (a+b)/2,np.sqrt(np.abs(a**2-b**2)),-np.array([x0*np.cos(th)-y0*np.sin(th),x0*np.sin(th)+y0*np.cos(th)]),a


def graph_q_circles2(Q,a12,a22,b1,b2,c,n,type,scale):
    plt.plot(Q[:,0],Q[:,1],'ob',alpha=0.8)

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
    print('th: ',th)

    x0=b1/(2*a11)
    y0=b2/(2*a22)
    a=np.sqrt((pow(x0,2)*a11+pow(y0,2)*a22-c)/a11)
    b=np.sqrt((pow(x0,2)*a11+pow(y0,2)*a22-c)/a22)


    a=np.maximum(a,b)*scale
    b=a+0

    print('a:',a)
    print('b:', b)
    print('(a+b)^2: ',pow(a+b,2))

    theta=np.linspace(0,2*np.pi,n)
    x=a*np.cos(theta)-x0
    y=b*np.sin(theta)-y0
    plt.plot(x*np.cos(th)-y*np.sin(th),x*np.sin(th)+y*np.cos(th), linewidth=2,dashes=type)
    plt.axhline(0,color='k')
    plt.axvline(0,color='k')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    # plt.ylim([-5,5])
    # plt.xlim(([-21,6]))
    return (a+b)/2,np.sqrt(np.abs(a**2-b**2)),-np.array([x0*np.cos(th)-y0*np.sin(th),x0*np.sin(th)+y0*np.cos(th)]),a
