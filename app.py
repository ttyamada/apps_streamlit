#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##################

# This script plot two 3fold vectors which on the 3-fold section 
# in 3D parallel and perpendicular space

##################
#import sys
#import math
#import string
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PI=np.pi
TAU=(1.0+np.sqrt(5.0))/2.0

def _hvector(n1,n2,n3,n4,n5,n6):
    """Direct lattice vector
    """
    n=np.array([n1,n2,n3,n4,n5,n6])
    n.shape=(1,6)
    c1=apar/np.sqrt(2.0+TAU)
    c2=aperp/np.sqrt(2.0+TAU)
    
    # lattice spaceing with a_{par}
    m1=c1*np.array([1.0,  TAU,  0.0,\
                    TAU,  0.0,  1.0,\
                    TAU,  0.0, -1.0,\
                    0.0,  1.0, -TAU,\
                   -1.0,  TAU,  0.0,\
                    0.0,  1.0,  TAU])
    m1.shape=(6,3)        
    
    # lattice spaceing with a_{perp}      
    m2=c2*np.array([ TAU, -1.0,  0.0,\
                    -1.0,  0.0,  TAU,\
                    -1.0,  0.0, -TAU,\
                     0.0,  TAU,  1.0,\
                    -TAU, -1.0,  0.0,\
                     0.0,  TAU, -1.0])
    m2.shape=(6,3)
    
    # Join a sequence of arrays         
    m=np.concatenate((m1,m2), axis=1)

    # similarity transformation matrix; S
    #s=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,\
    #            0.5, 0.5, 0.5,-0.5,-0.5, 0.5,\
    #            0.5, 0.5, 0.5, 0.5,-0.5,-0.5,\
    #            0.5,-0.5, 0.5, 0.5, 0.5,-0.5,\
    #            0.5,-0.5,-0.5, 0.5, 0.5, 0.5,\
    #            0.5, 0.5,-0.5,-0.5, 0.5, 0.5])
    #s.shape=(6,6)
    #s=np.linalg.inv(s)
    #s=_matrixpow(s.T,ost)
    #m1=np.dot(s,m)
    m1=m
    
    m2=m1[:,[0,1,2]]
    val=np.dot(n,m2) # projected n vector onto Epar
    val1=val[0,0] # x in Epar
    val2=val[0,1] # y in Epar
    val3=val[0,2] # z in Epar
    val7=np.sqrt(val1**2+val2**2+val3**2)

    m3=m1[:,[3,4,5]]
    val=np.dot(n,m3) # projected n vector onto Eperp
    val4=val[0,0] # x in Eperp
    val5=val[0,1] # y in Eperp
    val6=val[0,2] # z in Eperp
    val8=np.sqrt(val4**2+val5**2+val6**2)

    return val1,val2,val3,val4,val5,val6,val7,val8


def hvector1(n1,n2,n3,n4,n5,n6,stms):
    """Direct lattice vector
    """
    n=np.array([n1,n2,n3,n4,n5,n6])
    n.shape=(1,6)
    c1=apar/np.sqrt(2.0+TAU)
    c2=aperp/np.sqrt(2.0+TAU)
    
    # lattice spaceing with a_{par}
    m1=c1*np.array([1.0,  TAU,  0.0,\
                    TAU,  0.0,  1.0,\
                    TAU,  0.0, -1.0,\
                    0.0,  1.0, -TAU,\
                   -1.0,  TAU,  0.0,\
                    0.0,  1.0,  TAU])
    m1.shape=(6,3)        
    
    # lattice spaceing with a_{perp}      
    m2=c2*np.array([ TAU, -1.0,  0.0,\
                    -1.0,  0.0,  TAU,\
                    -1.0,  0.0, -TAU,\
                     0.0,  TAU,  1.0,\
                    -TAU, -1.0,  0.0,\
                     0.0,  TAU, -1.0])
    m2.shape=(6,3)
    
    # Join a sequence of arrays         
    m=np.concatenate((m1,m2), axis=1)

    # similarity transformation matrix; S
    s=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,\
                0.5, 0.5, 0.5,-0.5,-0.5, 0.5,\
                0.5, 0.5, 0.5, 0.5,-0.5,-0.5,\
                0.5,-0.5, 0.5, 0.5, 0.5,-0.5,\
                0.5,-0.5,-0.5, 0.5, 0.5, 0.5,\
                0.5, 0.5,-0.5,-0.5, 0.5, 0.5])
    s.shape=(6,6)
    #s=np.linalg.inv(s) # comment-out when calculate direct space component
    s=s.T
    s=_matrixpow(s,stms)
    m1=np.dot(s,m)
    #m1=m
    
    m2=m1[:,[0,1,2]]
    val=np.dot(n,m2) # projected n vector onto Epar
    val1=val[0,0] # x in Epar
    val2=val[0,1] # y in Epar
    val3=val[0,2] # z in Epar
    val7=np.sqrt(val1**2+val2**2+val3**2)

    m3=m1[:,[3,4,5]]
    val=np.dot(n,m3) # projected n vector onto Eperp
    val4=val[0,0] # x in Eperp
    val5=val[0,1] # y in Eperp
    val6=val[0,2] # z in Eperp
    val8=np.sqrt(val4**2+val5**2+val6**2)

    return val1,val2,val3,val4,val5,val6,val7,val8


def _matrixpow(ma,n):
    ma=np.array(ma)
    (mx,my)=ma.shape
    if mx==my:
        if n==0:
            return np.identity(mx)
        elif n<0:
            return np.zeros((mx,mx))
        else:
            tmp=np.identity(mx)
            for i in range(n):
                tmp=np.dot(tmp,ma)
            return tmp
    else:
        print('matrix has not regular shape')
        return  
        
    
if __name__ == '__main__':
    #
    ##############################################
    # Lattice constant in Epar
    ##############################################
    #apar=3
    apar=1.0

    ##############################################
    # Lattice constant in Epar
    ##############################################
    #aperp=1.0
    aperp=apar
    
    ##############################################
    # Order of similality transformation
    # order of similarlity transformation (ost) 
    # for reciprocal lattice vectpr
    ##############################################
    ost=0
    
    
    ##############################################
    # 5 fold 6D vectors for f-fold section
    ##############################################
    #
    (h1,h2,h3,h4,h5,h6)=(0,1,1,1,1,1)   # 1st 5-fold
    (k1,k2,k3,k4,k5,k6)=(1,0,0,0,0,0)  # 2nd 5-fold 
    

    # projection vector 2fold, lz
    (lz1,lz2,lz3,lz4,lz5,lz6)=(0,1,0,0,0,-1)
    # two orthogonal 2fold vector perpendicular to lz
    (lx1,lx2,lx3,lx4,lx5,lx6)=(0,1,0,0,0,1)
    (ly1,ly2,ly3,ly4,ly5,ly6)=(1,0,0,1,0,0)
    
    #
    h1=_hvector(h1,h2,h3,h4,h5,h6)
    h2=_hvector(k1,k2,k3,k4,k5,k6)
    h3=_hvector(lz1,lz2,lz3,lz4,lz5,lz6)
    h4=_hvector(lx1,lx2,lx3,lx4,lx5,lx6)
    h5=_hvector(ly1,ly2,ly3,ly4,ly5,ly6)
    
    
    #####   3D PLOT   ####
    
    ################################
    # basis vectors for par space (perp space)
    ################################
    vectors_basis=np.array([[0,0,0,1,0,0], \
                            [0,0,0,0,1,0], \
                            [0,0,0,0,0,1]]) 
    vectors_3f=np.array([0,0,0,1,1,1])
    
    ################################
    # 6D d vectors
    ################################
    d1=hvector1(1,0,0,0,0,0,ost)
    d2=hvector1(0,1,0,0,0,0,ost)
    d3=hvector1(0,0,1,0,0,0,ost)
    d4=hvector1(0,0,0,1,0,0,ost)
    d5=hvector1(0,0,0,0,1,0,ost)
    d6=hvector1(0,0,0,0,0,1,ost)
    #
    d7=hvector1(-1,0,0,0,0,0,ost)
    d8=hvector1(0,-1,0,0,0,0,ost)
    d9=hvector1(0,0,-1,0,0,0,ost)
    d10=hvector1(0,0,0,-1,0,0,ost)
    d11=hvector1(0,0,0,0,-1,0,ost)
    d12=hvector1(0,0,0,0,0,-1,ost)
    
    ################################
    # projection onto par space
    ################################
    vectors_par=np.array([[0,0,0,d1[0],d1[1],d1[2]], \
                           [0,0,0,d2[0],d2[1],d2[2]], \
                           [0,0,0,d3[0],d3[1],d3[2]], \
                           [0,0,0,d4[0],d4[1],d4[2]], \
                           [0,0,0,d5[0],d5[1],d5[2]], \
                           [0,0,0,d6[0],d6[1],d6[2]]]) 
    h1_par=np.array([0,0,0,h1[0],h1[1],h1[2]])
    h2_par=np.array([0,0,0,h2[0],h2[1],h2[2]])
    h3_par=np.array([0,0,0,h3[0],h3[1],h3[2]])
    h4_par=np.array([0,0,0,h4[0],h4[1],h4[2]])
    h5_par=np.array([0,0,0,h5[0],h5[1],h5[2]])  
    
    edges_ico_par=np.array([[d5[0],d5[1],d5[2],d1[0],d1[1],d1[2]], \
                            [d11[0],d11[1],d11[2],d1[0],d1[1],d1[2]], \
                            [d5[0],d5[1],d5[2],d7[0],d7[1],d7[2]], \
                            [d11[0],d11[1],d11[2],d7[0],d7[1],d7[2]], \
                            [d3[0],d3[1],d3[2],d2[0],d2[1],d2[2]], \
                            [d9[0],d9[1],d9[2],d2[0],d2[1],d2[2]], \
                            [d3[0],d3[1],d3[2],d8[0],d8[1],d8[2]], \
                            [d9[0],d9[1],d9[2],d8[0],d8[1],d8[2]], \
                            [d4[0],d4[1],d4[2],d6[0],d6[1],d6[2]], \
                            [d5[0],d5[1],d5[2],d6[0],d6[1],d6[2]],\
                            [d4[0],d4[1],d4[2],d12[0],d12[1],d12[2]], \
                            [d5[0],d5[1],d5[2],d12[0],d12[1],d12[2]]])
    
    """
    m1=c1*np.array([1.0,  TAU,  0.0,\
                    TAU,  0.0,  1.0,\
                    TAU,  0.0, -1.0,\
                    0.0,  1.0, -TAU,\
                   -1.0,  TAU,  0.0,\
                    0.0,  1.0,  TAU])
    """
                            
    ################################
    # projection onto perp space                    
    ################################
    vectors_perp=np.array([[0,0,0,d1[3],d1[4],d1[5]], \
                            [0,0,0,d2[3],d2[4],d2[5]], \
                            [0,0,0,d3[3],d3[4],d3[5]], \
                            [0,0,0,d4[3],d4[4],d4[5]], \
                            [0,0,0,d5[3],d5[4],d5[5]], \
                            [0,0,0,d6[3],d6[4],d6[5]]])
        
    edges_ico_perp=np.array([[d1[3],d1[4],d1[5],d5[3],d5[4],d5[5]], \
                           [d7[3],d7[4],d7[5],d5[3],d5[4],d5[5]], \
                           [d1[3],d1[4],d1[5],d9[3],d9[4],d9[5]], \
                           [d7[3],d7[4],d7[5],d9[3],d9[4],d9[5]], \
                           [d2[3],d2[4],d2[5],d3[3],d3[4],d3[5]], \
                           [d8[3],d8[4],d8[5],d3[3],d3[4],d3[5]], \
                           [d2[3],d2[4],d2[5],d9[3],d9[4],d9[5]], \
                           [d8[3],d8[4],d8[5],d9[3],d9[4],d9[5]], \
                           [d6[3],d6[4],d6[5],d4[3],d4[4],d4[5]], \
                           [d12[3],d12[4],d12[5],d4[3],d4[4],d4[5]], \
                           [d6[3],d6[4],d6[5],d10[3],d10[4],d10[5]], \
                           [d12[3],d12[4],d12[5],d10[3],d10[4],d10[5]]])
        
    h1_perp=np.array([0,0,0,h1[3],h1[4],h1[5]])
    h2_perp=np.array([0,0,0,h2[3],h2[4],h2[5]])
    h3_perp=np.array([0,0,0,h3[3],h3[4],h3[5]])
    h4_perp=np.array([0,0,0,h4[3],h4[4],h4[5]])
    h5_perp=np.array([0,0,0,h5[3],h5[4],h5[5]])
        
    """
    m2=c2*np.array([ TAU, -1.0,  0.0,\
                    -1.0,  0.0,  TAU,\
                    -1.0,  0.0, -TAU,\
                     0.0,  TAU,  1.0,\
                    -TAU, -1.0,  0.0,\
                     0.0,  TAU, -1.0])
    """
                     
                     
    ################################
    # icosahedron in par space
    ################################
    vertec_par_ico=np.array([d1[0],d1[1],d1[2], \
                    d2[0],d2[1],d2[2], \
                    d3[0],d3[1],d3[2], \
                    d4[0],d4[1],d4[2], \
                    d5[0],d5[1],d5[2], \
                    d6[0],d6[1],d6[2], \
                    d7[0],d7[1],d7[2], \
                    d8[0],d8[1],d8[2], \
                    d9[0],d9[1],d9[2], \
                    d10[0],d10[1],d10[2], \
                    d11[0],d11[1],d11[2], \
                    d12[0],d12[1],d12[2]])
    vertec_par_ico.shape=(12,3)

    
    ################################
    # icosahedron in perp space
    ################################
    vertec_perp_ico=np.array([d1[3],d1[4],d1[5], \
                    d2[3],d2[4],d2[5], \
                    d3[3],d3[4],d3[5], \
                    d4[3],d4[4],d4[5], \
                    d5[3],d5[4],d5[5], \
                    d6[3],d6[4],d6[5], \
                    d7[3],d7[4],d7[5], \
                    d8[3],d8[4],d8[5], \
                    d9[3],d9[4],d9[5], \
                    d10[3],d10[4],d10[5], \
                    d11[3],d11[4],d11[5], \
                    d12[3],d12[4],d12[5]])
    vertec_perp_ico.shape=(12,3)    
    
    ########
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax2 = fig2.add_subplot(111, projection='3d')
    

    ################################
    # basis vectors for par space (perp space)
    ################################
    """
    for vector in vectors_basis:
        v = np.array([vector[3],vector[4],vector[5]])
        vlength=np.linalg.norm(v)
        ax1.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5],
                pivot='tail',arrow_length_ratio=0.2/vlength, color="m")
        ax2.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5],
                pivot='tail',arrow_length_ratio=0.2/vlength, color="m")
    """
    
    tmpval=1.5
    for vector in vectors_basis:
        v = np.array([vector[3],vector[4],vector[5]])
        vlength=np.linalg.norm(v)
        ax1.quiver(vector[0]*tmpval,vector[1]*tmpval,vector[2]*tmpval,vector[3]*tmpval,vector[4]*tmpval,vector[5]*tmpval,
                pivot='tail',linewidths=1.,arrow_length_ratio=0.1/vlength, color = "gray")
        ax2.quiver(vector[0]*tmpval,vector[1]*tmpval,vector[2]*tmpval,vector[3]*tmpval,vector[4]*tmpval,vector[5]*tmpval,
                pivot='tail',linewidths=1.,arrow_length_ratio=0.1/vlength, color = "gray")
    tmpval=0.4
    for vector in vectors_basis:
        v = np.array([vector[3],vector[4],vector[5]])
        vlength=np.linalg.norm(v)
        ax1.quiver(vector[0]*tmpval,vector[1]*tmpval,vector[2]*tmpval,vector[3]*tmpval,vector[4]*tmpval,vector[5]*tmpval,
                pivot='tail',linewidths=2.,arrow_length_ratio=0.1/vlength, color="r")
        ax2.quiver(vector[0]*tmpval,vector[1]*tmpval,vector[2]*tmpval,vector[3]*tmpval,vector[4]*tmpval,vector[5]*tmpval,
                pivot='tail',linewidths=2.,arrow_length_ratio=0.1/vlength, color="r")
    """
    # 3fold axis
    ax1.quiver(vectors_3f[0],vectors_3f[1],vectors_3f[2],vectors_3f[3],vectors_3f[4],vectors_3f[5],
            pivot='tail',arrow_length_ratio=0.2/vlength,colors='g')  
    ax2.quiver(vectors_3f[0],vectors_3f[1],vectors_3f[2],vectors_3f[3],vectors_3f[4],vectors_3f[5],
            pivot='tail',arrow_length_ratio=0.2/vlength,colors='g')          
    
    """
    
    
    
    
    
    
    ################################
    # projection onto par space     
    ################################
    # d vectors
    for vector in vectors_par:
        v = np.array([vector[3],vector[4],vector[5]])
        vlength=np.linalg.norm(v)
        ax1.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5],
                pivot='tail',linewidths=1.5,arrow_length_ratio=0.1/vlength, color="c")
    """
    for vector in edges_ico_par:
        v = np.array([vector[3],vector[4],vector[5]])
        #vlength=np.linalg.norm(v)
        ax1.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5],
                pivot='tail',arrow_length_ratio=0.)
    """
    """
    # 
    # h1 and h2 vectors
    v = np.array([h1_par[3],h1_par[4],h1_par[5]])
    vlength=np.linalg.norm(v)
    ax1.quiver(h1_par[0],h1_par[1],h1_par[2],h1_par[3],h1_par[4],h1_par[5],
                pivot='tail',arrow_length_ratio=0.2/vlength,colors='g')
    v = np.array([h2_par[3],h2_par[4],h2_par[5]])
    vlength=np.linalg.norm(v)
    ax1.quiver(h2_par[0],h2_par[1],h2_par[2],h2_par[3],h2_par[4],h2_par[5],
                pivot='tail',arrow_length_ratio=0.2/vlength,colors='m') 
    # orthogonal 2-fold vectors 
    v = np.array([h3_par[3],h3_par[4],h3_par[5]])
    vlength=np.linalg.norm(v)
    ax1.quiver(h3_par[0],h3_par[1],h3_par[2],h3_par[3],h3_par[4],h3_par[5],
                pivot='tail',arrow_length_ratio=0.2/vlength,colors='r') 

    v = np.array([h4_par[3],h4_par[4],h4_par[5]])
    vlength=np.linalg.norm(v)
    ax1.quiver(h4_par[0],h4_par[1],h4_par[2],h4_par[3],h4_par[4],h4_par[5],
                pivot='tail',arrow_length_ratio=0.2/vlength,colors='b')
    
    v = np.array([h5_par[3],h5_par[4],h5_par[5]])
    vlength=np.linalg.norm(v)
    ax1.quiver(h5_par[0],h5_par[1],h5_par[2],h5_par[3],h5_par[4],h5_par[5],
                pivot='tail',arrow_length_ratio=0.2/vlength,colors='y') 
    """
    ################################
    # projection onto perp space    
    ################################
    for vector in vectors_perp:
        v = np.array([vector[3],vector[4],vector[5]])
        vlength=np.linalg.norm(v)
        ax2.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5],
                pivot='tail',linewidths=1.5,arrow_length_ratio=0.1/vlength, color="c")
    """
    for vector in edges_ico_perp:
        v = np.array([vector[3],vector[4],vector[5]])
        #vlength=np.linalg.norm(v)
        ax2.quiver(vector[0],vector[1],vector[2],vector[3],vector[4],vector[5],
                pivot='tail',arrow_length_ratio=0.1)
    """
    """
    #
    v = np.array([h1_perp[3],h1_perp[4],h1_perp[5]])
    vlength=np.linalg.norm(v)
    ax2.quiver(h1_perp[0],h1_perp[1],h1_perp[2],h1_perp[3],h1_perp[4],h1_perp[5],
                pivot='tail',arrow_length_ratio=0.2/vlength,colors='g')
    v = np.array([h2_perp[3],h2_perp[4],h2_perp[5]])
    vlength=np.linalg.norm(v)
    ax2.quiver(h2_perp[0],h2_perp[1],h2_perp[2],h2_perp[3],h2_perp[4],h2_perp[5],
                pivot='tail',arrow_length_ratio=0.2/vlength,colors='m')

    # orthogonal 2-fold vectors     
    v = np.array([h3_perp[3],h3_perp[4],h3_perp[5]])
    vlength=np.linalg.norm(v)
    ax2.quiver(h3_perp[0],h3_perp[1],h3_perp[2],h3_perp[3],h3_perp[4],h3_perp[5],
                pivot='tail',arrow_length_ratio=0.2/vlength,colors='r')
    v = np.array([h4_perp[3],h4_perp[4],h4_perp[5]])
    vlength=np.linalg.norm(v)
    ax2.quiver(h4_perp[0],h4_perp[1],h4_perp[2],h4_perp[3],h4_perp[4],h4_perp[5],
                pivot='tail',arrow_length_ratio=0.2/vlength,colors='b')
    
    v = np.array([h5_perp[3],h5_perp[4],h5_perp[5]])
    vlength=np.linalg.norm(v)
    ax2.quiver(h5_perp[0],h5_perp[1],h5_perp[2],h5_perp[3],h5_perp[4],h5_perp[5],
                pivot='tail',arrow_length_ratio=0.2/vlength,colors='y')
    """            
    ################################
    # icosahedron in par space
    ################################
    ax1.scatter(vertec_par_ico[:,0], vertec_par_ico[:,1], vertec_par_ico[:,2], s=5.0, color="b")
    
    ################################
    # icosahedron in perp space
    ################################
    ax2.scatter(vertec_perp_ico[:,0], vertec_perp_ico[:,1], vertec_perp_ico[:,2], s=5.0, color="b")

    rangelim=4
    ax1.set_xlim([-rangelim,rangelim])
    ax1.set_ylim([-rangelim,rangelim])
    ax1.set_zlim([-rangelim,rangelim])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax2.set_xlim([-rangelim,rangelim])
    ax2.set_ylim([-rangelim,rangelim])
    ax2.set_zlim([-rangelim,rangelim])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    
    #ax1.set_aspect('equal')
    #ax2.set_aspect('equal')
    
    ax1.view_init(elev=15, azim=15)
    ax2.view_init(elev=15, azim=15)
    
    #ax1.w_xaxis.set_pane_color((0., 0., 0., 0.))
    #ax1.w_yaxis.set_pane_color((0., 0., 0., 0.))
    #ax1.w_zaxis.set_pane_color((0., 0., 0., 0.))
    #ax2.w_xaxis.set_pane_color((0., 0., 0., 0.))
    #ax2.w_yaxis.set_pane_color((0., 0., 0., 0.))
    #ax2.w_zaxis.set_pane_color((0., 0., 0., 0.))
    
    ax1.grid(False)
    ax2.grid(False)
    
    ax1.text2D(0.05, 0.95, "Paralell space", transform=ax1.transAxes)
    ax2.text2D(0.05, 0.95, "Perpendicular space", transform=ax2.transAxes)
    
    plt.show()
