# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 12:14:07 2020

ESN worm controller


@author: jlh75
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

#np.random.seed(0)

""" domain """

n1=100
n2=10
X=np.zeros((n1*n2,2))
for i in range(n1):
    for j in range(n2):
        X[j+n2*i]=[i,j]

""" food distribution """

lF=5
SIG_F=np.exp(-euclidean_distances(X,X)**2/(2*lF**2))
F=np.random.multivariate_normal(np.zeros(n1*n2),SIG_F)

plt.figure(figsize=(20,2))
plt.imshow(F.reshape((n1,n2)).T)
plt.show()

""" transitions """

T=np.zeros((n1*n2,3),int)
for i in range(n1-1):
    j=0
    T[j+n2*i]=[(j+1)+n2*(i+1),(j)+n2*(i+1),(j)+n2*(i+1)]
    for j in range(1,n2-1):
        T[j+n2*i]=[(j+1)+n2*(i+1),(j)+n2*(i+1),(j-1)+n2*(i+1)]
    j=n2-1
    T[j+n2*i]=[(j)+n2*(i+1),(j)+n2*(i+1),(j-1)+n2*(i+1)]    
i=n1-1
j=0
T[j+n2*i]=[(j+1)+n2*(0),(j)+n2*(0),(j)+n2*(0)]
for j in range(1,n2-1):
    T[j+n2*i]=[(j+1)+n2*(0),(j)+n2*(0),(j-1)+n2*(0)]
j=n2-1
T[j+n2*i]=[(j)+n2*(0),(j)+n2*(0),(j-1)+n2*(0)]

""" simulate some random behaviour """

x=0
for t in range(10):
    plt.figure(figsize=(20,2))
    plt.imshow(F.reshape((n1,n2)).T)
    plt.plot(X[x][0],X[x][1],'d',color='red')
    plt.show()
    x=np.random.choice(T[x])
    input('press a key')
    
""" record longer random orbit """

N=10000
H=np.zeros(N,int)
A=np.random.randint(3,size=N,dtype=int)
for t in range(1,N):
    H[t]=T[H[t-1],A[t-1]]
    
plt.plot([F[h] for h in H[:200]])
plt.show()

""" future discuonted rewards """

gamma=0.6
G=np.zeros(N)
for i in range(1,N):
    G[N-1-i]=F[H[N-i]]+gamma*G[N-i]

plt.plot(G[:200])
plt.show()

""" generate ESN """

nr=100
pr=0.05
alpha=0.9
gain=10
M=np.zeros((nr,nr))
Win=np.zeros(nr)
for i in range(nr):
    if np.random.rand()<pr:
        Win[i]=gain*np.random.randn()
    for j in range(nr):
        if np.random.rand()<pr:
            M[i,j]=np.random.randn()
M=alpha*M/np.linalg.norm(M,2)


""" drive ESN """

R=np.zeros((N,nr))
for t in range(1,N):
    R[t]=np.tanh(np.matmul(M,R[t-1])+Win*F[H[t]])
    
""" learn linear Q function (same as Wout) """
       
Qout=np.zeros((nr,3))
for a in range(3):
    IA=[i for i in range(N) if A[i]==a]
    Qout[:,a]=np.linalg.lstsq(R[IA,:],G[IA])[0]

plt.plot(G[2000:2200])
for a in range(3):
    IA=[i for i in range(2000,2200) if A[i]==a]
    plt.plot(np.array(IA)-2000,np.matmul(R[IA],Qout[:,a]),'.',label='action '+str(a))
plt.legend()
plt.show()


""" simulate greedy epsilon behaviour from end of data """


eps=0.25
alpha = 1e-3

x=H[-1]
r=R[-1]
for t in range(100000):
    """ pick action a """
    if np.random.rand()<eps:
        a=np.random.choice(3)
    else:
        a=np.argmax(np.matmul(r,Qout))
    """ find new r and new x """
    xnew=T[x,a]
    rnew=np.tanh(np.matmul(M,r)+Win*F[xnew])
    """ update Qa """
    Qout[:,a] = Qout[:,a] + alpha * r * (F[xnew] + gamma *  np.max(np.matmul(rnew,Qout)) - np.matmul(r,Qout[:,a]))
    """ update """
    x=xnew
    r=rnew

eps=0.05
for t in range(200):
    plt.figure(figsize=(20,2))
    plt.imshow(F.reshape((n1,n2)).T)
    plt.plot(X[x][0],X[x][1],'d',color='red')
    plt.show()
#    if np.random.rand()<eps:
#        x=np.random.choice(T[x,:])
#    else:
#        x=T[x,np.argmax(np.matmul(r,Qout))]
    #r=np.tanh(np.matmul(M,r)+Win*F[x])
    """ pick action a """
    if np.random.rand()<eps:
        a=np.random.choice(3)
    else:
        a=np.argmax(np.matmul(r,Qout))
    """ find new r and new x """
    xnew=T[x,a]
    rnew=np.tanh(np.matmul(M,r)+Win*F[xnew])
    """ update Qa """
    #Qout[:,a] = Qout[:,a] + alpha * r * (F[xnew] + gamma *  np.max(np.matmul(rnew,Qout)) - np.matmul(r,Qout[:,a]))
    """ update """
    x=xnew
    r=rnew
    input('press a key')




