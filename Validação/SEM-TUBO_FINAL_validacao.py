import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.optimize import fsolve


#PARAMETROS DO PROBLEMA
eta = 0.001
E = 210*10**9*(1 + 1J*eta)
rhot = 7800
De = 0.2
t = 0.005
Di = De-2*t

I = (np.pi/64) * (De**4 - Di**4) 

At = (np.pi/4) * (De**2 - Di**2)
Ai = (np.pi/4) * (Di**2)

rhol = 1000
rhog = 1.18
alpham = 0
rhom = rhol - alpham*(rhol-rhog)

rhoA = (rhot*At + rhom*Ai)
EI = I*E

L_extremidade_e = 1.1
L_extremidade_d = 1.3
L_ressonadores = 0.2

L = [L_extremidade_e,L_ressonadores,L_ressonadores,L_ressonadores,L_extremidade_d] #Tamanhos da Viga


#### PARÂMETROS DO RESSONADOR ######

m = 0.000000002
w1 = 1*2*np.pi
w2 = 1*2*np.pi
deltaf = (w2-w1)/(w1*(2-1))
k1 = m*w1**2

qsi = 0.005

Nr = 3



def ressonador(N,Omegaj, m,w1,deltaf,k1,qsi):
    
    
    Lambda5j = k1*((1 + (N-1)*deltaf)**2 + 2J*Omegaj*((1 + (N-1)*deltaf)**2)*qsi )
    
    Lambda4j = k1*((1 + (N-1)*deltaf)**2 - Omegaj**2 + 2J*Omegaj*((1 + (N-1)*deltaf)**2)*qsi)
    
    return [Lambda4j,Lambda5j]
    

def gammas(Omegaj,Lj, m,rhoA,EI,k1):
    
    muj = m/(rhoA*Lj)
    alphaj = k1/(EI/Lj**3)
    
    betaj = np.sqrt(Omegaj) * (alphaj/muj)**(0.25)
    
    Lambda1j = (EI/Lj**3)*((np.cos(betaj)*np.sinh(betaj) 
                + np.sin(betaj)*np.cosh(betaj))*betaj**3/
    (1-np.cos(betaj)*np.cosh(betaj)))
    
    Lambda2j = (EI/Lj**3)*((-np.cos(betaj)*np.sinh(betaj) 
                + np.sin(betaj)*np.cosh(betaj))*betaj*Lj**2/
    (1-np.cos(betaj)*np.cosh(betaj)))
    
    Lambda3j = (EI/Lj**3)*((-np.cos(betaj) 
                + np.cosh(betaj))*betaj**2*Lj/
    (1-np.cos(betaj)*np.cosh(betaj)))
    
    Lambda1_j = (EI/Lj**3)*((np.sin(betaj) 
                + np.sinh(betaj))*betaj**3/
    (1-np.cos(betaj)*np.cosh(betaj)))
    
    Lambda2_j = (EI/Lj**3)*((-np.sin(betaj) 
                + np.sinh(betaj))*betaj*Lj**2/
    (1-np.cos(betaj)*np.cosh(betaj)))
    
    Lambda3_j = (EI/Lj**3)*((np.sin(betaj)*np.sinh(betaj))*betaj**2*Lj/
    (1-np.cos(betaj)*np.cosh(betaj)))
    
    return [Lambda1j,Lambda2j,Lambda3j,Lambda1_j,Lambda2_j,Lambda3_j]
    


######## ADMENSIONALIZAÇÃO ########



wf = np.linspace(100*2*np.pi,600*2*np.pi,2000,endpoint=True)

Omega = wf/w1


U = np.zeros( ( len(wf) , int(4+2 + 3*(Nr) -2) ),dtype='cdouble' )

for ii in range(len(wf)):
    
    
    F = np.zeros( ( 4+2 + 3*(Nr) ),dtype='cdouble' )
    
    F[-2] = 1 #massa total x aceleração da base
    
    SMB = np.zeros( ( 4+2 + 3*(Nr) , 4+2 + 3*(Nr) ),dtype='cdouble' )
    
    
    
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[0], m,rhoA,EI,k1)
    
    SMB[0:4,0:4] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
                                  [Lambda3_,Lambda2,-Lambda3,Lambda2_],
                                  [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
                                  [Lambda3,Lambda2_,-Lambda3_,Lambda2] ], dtype = 'cdouble')
    
    for jj in range(Nr):
        
        Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[jj+1], m,rhoA,EI,k1)
        
        Lambda4,Lambda5 = ressonador(jj+1,Omega[ii], m,w1,deltaf,k1,qsi)
        
        if jj==0:
            SMB[2:7,2:7] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3,0],
                                          [Lambda3_,Lambda2,-Lambda3,Lambda2_,0],
                                          [-Lambda1_,-Lambda3,Lambda1+Lambda5,-Lambda3_,-Lambda5],
                                          [Lambda3,Lambda2_,-Lambda3_,Lambda2,0],
                                          [0,0,-Lambda5,0,Lambda4]])
        else:
            SMB[int(2+jj*(2) + jj - 1):int(2+jj*(2) + jj - 1+2),int(2+jj*(2) + jj - 1):int(2+jj*(2) + jj - 1 + 2)] +=  np.array([ [Lambda1, Lambda3_],
                                                                                                                [Lambda3_,Lambda2]])
            
            SMB[int(2+jj*(2) + jj - 1):int(2+jj*(2) + jj - 1+2),int(2+jj*(2) + jj - 1 + 3):int(2+jj*(2) + jj - 1 + 5)] +=  np.array([ [-Lambda1_,Lambda3],
                                                                                                                    [-Lambda3,Lambda2_]])
            

            SMB[int(2+2 + jj*(3)):int(2+5 + jj*(3)),int(2+2 + jj*3 -3 ):int(2+2 + jj*3 -1)] +=  np.array([[-Lambda1_,-Lambda3],
                                                                                         [Lambda3,Lambda2_],
                                                                                         [0,0] ])
            
            SMB[int(2+2 + jj*(3)):int(2+5 + jj*(3)),int(2+2 + jj*3 ):int(2+2 + jj*3 + 3)] +=  np.array([[Lambda1+Lambda5,-Lambda3_,-Lambda5],
                                                                                       [-Lambda3_,Lambda2,0],
                                                                                       [-Lambda5,0,Lambda4] ])
        
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[-1], m,rhoA,EI,k1)
    
    SMB[-5:-3,-5:-3] +=  [ [Lambda1, Lambda3_],
                          [Lambda3_,Lambda2]]
    
    SMB[-5:-3,-2:] +=  [ [-Lambda1_,Lambda3],
                        [-Lambda3,Lambda2_]]
    

    SMB[-2:,-5:-3] +=  [[-Lambda1_,-Lambda3],
                        [Lambda3,Lambda2_]]
    
    SMB[-2:,-2:] +=  [[Lambda1,-Lambda3_],
                      [-Lambda3_,Lambda2] ]
    
    CC = np.array([0,1])
    

    for kk in [0,1]:
        SMB = np.delete(SMB,CC,axis=kk)
        if kk==0:
            F = np.delete(F,CC,axis=kk)
    U[ii,:] = la.solve(SMB,F).T


###PLOT FRF

plt.figure(1,figsize=(12,8))
  
plt.plot(wf/(2*np.pi),20*np.log10(np.abs(U[:,-2])),'--',color='black',label='SEM-Validação 4')

plt.xlabel("Frequência (hz)",fontsize=20)
plt.ylabel("Receptância (dB re 1m)",fontsize=20)

plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 20)
plt.grid('on')
plt.show()

U_sem4 = U[:,-2]  


    


