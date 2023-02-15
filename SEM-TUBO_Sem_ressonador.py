import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

### SEM DUTO S/ RESSONADORES ÚNICO ELEMENTO

#PARAMETROS DO PROBLEMA
eta = 0.001
E = 210*10**9*(1+ 1J*eta)
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
EI = E*I

L = 3 #Tamanho da Viga


#### PARÂMETROS DO RESSONADOR ######


######## ADMENSIONALIZAÇÃO ########

mu = 1/(rhoA*L)
alpha = 1/(EI/L**3)
qsi = 0

wf = np.linspace(100*2*np.pi,600*2*np.pi,2000,endpoint=True)

Omega = wf


beta = np.sqrt(Omega)*(alpha/mu)**(0.25)



U = np.zeros( ( len(beta) , int(4) -2) ,dtype = 'cdouble' )

for ii in range(len(beta)):
    
    F = np.zeros( (4) ,dtype = 'cdouble')
    F[-2] = 1/(EI/L**2)
    
    
    SMB = np.zeros(  (4 , 4 ) ,dtype = 'cdouble')
    
    Lambda1 = ((np.cos(beta[ii])*np.sinh(beta[ii]) 
                + np.sin(beta[ii])*np.cosh(beta[ii]))*beta[ii]**3/
    (1-np.cos(beta[ii])*np.cosh(beta[ii])))
    
    Lambda2 = ((-np.cos(beta[ii])*np.sinh(beta[ii]) 
                + np.sin(beta[ii])*np.cosh(beta[ii]))*beta[ii]/
    (1-np.cos(beta[ii])*np.cosh(beta[ii])))
    
    Lambda3 = ((-np.cos(beta[ii]) 
                + np.cosh(beta[ii]))*beta[ii]**2/
    (1-np.cos(beta[ii])*np.cosh(beta[ii])))
    
    Lambda1_ = ((np.sin(beta[ii]) 
                + np.sinh(beta[ii]))*beta[ii]**3/
    (1-np.cos(beta[ii])*np.cosh(beta[ii])))
    
    Lambda2_ = ((-np.sin(beta[ii]) 
                + np.sinh(beta[ii]))*beta[ii]/
    (1-np.cos(beta[ii])*np.cosh(beta[ii])))
    
    Lambda3_ = ((np.sin(beta[ii])*np.sinh(beta[ii]))*beta[ii]**2/
    (1-np.cos(beta[ii])*np.cosh(beta[ii])))

    SMB = [ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
           [Lambda3_,Lambda2,-Lambda3,Lambda2_],
           [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
           [Lambda3,Lambda2_,-Lambda3_,Lambda2]]

    CC = np.array([0,1])

    for kk in [0,1]:
        SMB = np.delete(SMB,CC,axis=kk)
        if kk==0:
            F = np.delete(F,CC,axis=kk)
    U[ii,:] = la.solve(SMB,F).T

# plt.figure(2)
# plt.spy(SMB, marker=None, markersize=4)  


plt.figure(1,figsize=(12,8))

  
plt.plot(wf/(2*np.pi),20*np.log10(np.abs(U[:,-2]*L)),'--',color='orange',label='SEM-Validação 1')

plt.xlabel("Frequência (hz)",fontsize=20)
plt.ylabel("Receptância (dB re 1m)",fontsize=20)

plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 20)
plt.grid('on')
plt.show()


U_sem1 = U[:,-2]*L   


    


