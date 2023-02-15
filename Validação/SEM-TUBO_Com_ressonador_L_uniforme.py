import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la



### SEM DUTO C/ RESSONADORES DIVIDIDO EM VÁRIOS ELEMENTOS UNIFORMES

#PARAMETROS DO PROBLEMA
eta = 0.001 # Amortecimento estrutural
E = 210*10**9*(1 + 1J*eta) # Modulo de elasticidade
rhot = 7800 # Massa específica do aço
De = 0.2 # Diâmetro externo da tubulação
t = 0.005 # Espessura da parede
Di = De-2*t # Diâmetro interno da tubulação

I = (np.pi/64) * (De**4 - Di**4) # Momento de incercia de massa

At = (np.pi/4) * (De**2 - Di**2)
Ai = (np.pi/4) * (Di**2)

rhol = 1000 # Massa específica do líquido
rhog = 1.18 # Massa específica do gás
alpham = 0 # Fração de vazio
rhom = rhol - alpham*(rhol-rhog) # Massa específica da mistura

rhoA = (rhot*At + rhom*Ai) #Massa da estrutura/ unidade de comprimento
EI = E*I #Rigidez flexural da estrutura

LT = 3 #Tamanho da Viga

n_ele = 3 # Número de elementos de viga
L = LT/n_ele # Tamanho de cada elemento

#### PARÂMETROS DO RESSONADOR ######

m = 0.000000002
w1 = 1*2*np.pi
w2 = 1*2*np.pi
deltaf = (w2-w1)/(w1*(2-1))
k1 = m*w1**2

qsi = 0.005

Nr = 3
L = LT/Nr


def ressonador(N,Omegaj):
    global m,w1,deltaf,k1,qsi
    
    
    Lambda5j = alpha*((1 + (N-1)*deltaf)**2 + 2J*Omegaj*((1 + (N-1)*deltaf)**2)*qsi )
    
    Lambda4j = alpha*((1 + (N-1)*deltaf)**2 - Omegaj**2 + 2J*Omegaj*((1 + (N-1)*deltaf)**2)*qsi)
    
    return [Lambda4j,Lambda5j]
    


######## ADMENSIONALIZAÇÃO ########

mu = m/(rhoA*L)
alpha = k1/(EI/L**3)


wf = np.linspace(100*2*np.pi,600*2*np.pi,2000,endpoint=True)

Omega = wf/w1

beta = np.zeros((len(Omega),1),dtype='cdouble')


beta = np.sqrt(Omega)*(alpha/mu)**(0.25)



U = np.zeros( ( len(beta) , int(5 + 3*(Nr-1) -2) ),dtype='cdouble' )

for ii in range(len(beta)):
    
    F = np.zeros( ( int(5 + 3*(Nr-1) ) ),dtype='cdouble' )
    F[-3] = 1/(EI/L**2)
    
    
    SMB = np.zeros( ( (int(5 + 3*(Nr-1) )) , int(5 + 3*(Nr-1) ) ),dtype='cdouble' )
    
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
    
    
    for jj in range(Nr):
        Lambda4,Lambda5 = ressonador(jj+1,Omega[ii])
        if jj==0:
            SMB[int(0):int(5),int(0):int(5)] = [ [Lambda1, Lambda3_,-Lambda1_,Lambda3,0],
                                          [Lambda3_,Lambda2,-Lambda3,Lambda2_,0],
                                          [-Lambda1_,-Lambda3,Lambda1+Lambda5,-Lambda3_,-Lambda5],
                                          [Lambda3,Lambda2_,-Lambda3_,Lambda2,0],
                                          [0,0,-Lambda5,0,Lambda4] ]
        else:
            SMB[int(jj*(2) + jj - 1):int(jj*(2) + jj - 1+2),int(jj*(2) + jj - 1):int(jj*(2) + jj - 1 + 2)] +=  [ [Lambda1, Lambda3_],
                                                                                                                [Lambda3_,Lambda2]]
            
            SMB[int(jj*(2) + jj - 1):int(jj*(2) + jj - 1+2),int(jj*(2) + jj - 1 + 3):int(jj*(2) + jj - 1 + 5)] +=  [ [-Lambda1_,Lambda3],
                                                                                                                    [-Lambda3,Lambda2_]]
            

            SMB[int(2 + jj*(3)):int(5 + jj*(3)),int(2 + jj*3 -3 ):int(2 + jj*3 -1)] +=  [[-Lambda1_,-Lambda3],
                                                      [Lambda3,Lambda2_],
                                                      [0,0] ]
            
            SMB[int(2 + jj*(3)):int(5 + jj*(3)),int(2 + jj*3 ):int(2 + jj*3 + 3)] +=  [[Lambda1+Lambda5,-Lambda3_,-Lambda5],
                                                      [-Lambda3_,Lambda2,0],
                                                      [-Lambda5,0,Lambda4] ]
            
    CC = np.array([0,1])

    for kk in [0,1]:
        SMB = np.delete(SMB,CC,axis=kk)
        if kk==0:
            F = np.delete(F,CC,axis=kk)
    U[ii,:] = la.solve(SMB,F).T

#PLot

plt.figure(1,figsize=(12,8))
  
plt.plot(wf/(2*np.pi),20*np.log10(np.abs(U[:,-3]*L)),'--',color='red',label='SEM-Validação 3')

plt.xlabel("Frequência (hz)",fontsize=20)
plt.ylabel("Receptância (dB re 1m)",fontsize=20)

plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 20)
plt.grid('on')
plt.show()

U_sem3 = U[:,-3]*L  
