import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

### SEM DUTO S/ RESSONADORES DIVIDIDO EM VÁRIOS ELEMENTOS

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


######## ADMENSIONALIZAÇÃO ########

mu = 1/(rhoA*L) #Massa do elemento
alpha = 1/(EI/L**3) # Rigidez do elemento


wf = np.linspace(100*2*np.pi,600*2*np.pi,2000,endpoint=True, dtype='cdouble') # Intervalo de análise da frequência

Omega = wf 


beta = np.sqrt(Omega)*(alpha/mu)**(0.25) #Raiz da equação


#Vetor deslocamentos -2 devido ao engaste na esquerda
U = np.zeros( ( len(beta) , 2*(n_ele+1) - 2) ,dtype = 'cdouble' ) 

for ii in range(len(beta)):
    
    F = np.zeros( (2*(n_ele+1)) ,dtype = 'cdouble')
    F[-2] = 1/(EI/L**2) #Foraçamento unitário na extremidade direita
    
    
    SB = np.zeros(  (2*(n_ele+1) , 2*(n_ele+1) ) ,dtype = 'cdouble') #Matriz de rigidez dinâmica espectral 
    
    #Calcula os termos da SB
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
    
    for kk in range(n_ele):

        SB[0 + 2*kk : 4+ 2*kk , 0 + 2*kk : 4 + 2*kk] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
                                                                   [Lambda3_,Lambda2,-Lambda3,Lambda2_],
                                                                   [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
                                                                   [Lambda3,Lambda2_,-Lambda3_,Lambda2]],dtype='cdouble')

    CC = np.array([0,1]) # CC de engaste na raiz
    for kk in [0,1]:
        SB = np.delete(SB,CC,axis=kk)
        if kk==0:
            F = np.delete(F,CC,axis=kk)
            
    #Solução dos deslocamentos nodais pela inversa
    SBinv = la.inv(SB)     
    U[ii,:] = SBinv@F 
    
    #Solução dos deslocamentos nodais pelo sistema linear
    #U[ii,:] = la.solve(SB,F).T 
    # plt.spy(SB, markersize=8)


# Plot do resultado


plt.figure(1,figsize=(12,8))
  
plt.plot(wf/(2*np.pi),20*np.log10(np.abs(U[:,-2]*L)),'--',color='blue',label='SEM-Validação 2')

plt.xlabel("Frequência (hz)",fontsize=20)
plt.ylabel("Receptância (dB re 1m)",fontsize=20)

plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 20)
plt.grid('on')
plt.show()

U_sem2 = U[:,-2]*L  


    


    


