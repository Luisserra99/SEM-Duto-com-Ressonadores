import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy.optimize import fsolve


#PARAMETROS DO PROBLEMA
eta = 0.001 #Amortecimento estrutural
Rigidez = 46305.35*(1 + 1J*eta)

Di = 50.8*10**(-3) #Diametro interno da tubulação
ht = 3.5*10**(-3) #Espessura da parede da tubulação
De = Di + 2*ht #Diametro externo da tubulação
Areai = np.pi* (Di)**2/4 #Area interna da tubulação
Areat = np.pi* (De)**2/4 - np.pi* (Di)**2/4 # Area da tubulação
rhot = 7270.4 #Massa específica do material da tubulação

rhol = 1000 #Massa específica do líquido
rhog = 1.18 #Massa específica do gás
alpham = 0  #Fração de vazio
rhom = rhol - alpham*(rhol-rhog) #Massa espscífica da mistura

rhoA = (rhot*Areat + rhom*Areai) #Densidade linear de massa
EI = Rigidez #Rigidez flexural

L_extremidade_e1 = 0.2 #Comprimento do laod direito igual o do esquerdo -d da relação de massa do ressonador
L_extremidade_e2 = 1.1 #Comprimento do laod direito igual o do esquerdo -d da relação de massa do ressonador
L_extremidade_d1 = 1.1 #Tamanho da extremidade direita
L_extremidade_d2 = 0.2 #Tamanho da extremidade direita
L_ressonadores = 0.2 #Distânica entre os ressonadores

L = [L_extremidade_e1,L_extremidade_e2,L_ressonadores,L_ressonadores,L_ressonadores,L_ressonadores,L_extremidade_d1,L_extremidade_d2] #Tamanhos dos elementos ao longo da Viga


#### PARÂMETROS DO RESSONADOR ######
Nr = 3

porcentage_massa = 0.2

m = rhoA*sum(L)*porcentage_massa/Nr
w1 = 450*2*np.pi
w2 = 470*2*np.pi
deltaf = (w2-w1)/(w1*(2-1))
k1 = m*w1**2

qsi = 0.005



### Elementos Espectrais do ressonador
def ressonador(N,Omegaj, m,w1,deltaf,k1,qsi):
    
    
    Lambda5j = k1*((1 + (N-1)*deltaf)**2 + 2J*Omegaj*((1 + (N-1)*deltaf)**2)*qsi )
    
    Lambda4j = k1*((1 + (N-1)*deltaf)**2 - Omegaj**2 + 2J*Omegaj*((1 + (N-1)*deltaf)**2)*qsi)
    
    return [Lambda4j,Lambda5j]
    
### Elementos Espectrais da Viga
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

wf = np.linspace(250*2*np.pi,500*2*np.pi,2000,endpoint=True)#Intervalo de frequência

Omega = wf/w1 #Frequência admensional

###### SOLUÇÃO ######
CC = np.array([0,-2])
    

U = np.zeros( ( len(wf) , int(6*2 + 3*(Nr))-len(CC) ),dtype='cdouble' ) #Vetor de deslocamentos

for ii in range(len(wf)):    
    
    F = np.zeros( ( 6*2 + 3*(Nr) ),dtype='cdouble' ) #Vetor de forçamentos
    
    
    SVM = np.zeros( ( 6*2 + 3*(Nr) , 6*2 + 3*(Nr) ),dtype='cdouble' ) #Matriz de rigidez dinâmica espectral
    
    
    
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[0], m,rhoA,EI,k1)
    
    SVM[0:4,0:4] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
                                  [Lambda3_,Lambda2,-Lambda3,Lambda2_],
                                  [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
                                  [Lambda3,Lambda2_,-Lambda3_,Lambda2] ], dtype = 'cdouble')
    
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[1], m,rhoA,EI,k1)
    
    SVM[2:6,2:6] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
                                  [Lambda3_,Lambda2,-Lambda3,Lambda2_],
                                  [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
                                  [Lambda3,Lambda2_,-Lambda3_,Lambda2] ], dtype = 'cdouble')
    
    for jj in range(Nr):
        
        Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[jj+2], m,rhoA,EI,k1)
        
        Lambda4,Lambda5 = ressonador(jj+1,Omega[ii], m,w1,deltaf,k1,qsi)
        
        if jj==0:
            SVM[4:9,4:9] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3,0],
                                          [Lambda3_,Lambda2,-Lambda3,Lambda2_,0],
                                          [-Lambda1_,-Lambda3,Lambda1+Lambda5,-Lambda3_,-Lambda5],
                                          [Lambda3,Lambda2_,-Lambda3_,Lambda2,0],
                                          [0,0,-Lambda5,0,Lambda4]])
        else:
            SVM[int(4+jj*(2) + jj - 1):int(4+jj*(2) + jj - 1+2),int(4+jj*(2) + jj - 1):int(4+jj*(2) + jj - 1 + 2)] +=  np.array([ [Lambda1, Lambda3_],
                                                                                                                [Lambda3_,Lambda2]])
            
            SVM[int(4+jj*(2) + jj - 1):int(4+jj*(2) + jj - 1+2),int(4+jj*(2) + jj - 1 + 3):int(4+jj*(2) + jj - 1 + 5)] +=  np.array([ [-Lambda1_,Lambda3],
                                                                                                                    [-Lambda3,Lambda2_]])
            

            SVM[int(4+2 + jj*(3)):int(4+5 + jj*(3)),int(4+2 + jj*3 -3 ):int(4+2 + jj*3 -1)] +=  np.array([[-Lambda1_,-Lambda3],
                                                                                         [Lambda3,Lambda2_],
                                                                                         [0,0] ])
            
            SVM[int(4+2 + jj*(3)):int(4+5 + jj*(3)),int(4+2 + jj*3 ):int(4+2 + jj*3 + 3)] +=  np.array([[Lambda1+Lambda5,-Lambda3_,-Lambda5],
                                                                                       [-Lambda3_,Lambda2,0],
                                                                                       [-Lambda5,0,Lambda4] ])
        
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[-3], m,rhoA,EI,k1)
    
    
    SVM[-9:-7,-9:-7] +=  [ [Lambda1, Lambda3_],
                          [Lambda3_,Lambda2]]
    
    SVM[-9:-7,-6:-4] +=  [ [-Lambda1_,Lambda3],
                        [-Lambda3,Lambda2_]]
    

    SVM[-6:-4,-9:-7] +=  [[-Lambda1_,-Lambda3],
                        [Lambda3,Lambda2_]]
    
    SVM[-6:-4,-6:-4] +=  [[Lambda1,-Lambda3_],
                      [-Lambda3_,Lambda2] ]
    
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[-2], m,rhoA,EI,k1)
    
    SVM[-6:-2,-6:-2] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
                                  [Lambda3_,Lambda2,-Lambda3,Lambda2_],
                                  [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
                                  [Lambda3,Lambda2_,-Lambda3_,Lambda2] ], dtype = 'cdouble')
    
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[-1], m,rhoA,EI,k1)
    
    SVM[-4:,-4:] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
                                  [Lambda3_,Lambda2,-Lambda3,Lambda2_],
                                  [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
                                  [Lambda3,Lambda2_,-Lambda3_,Lambda2] ], dtype = 'cdouble')
    
    #Forçamento
    F[2] = -1
    #F[-4] = -1
    for kk in [0,1]:
        SVM = np.delete(SVM,CC,axis=kk)
        if kk==0:
            F = np.delete(F,CC,axis=kk)
    U[ii,:] = la.solve(SVM,F).T

####Solução homogênea

m = 1*10**(-8)
k1 = m*w1**2

qsi = 0.005

Uanlitico = np.zeros( ( len(wf) , int(6*2 + 3*(Nr))-len(CC) ),dtype='cdouble' ) #Vetor de deslocamentos

for ii in range(len(wf)):    
    
    F = np.zeros( ( 6*2 + 3*(Nr) ),dtype='cdouble' ) #Vetor de forçamentos
    
    
    SVM = np.zeros( ( 6*2 + 3*(Nr) , 6*2 + 3*(Nr) ),dtype='cdouble' ) #Matriz de rigidez dinâmica espectral
    
    
    
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[0], m,rhoA,EI,k1)
    
    SVM[0:4,0:4] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
                                  [Lambda3_,Lambda2,-Lambda3,Lambda2_],
                                  [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
                                  [Lambda3,Lambda2_,-Lambda3_,Lambda2] ], dtype = 'cdouble')
    
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[1], m,rhoA,EI,k1)
    
    SVM[2:6,2:6] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
                                  [Lambda3_,Lambda2,-Lambda3,Lambda2_],
                                  [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
                                  [Lambda3,Lambda2_,-Lambda3_,Lambda2] ], dtype = 'cdouble')
    
    for jj in range(Nr):
        
        Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[jj+2], m,rhoA,EI,k1)
        
        Lambda4,Lambda5 = ressonador(jj+1,Omega[ii], m,w1,deltaf,k1,qsi)
        
        if jj==0:
            SVM[4:9,4:9] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3,0],
                                          [Lambda3_,Lambda2,-Lambda3,Lambda2_,0],
                                          [-Lambda1_,-Lambda3,Lambda1+Lambda5,-Lambda3_,-Lambda5],
                                          [Lambda3,Lambda2_,-Lambda3_,Lambda2,0],
                                          [0,0,-Lambda5,0,Lambda4]])
        else:
            SVM[int(4+jj*(2) + jj - 1):int(4+jj*(2) + jj - 1+2),int(4+jj*(2) + jj - 1):int(4+jj*(2) + jj - 1 + 2)] +=  np.array([ [Lambda1, Lambda3_],
                                                                                                                [Lambda3_,Lambda2]])
            
            SVM[int(4+jj*(2) + jj - 1):int(4+jj*(2) + jj - 1+2),int(4+jj*(2) + jj - 1 + 3):int(4+jj*(2) + jj - 1 + 5)] +=  np.array([ [-Lambda1_,Lambda3],
                                                                                                                    [-Lambda3,Lambda2_]])
            

            SVM[int(4+2 + jj*(3)):int(4+5 + jj*(3)),int(4+2 + jj*3 -3 ):int(4+2 + jj*3 -1)] +=  np.array([[-Lambda1_,-Lambda3],
                                                                                         [Lambda3,Lambda2_],
                                                                                         [0,0] ])
            
            SVM[int(4+2 + jj*(3)):int(4+5 + jj*(3)),int(4+2 + jj*3 ):int(4+2 + jj*3 + 3)] +=  np.array([[Lambda1+Lambda5,-Lambda3_,-Lambda5],
                                                                                       [-Lambda3_,Lambda2,0],
                                                                                       [-Lambda5,0,Lambda4] ])
        
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[-3], m,rhoA,EI,k1)
    
    
    SVM[-9:-7,-9:-7] +=  [ [Lambda1, Lambda3_],
                          [Lambda3_,Lambda2]]
    
    SVM[-9:-7,-6:-4] +=  [ [-Lambda1_,Lambda3],
                        [-Lambda3,Lambda2_]]
    

    SVM[-6:-4,-9:-7] +=  [[-Lambda1_,-Lambda3],
                        [Lambda3,Lambda2_]]
    
    SVM[-6:-4,-6:-4] +=  [[Lambda1,-Lambda3_],
                      [-Lambda3_,Lambda2] ]
    
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[-2], m,rhoA,EI,k1)
    
    SVM[-6:-2,-6:-2] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
                                  [Lambda3_,Lambda2,-Lambda3,Lambda2_],
                                  [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
                                  [Lambda3,Lambda2_,-Lambda3_,Lambda2] ], dtype = 'cdouble')
    
    Lambda1,Lambda2,Lambda3,Lambda1_,Lambda2_,Lambda3_ = gammas(Omega[ii],L[-1], m,rhoA,EI,k1)
    
    SVM[-4:,-4:] += np.array([ [Lambda1, Lambda3_,-Lambda1_,Lambda3],
                                  [Lambda3_,Lambda2,-Lambda3,Lambda2_],
                                  [-Lambda1_,-Lambda3,Lambda1,-Lambda3_],
                                  [Lambda3,Lambda2_,-Lambda3_,Lambda2] ], dtype = 'cdouble')
    
    #Forçamento
    F[2] = -1
    #F[-4] = -1
    for kk in [0,1]:
        SVM = np.delete(SVM,CC,axis=kk)
        if kk==0:
            F = np.delete(F,CC,axis=kk)
    Uanlitico[ii,:] = la.solve(SVM,F).T



### COMPARAÇÃO DE AMPLITUDES Viga/viga homogenea###
plt.figure(1,figsize=(12,10))



plt.plot(wf/(2*np.pi),np.abs( np.real(U[:,5])/np.real(Uanlitico[:,5]) ),color = 'green',label=str(r'$|W_{1}/W_{1,homogêneo}|$') )
plt.plot(wf/(2*np.pi),np.abs(np.real(U[:,8])/np.real(Uanlitico[:,8]) ),color='orange',label=str(r'|$W_{2}/W_{2,homogêneo}$|') )
plt.plot(wf/(2*np.pi),np.abs(np.real(U[:,11])/np.real(Uanlitico[:,11])) ,color='blue',label=str(r'$|W_{3}/W_{3,homogêneo}|$') )


plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Frequência (hz)',fontsize=22)
plt.ylabel(r'Amplitude relativa $(|W_j/W_{j,homogêneo}|)$',fontsize=22) 
plt.xlim(250,500)
plt.ylim(1,10)
plt.grid('on')
plt.legend(fontsize=20)



# ### COMPARAÇÃO DE AMPLITUDES Meta-viga###
plt.figure(2,figsize=(12,10))


# #Forçamento pela Esquerda
# plt.plot(wf/(2*np.pi),np.abs( np.real(U[:,5])/np.real(U[:,3]) ),color = 'green',label=str(r'$|W_{1}/W_{0}|$') )
# plt.plot(wf/(2*np.pi),np.abs(np.real(U[:,8])/np.real(U[:,3]) ),color='orange',label=str(r'$|W_{2}/W_{0}|$') )
# plt.plot(wf/(2*np.pi),np.abs(np.real(U[:,11])/np.real(U[:,3])) ,color='blue',label=str(r'$|W_{3}/W_{0}|$') )

# #Forçamento pela Direita
plt.plot(wf/(2*np.pi),np.abs( np.real(U[:,5])/np.real(U[:,14]) ),color = 'green',label=str(r'$|W_{1}/W_{0}|$') )
plt.plot(wf/(2*np.pi),np.abs(np.real(U[:,8])/np.real(U[:,14]) ),color='orange',label=str(r'$|W_{2}/W_{0}|$') )
plt.plot(wf/(2*np.pi),np.abs(np.real(U[:,11])/np.real(U[:,14])) ,color='blue',label=str(r'$|W_{3}/W_{0}|$') )

plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Frequência (hz)',fontsize=22)
plt.ylabel(r'Amplitude realtiva $(|W_{j}/W_{0}|)$',fontsize=22) 

plt.xlim(250,500)
plt.yscale('log',base=10)
plt.grid('on')
plt.legend(fontsize=20)


###Parte real da amplitude ######

plt.figure(3,figsize=(12,10))

#Esquerda
plt.plot(wf/(2*np.pi), np.real(U[:,5])/np.real(U[:,3]) ,color = 'green',label=str(r'$W_{1}/W_{0}$') )
plt.plot(wf/(2*np.pi),np.real(U[:,8])/np.real(U[:,3]) ,color='orange',label=str(r'$W_{2}/W_{0}$') )
plt.plot(wf/(2*np.pi),np.real(U[:,11])/np.real(U[:,3]) ,color='blue',label=str(r'$W_{3}/W_{0}$') )

#Direita
# plt.plot(wf/(2*np.pi), np.real(U[:,5])/np.real(U[:,14]) ,color = 'green',label=str(r'$|W_{1}/W_{0}|$') )
# plt.plot(wf/(2*np.pi),np.real(U[:,8])/np.real(U[:,14]) ,color='orange',label=str(r'$|W_{2}/W_{0}|$') )
# plt.plot(wf/(2*np.pi),np.real(U[:,11])/np.real(U[:,14]) ,color='blue',label=str(r'$|W_{3}/W_{0}|$') )



plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Frequência (hz)',fontsize=22)
plt.ylabel(r'Amplitude relativa $(W_{j}/W_{0})$',fontsize=22) 
plt.xlim(250,500)
plt.ylim(-15,15)
plt.grid('on')
plt.legend(fontsize=20)


# ### COMPARAÇÃO DE AMPLITUDES Meta-viga W3/W1###
plt.figure(4,figsize=(12,10))


# #Forçamento pela Esquerda
plt.plot(wf/(2*np.pi),np.abs( np.real(U[:,11])/np.real(U[:,5]) ),color = 'green',label=str(r'$|W_{1}/W_{0}|$') )

# #Forçamento pela Direita
#plt.plot(wf/(2*np.pi),np.abs( np.real(U[:,11])/np.real(U[:,5])*100 ),color = 'green',label=str('R1') )


plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('Frequência (hz)',fontsize=22)
plt.ylabel(r'Amplitude realtiva $(|W_{j}/W_{0}|)$',fontsize=22) 

plt.xlim(300,400)
plt.yscale('log',base=10)
plt.grid('on')
plt.legend(fontsize=20)


### NÚMERO DE ONDA ###

###ONDA ANALITICA SEM RESSONADOR###
K_onda_r = np.sqrt(wf)*(rhoA/(EI))**0.25


### NÚMERO DE ONDA USANDO 2 PONTOS W e theta
def calculo_VigaR(x,W0,W1,Theta0,Theta1,d):
    A1 = np.log((Theta1 + 1j*x*W1)/((Theta0 + 1j*x*W0)))
    A2 = (1J*x*d)
    return np.imag(A1-A2)


Kw3R = np.zeros( len(wf) , dtype = 'double')
Kw2R = np.zeros( len(wf) , dtype = 'double')
Kw1R = np.zeros( len(wf) , dtype = 'double')

Kw3Ra = np.zeros( len(wf) , dtype = 'cdouble')
Kw2Ra = np.zeros( len(wf) , dtype = 'cdouble')
Kw1Ra = np.zeros( len(wf) , dtype = 'cdouble')

w3 = w1*deltaf*(3-1) + w1
var_epsilon = (sum(L)*porcentage_massa/Nr)/L_ressonadores

for ii in range(len(wf)):
    Kw3R[ii] = fsolve(calculo_VigaR,args=(U[ii,-8],U[ii,-5],U[ii,-7],U[ii,-4],L_ressonadores),x0=6+4*wf[ii]/wf[-1])
    Kw2R[ii] = fsolve(calculo_VigaR,args=(U[ii,-11],U[ii,-8],U[ii,-10],U[ii,-7],L_ressonadores),x0=6+4*wf[ii]/wf[-1])
    Kw1R[ii] = fsolve(calculo_VigaR,args=(U[ii,-14],U[ii,-11],U[ii,-13],U[ii,-10],L_ressonadores),x0=6+4*wf[ii]/wf[-1])
    
    Kw1Ra[ii] = ( (rhoA/(EI))*wf[ii]**2*(1 + var_epsilon*1/(1 - (wf[ii]/w1)**2 ) )  )**0.25
    Kw2Ra[ii] = ( (rhoA/(EI))*wf[ii]**2*(1 + var_epsilon*1/(1 - (wf[ii]/w2)**2 ) )  )**0.25
    Kw3Ra[ii] = ( (rhoA/(EI))*wf[ii]**2*(1 + var_epsilon*1/(1 - (wf[ii]/w3)**2 ) )  )**0.25


####PLOT NUMERO DE ONDA

plt.figure(5,figsize=(12,8))
plt.plot(wf/(2*np.pi),Kw3R*sum(L)/(2*np.pi),color='blue',label = r'$k_{r3}$')
plt.plot(wf/(2*np.pi),Kw2R*sum(L)/(2*np.pi),color='orange',label = r'$k_{r2}$')
plt.plot(wf/(2*np.pi),Kw1R*sum(L)/(2*np.pi),color='green',label = r'$k_{r1}$')
plt.plot(wf/(2*np.pi),K_onda_r*sum(L)/(2*np.pi),color='red',label = r'$k_{homogêneo}$')

plt.plot(wf/(2*np.pi),np.real(Kw3Ra)*sum(L)/(2*np.pi),'--',color='blue',label = r'$k_{ra3}$')
plt.plot(wf/(2*np.pi),np.real(Kw2Ra)*sum(L)/(2*np.pi),'--',color='orange',label = r'$k_{ra2}$')
plt.plot(wf/(2*np.pi),np.real(Kw1Ra)*sum(L)/(2*np.pi),'--',color='green',label = r'$k_{ra1}$')


plt.ylabel('Número de onda',fontsize=20)
plt.xlabel('Frequência (hz)',fontsize=20)
plt.xticks(size=20)
plt.yticks(size=20)
plt.grid('on')
plt.ylim(0,5)
plt.xlim(250,400)
plt.legend(fontsize=18,loc='lower right')




    



