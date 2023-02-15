# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 09:28:57 2021
FEM Beam
@author: Marcela
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import numpy.linalg as la


### FEM DUTO S/ RESSONADORES 


#Gerando coordenadas e nos de conexões.
L=3; #comprimento da viga
nNo = 60                #numero de nós
numeroEle = nNo-1                   #numero de elementos   
nElevector = np.arange(numeroEle) #1:numeroEle;
nosEle_length = L/numeroEle;
numNoI = np.arange(0,nNo-1,1)       #1:nNo-1;
numNoJ = np.arange(1,nNo,1)         #2:nNo;
Coodx_NoI=  np.arange(0,L,nosEle_length)         #0:nosEle_length:L-nosEle_length;
Coody_NoI=  np.arange(0,L,nosEle_length)*0                            #zeros(1,length(Coodx_NoI)); 
Coodx_NoJ=  np.arange(nosEle_length,L+nosEle_length,nosEle_length)    #nosEle_length:nosEle_length:L;
Coody_NoJ= np.arange(nosEle_length,L+nosEle_length,nosEle_length)*0   #zeros(1,length(Coodx_NoJ)); 

INC=np.vstack((nElevector,     # num elementos 
               numNoI,         # número do nó i
               numNoJ,         # número do nó j 
               Coodx_NoI,      # coordenada x do nó i
               Coody_NoI,      # coordenada y do nó i
               Coodx_NoJ,      # coordenada x do nó j
               Coody_NoJ))    # coordenada y do nó j

# INC = np.array([[0,1,2,3],          # num elementos 
#                 [0,1,2,3],          # número do nó i
#                 [1,2,3,4],          # número do nó j
#                 [0,0.25,0.5,0.75],  # coordenada x do nó i
#                 [0,0,0,0],          #coordenada y do nó i
#                 [0.25,0.5,0.75,1],  # coordenada x do nó j
#                 [0, 0, 0, 0]])      #coordenada y do nó j
# # INC.astype(int)

# Print structure
cx = np.vstack((Coodx_NoI, Coodx_NoJ))    # coordenada x do nó i e j
cy = np.vstack((Coody_NoI, Coody_NoJ))    # coordenada y do nó i e j

# Propriedade do material ------------------------------------

eta = 0.001
E = 210*10**9*(1 +1J*eta)
rhot = 7800
De = 0.2
t = 0.005
Di = De-2*t

Iz = (np.pi/64) * (De**4 - Di**4) 

At = (np.pi/4) * (De**2 - Di**2)
Ai = (np.pi/4) * (Di**2)

rhol = 1000
rhog = 1.18
alpham = 0
rhom = rhol - alpham*(rhol-rhog)

rhoA = (rhot*At + rhom*Ai)

# Alocação ------------------------------------
n = numeroEle               # número de elementos
ngl = 2                   # número de graus de liberdade por nó
GL =(n+1)*ngl               # número de graus de liberdade da estrutura
Kg = np.zeros((GL,GL))     # Matriz de rigidez da estrutura (Global)
Mg = np.zeros((GL,GL))      # Matriz de massa da estrutura (Global)
Ug = np.zeros((GL,1))       # Vetor de deslocamentos (Global)
In = np.eye(GL) 
Fg = np.zeros((GL,1), dtype = 'cdouble')           #Vetor de forças
Fy = 1 #Newton
Fg[-2] = Fy #força aplicada nos devidos graus de liber

# Matrices globais --------------------------------------------
def matrizes(n, INC, E, Iz, rhoA, Kg, Mg, In):
    # Motando a matriz Global
    for jj in range(0,n):
        # print(jj)
        le = np.sqrt((INC[5,jj]-INC[3,jj])**2+(INC[6,jj]-INC[4,jj])**2) # comprimento de cada elemento (l)
        c = int((INC[5,jj]-INC[3,jj])/le)   #direção dos deslocamentos cos(teta) 
        s = int((INC[6,jj]-INC[4,jj])/le)   #direção dos deslocamentos sen(teta)
        #----------------------------------------------------------------
        # Matriz dos cossenos diretores
        Mdir = np.array([[c, s,  0, 0],
                          [-s, c,  0, 0],
                          [0, 0,  c, s],
                          [0, 0, -s, c]],dtype = 'cdouble')
        # Matriz rigidez e massa do element truss
        k = E*Iz/le**3*np.array([[12,   6*le,    -12,    6*le],
                                 [6*le, 4*le**2, -6*le,  2*le**2],
                                 [-12,  -6*le,    12,   -6*le],
                                 [6*le, 2*le**2, -6*le,  4*le**2]],dtype = 'cdouble') #Matriz de rigidez do elemento (Local)
 
        m = (rhoA*le)/420*np.array([[156,     22*le,    54,    -13*le],
                                     [22*le,   4*le**2,  13*le, -3*le**2],
                                     [54,      13*le,    156,   -22*le],
                                     [-13*le,  -3*le**2, -22*le, 4*le**2]],dtype = 'cdouble') #Marriz de massa do elemento  (Local)
        #-----------------------------------------------------------------
        # Matriz do elemento (local) já rotacionada 
        k_re = Mdir.T @ k @ Mdir
        m_re = Mdir.T @ m @ Mdir
        # Transformação da matriz local para a matriz Global do elemento------------------------------
        a = In[2*jj:2*jj+4,:]
        #Matriz da estrutura (Global) = Somatório das matrizes Globais dos elementos------------------
        Kg = Kg + a.T@ k_re @a    #Rigidez
        Mg = Mg + a.T@ m_re @a    #Massa
        # plt.spy(Mg, markersize=4)
        # plt.show()
    return Kg, Mg

K, M = matrizes(n,INC,E,Iz, rhoA,Kg,Mg,In)

# Constains ------------------------------------
C_contorno = np.array([0,1]);
# remove the fixed degrees of freedom
for ii in [0,1]:
 	M = np.delete(M, C_contorno, axis=ii) #ii= 1 delete columns, ii=0 delete rows.
 	K = np.delete(K, C_contorno, axis=ii)
F = np.delete(Fg, C_contorno, 0)




# #% FRF via solucao direta ------------------------------------------------
freq = np.linspace(100,600,2000, endpoint=True)
wf = freq*2*np.pi
U =  np.zeros((len(wf),len(F)))

for i in range(len(wf)):
    U[i,:]=np.linalg.solve(K - wf[i]**2*M,F).T

#Plot results


plt.figure(1,figsize=(12,8))

  
plt.plot(freq,20*np.log10(np.abs(U[:,-2])),color = 'brown', label='MEF')
plt.xlabel("Frequência (hz)",fontsize=20)
plt.ylabel("Receptância (dB re 1m)",fontsize=20)

plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)
plt.legend(fontsize = 20)
plt.grid('on')
plt.show()

U_mef = U[:,-2]

