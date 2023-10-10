# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:10:46 2023

@author: Edwin
"""
    
def get_momentequalities(A,B,D,etaA,etaBS,etaBL,chsh_or_zx):
    equalities = []
    v = visibility
    if binning:
        
        for x in range(nX):
            equalities += [A[x] - (1-etaA) ]      
            
        for y in range(nY):
            equalities += [B[y] - (1-etaBS)]
            equalities += [D[y] - (1-etaBL)]

            
        for x in range(nX):
            for y in range(nY):
                equalities += [A[x]*B[y] - (v**2* etaA*etaBS* (-1)**(x*y) * 1/sqrt(2) + (1-etaA)*(1-etaBS) ) ]

                
                ### CHSH-ZX ##
                if chsh_or_zx == 'zx':
                    if x == y:
                        equalities += [A[x]*D[y] - ( v**2 * etaA*etaBL +  (1-etaA)*(1-etaBL) )]
                    else:
                        equalities+= [A[x]*D[y] -  (1-etaA)*(1-etaBL)  ]
                
                ## CHSH-CHSH ##
                if chsh_or_zx == 'chsh':
                    equalities += [A[x]*D[y] -  ( v**2*etaA*etaBL* (-1)**(x*y) * 1/sqrt(2) + (1-etaA)*(1-etaBL) ) ]    
    
    else:

        for x in range(nX):
            equalities += [A[x] ]
            equalities += [A[x]**2 - etaA]
            
            
            
        for y in range(nY):
            equalities += [B[y] ]
            equalities += [B[y]**2 - etaBS]
            equalities += [D[y]]
            equalities += [D[y]**2 - etaBL]
            
            
            
        for x in range(nX):
            for y in range(nY):
                equalities += [A[x]*B[y] -( v**2*(-1)**(x*y) *etaA*etaBS*1/(sqrt(2)) )]
                equalities += [A[x]*B[y]**2 ]
                equalities += [A[x]**2*B[y] ]
                equalities += [A[x]**2*B[y]**2 - etaA*etaBS]
                
                ### CHSH-ZX ##
                if chsh_or_zx == 'zx':
                    if x == y:
                        equalities += [A[x]*D[y] - v**2*etaA*etaBL]
                    else:
                        equalities+= [A[x]*D[y]]
                
                ## CHSH-CHSH ##
                if chsh_or_zx == 'chsh':
                    equalities += [A[x]*D[y] - ( v**2*(-1)**(x*y) *etaA*etaBL*1/(sqrt(2)))]
    
                equalities += [A[x]*D[y]**2 ]
                equalities += [A[x]**2*D[y]]
                equalities += [A[x]**2*D[y]**2 - etaA*etaBL]
                
    return equalities

    
def objective():
    return -etaBL


def get_subs(A,B,D):
    
    ## D0 and D1 commute##
    subs = {D[1]*D[0]:D[0]*D[1]}
    
    ## Ax and By commute##
    for a in A:
        for b in B:
            subs.update({b*a:a*b})
    
    ## Ax and Dy commute##
    for a in A:
        for d in D:
            subs.update({d*a:a*d})
            
    #other constraints ##
    if binning:
        for o in ncp.flatten([A,B,D]):
            subs.update({o**2 : 1})
    else:
        for o in ncp.flatten([A,B,D]):
            subs.update({o**3 : o})
        
    return subs


"""
Now we start with setting up the ncpol2sdpa computations
"""
import numpy as np
from math import sqrt
import ncpol2sdpa as ncp
import matplotlib.pyplot as plt


nA = 2
nB = 2
nX = 2
nY = 2
nZ = 2
d=2
chsh_or_zx = 'chsh'
binning = False
visibility = 0.94

if chsh_or_zx == 'chsh':
    if binning:
        etaA_list = [1]#np.linspace(0.96,1,20)
    else:
        etaA_list = [1] # np.linspace(0.9,0.92,3).tolist() + np.linspace(0.92,0.94,4)[1:].tolist() + np.linspace(0.94,0.96,5)[1:].tolist()  + np.linspace(0.96,0.98,7)[1:].tolist() + \
            # np.linspace(0.98,0.99,7)[1:].tolist() + np.linspace(0.99,1,8)[1:].tolist()

if chsh_or_zx == 'zx':
    if binning:
        etaA_list = [1]#np.linspace(0.9586,1,10)
    else:
        etaA_list = [1] #np.linspace(0.8942,0.8955,5).tolist() + np.linspace(0.8955,1,10)[1:].tolist() 
    



A0 = ncp.generate_operators('A0',1,hermitian=True)[0]
A1 = ncp.generate_operators('A1',1,hermitian=True)[0]
B0 = ncp.generate_operators('B0',1,hermitian=True)[0]
B1 = ncp.generate_operators('B1',1,hermitian=True)[0]
D0 = ncp.generate_operators('D0',1,hermitian=True)[0]
D1 = ncp.generate_operators('D1',1,hermitian=True)[0]

A = [A0,A1]
B = [B0,B1]
D = [D0,D1]

if binning:
    etaBL = 1 - D0
else:
    etaBL = D[0]**2

obj = objective()

if binning:
    level1 = ncp.flatten([A,B,D])
    
    AA_terms = [A0*A1, A1*A0]
    BB_terms = [B0*B1, B1*B0]
    DD_terms = [D0*D1, D1*D0]
    AB_terms = [a*b for a in A for b in B]
    AD_terms = [a*d for a in A for d in D]
    BD_terms = [b*d for b in B for d in D]
    
    level2 = level1 \
        +AA_terms\
            +BB_terms\
                +DD_terms\
                    +AB_terms\
                        +AD_terms\
                            +BD_terms
                            
    ABD_terms = [a*b*d for a in A for b in B for d in D]
    ADB_terms = [a*d*b for a in A for d in D for b in B]  
    
    AAB_terms = [aa*b for aa in AA_terms for b in B]
    AAD_terms = [aa*d for aa in AA_terms for d in D]
    
    ABB_terms = [a*bb for a in A for bb in BB_terms]
    DBB_terms = [d*bb for d in D for bb in BB_terms]
    BDB_terms = [b*d*b_ for b in B for d in D for b_ in B]
    BBD_terms = [bb*d for bb in BB_terms for d in D]
    
    ADD_terms = [a*dd for a in A for dd in DD_terms]
    BDD_terms = [b*dd for b in B for dd in DD_terms]
    DBD_terms = [d*b*d_ for d in D for b in B for d_ in D]
    DDB_terms = [dd*b for dd in DD_terms for b in B]
    
    AAA_terms = [A0*A1*A0, A1*A0*A1]
    BBB_terms = [B0*B1*B0, B1*B0*B1]
    DDD_terms = [D0*D1*D0, D1*D0*D1]
    
    level3 = level2 \
        +ABD_terms\
            +ADB_terms\
                +AAB_terms\
                    +AAD_terms\
                        +ABB_terms\
                            +DBB_terms\
                                +BDB_terms\
                                    +BBD_terms\
                                        +ADD_terms\
                                            +BDD_terms\
                                                +DBD_terms\
                                                    +DDB_terms\
                                                        +AAA_terms\
                                                            +BBB_terms\
                                                                +DDD_terms
    AABB_terms = [aa*bb for aa in AA_terms for bb in BB_terms]
    AADD_terms = [aa*dd for aa in AA_terms for dd in DD_terms]
    BBDD_terms = [bb*dd for bb in BB_terms for dd in DD_terms]
    BDBD_terms = [b*d*b_*d_ for b in B for d in D for b_ in B for d_ in D]
    
    AAAA_terms = [A0*A1*A0*A1, A1*A0*A1*A0]
    BBBB_terms = [B0*B1*B0*B1, B1*B0*B1*B0]
    DDDD_terms = [D0*D1*D0*D1, D1*D0*D1*D0]

    level4 = level3\
        +AAAA_terms\
            +BBBB_terms\
                +DDDD_terms\
                        +AADD_terms\
                            +BBDD_terms\
                                +AABB_terms\
                                    # +BDBD_terms
    
else:
    level1 = ncp.flatten([A,B,D])
    
    AA_terms = [a*a_ for a in A for a_ in A]
    BB_terms = [b*b_ for b in B for b_ in B]
    DD_terms = [D0**2, D0*D1, D1**2]  
    AB_terms = [a*b for a in A for b in B]
    AD_terms = [a*d for a in A for d in D]
    BD_terms = [b*d for b in B for d in D]
        
    level2 = level1 \
        +AA_terms\
            +BB_terms\
                +DD_terms\
                    +AB_terms\
                        +AD_terms\
                            +BD_terms
    
    ABD_terms = [a*b*d for a in A for b in B for d in D]
    ADB_terms = [a*d*b for a in A for d in D for b in B]  
    
    AAB_terms = [aa*b for aa in AA_terms for b in B]
    AAD_terms = [aa*d for aa in AA_terms for d in D]
    
    ABB_terms = [a*bb for a in A for bb in BB_terms]
    DBB_terms = [d*bb for d in D for bb in BB_terms]
    BDB_terms = [b*d*b_ for b in B for d in D for b_ in B]
    BBD_terms = [bb*d for bb in BB_terms for d in D]
    
    ADD_terms = [a*dd for a in A for dd in DD_terms]
    BDD_terms = [b*dd for b in B for dd in DD_terms]
    DBD_terms = [d*b*d_ for d in D for b in B for d_ in D]
    DDB_terms = [dd*b for dd in DD_terms for b in B]
    
    AAA_terms = [A0**2*A1, A0*A1*A0, A0*A1**2, A1*A0**2, A1*A0*A1, A1**2*A0]
    BBB_terms = [B0**2*B1, B0*B1*B0, B0*B1**2, B1*B0**2, B1*B0*B1, B1**2*B0]
    DDD_terms = [D0**2*D1,  D0*D1**2]        

                              
    level3 = level2 \
        +ABD_terms\
            +ADB_terms\
                +AAB_terms\
                    +AAD_terms\
                        +ABB_terms\
                            +DBB_terms\
                                +BDB_terms\
                                    +BBD_terms\
                                        +ADD_terms\
                                            +BDD_terms\
                                                +DBD_terms\
                                                    +DDB_terms\
                                                        +AAA_terms\
                                                            +BBB_terms\
                                                                +DDD_terms
                                                            
        
    AAAA_terms = [A0**2*A1*A0, A0**2*A1**2, A0*A1*A0**2, A0*A1*A0*A1, A0*A1**2*A0, A1*A0**2*A1, A1*A0*A1*A0, A1*A0*A1**2, A1**2*A0**2, A1**2*A0*A1]
    BBBB_terms = [B0**2*B1*B0, B0**2*B1**2, B0*B1*B0**2, B0*B1*B0*B1, B0*B1**2*B0, B1*B0**2*B1, B1*B0*B1*B0, B1*B0*B1**2, B1**2*B0**2, B1**2*B0*B1]
    DDDD_terms = [D0**2*D1**2]
    
    AABB_terms = [aa*bb for aa in AA_terms for bb in BB_terms]
    AADD_terms = [aa*dd for aa in AA_terms for dd in DD_terms]
    BBDD_terms = [bb*dd for bb in BB_terms for dd in DD_terms]
    BDBD_terms = [b*d*b_*d_ for b in B for d in D for b_ in B for d_ in D]
    
    level4 = level3\
        +AAAA_terms\
            +BBBB_terms\
                +DDDD_terms\
                        +AADD_terms\
                            +BBDD_terms\
                                +AABB_terms\
                                    #+BDBD_terms\

                                    

subs = get_subs(A,B,D) 
sdp = ncp.SdpRelaxation(ncp.flatten([level4]), normalized=True, verbose=False)
if binning:
    t = 14 ##Number of independent elements in p(ab|xyz)
else:
    t = 44 ##Number of independent elements in p(ab|xyz)
sdp.get_relaxation(level = 1,
                    substitutions = subs,
                    objective=obj,
                    momentequalities = [A0 for j in range(t)])


etaBL_list = []
for etaA in etaA_list:
    etaBS = etaA
    sdp.process_constraints(momentequalities = get_momentequalities(A, B, D,etaA,etaBS,etaBL, chsh_or_zx))
    # prob = sdp.convert_to_cvxpy()
    # prob.solve(verbose = False)
    sdp.solve()
    etaBL_list += [-sdp.primal]
    # etaBL_list += [-prob.value]
    print(etaA, -sdp.primal)
    # print(etaA, -prob.value)


# plt.plot(etaA_list,etaBL_list,'r-' , linewidth=2)
# plt.grid(color='r', linestyle='-', linewidth=1)


# # save etaA_list and etaBL_list to text file
# np.savetxt('etaA_list.txt', etaA_list)
# np.savetxt('etaBL_list.txt', etaBL_list)




###### Discarded Code ### 
"""
def get_momentequalities(p_obs,A,B,D,etaA,etaBS,etaBL,chsh_or_zx):
    equalities = []
    if binning:
        
        for x in range(nX):
            equalities += [A[x] - (1-etaA) ]      
            
        for y in range(nY):
            equalities += [B[y] - (1-etaBS)]
            equalities += [D[y] - (1-etaBL)]

            
        for x in range(nX):
            for y in range(nY):
                equalities += [A[x]*B[y] - ( etaA*etaBS* (-1)**(x*y) * 1/sqrt(2) + (1-etaA)*(1-etaBS) ) ]

                
                ### CHSH-ZX ##
                if chsh_or_zx == 'zx':
                    if x == y:
                        equalities += [A[x]*D[y] - ( etaA*etaBL +  (1-etaA)*(1-etaBL) )]
                    else:
                        equalities+= [A[x]*D[y]]
                
                ## CHSH-CHSH ##
                if chsh_or_zx == 'chsh':
                    equalities += [A[x]*D[y] -  ( etaA*etaBL* (-1)**(x*y) * 1/sqrt(2) + (1-etaA)*(1-etaBL) ) ]    
    
    else:
        # A_obs,Asq_obs,B_obs,Bsq_obs,D_obs,Dsq_obs,AB_obs,ABsq_obs,AsqB_obs,AsqBsq_obs,\
        #     AD_obs,ADsq_obs,AsqD_obs,AsqDsq_obs = get_obs_correlators(p_obs)
        for x in range(nX):
            equalities += [A[x] ]
            equalities += [A[x]**2 - etaA]
            
            # equalities += [A[x] - A_obs[x]]
            # equalities += [A[x]**2 - Asq_obs[x]]
            
            
        for y in range(nY):
            equalities += [B[y] ]
            equalities += [B[y]**2 - etaBS]
            equalities += [D[y]]
            equalities += [D[y]**2 - etaBL]
            
            # equalities += [B[y] -  B_obs[y]]
            # equalities += [B[y]**2 - Bsq_obs[y] ]
            # equalities += [D[y] -  D_obs[y] ]
            # equalities += [D[y]**2 - Dsq_obs[y]]
            
            
        for x in range(nX):
            for y in range(nY):
                equalities += [A[x]*B[y] -( (-1)**(x*y) *etaA*etaBS*1/(sqrt(2)))]
                equalities += [A[x]*B[y]**2 ]
                equalities += [A[x]**2*B[y] ]
                equalities += [A[x]**2*B[y]**2 - etaA*etaBS]
                
                ### CHSH-ZX ##
                if chsh_or_zx == 'zx':
                    if x == y:
                        equalities += [A[x]*D[y] - etaA*etaBL]
                    else:
                        equalities+= [A[x]*D[y]]
                
                ## CHSH-CHSH ##
                if chsh_or_zx == 'chsh':
                    equalities += [A[x]*D[y] - ( (-1)**(x*y) *etaA*etaBL*1/(sqrt(2)))]
    
                equalities += [A[x]*D[y]**2 ]
                equalities += [A[x]**2*D[y]]
                equalities += [A[x]**2*D[y]**2 - etaA*etaBL]
                
                # equalities += [A[x]*B[y] - AB_obs[x,y]]
                # equalities += [A[x]*B[y]**2 - ABsq_obs[x,y]]
                # equalities += [A[x]**2*B[y] - AsqB_obs[x,y]]
                # equalities += [A[x]**2*B[y]**2 - AsqBsq_obs[x,y]]
                
                # equalities += [A[x]*D[y] - AD_obs[x,y]]
                # equalities += [A[x]*D[y]**2 - ADsq_obs[x,y]]
                # equalities += [A[x]**2*D[y] - AsqD_obs[x,y]]
                # equalities += [A[x]**2*D[y]**2 - AsqDsq_obs[x,y]]


        
    return equalities

def get_p_ideal():
    MA=np.zeros((nA,nX), dtype=object)
    MB=np.zeros((nB,nY), dtype=object)
    [MA[0,0], MA[1,0], MA[0,1], MA[1,1]] = [(I+Z)/2 ,(I-Z)/2, (I+X)/2, (I-X)/2]  
    [MB[0,0], MB[1,0], MB[0,1], MB[1,1]] = [(I+ZR)/2 ,(I-ZR)/2, (I+XR)/2, (I-XR)/2]

    bchsh = 0
    pchsh = np.zeros((nA,nB,nX,nY))
    psteer= np.zeros((nA,nB,nX,nY))
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                    pchsh[a,b,x,y] = np.trace(phip @ np.kron(MA[a,x],MB[b,y]))
                    bchsh += (-1)**(a+b+(x)*(y)) * pchsh[a,b,x,y]
                    psteer[a,b,x,y] = np.trace(phip @ np.kron(MA[a,x],MA[b,y]))

    return pchsh,psteer,bchsh
[I, X, Y, Z] = [qtp.qeye(2)[:], qtp.sigmax()[:], qtp.sigmay()[:], qtp.sigmaz()[:]]
[ZR , XR] = [(Z+X)/sqrt(2) , (Z-X)/sqrt(2)]

phip=qtp.bell_state('00')[:] @ qtp.bell_state('00')[:].T.conj()

def get_obs_correlators(p_obs):
    p, pA, pB = ind_elements(p_obs,'ncpol')
        
    
    A_obs = np.zeros(nX, dtype=object)
    for x in range(nX):
        for a in range(nA):
            A_obs[x] = A_obs[x] + (-1)**a * pA[a,x]
            
    Asq_obs = np.zeros(nX, dtype=object)
    for x in range(nX):
        for a in range(nA):
            Asq_obs[x] = Asq_obs[x] +  pA[a,x]
    
    B_obs = np.zeros(nY, dtype=object)
    for y in range(nY):
        for b in range(nB):    
            B_obs[y] = B_obs[y] +(-1)**b * pB[b,y,0]
 
    Bsq_obs = np.zeros(nY, dtype=object)
    for y in range(nY):
        for b in range(nB):    
            Bsq_obs[y] = Bsq_obs[y] + pB[b,y,0]    
    
    D_obs = np.zeros(nY, dtype=object)
    for y in range(nY):
        for b in range(nB):    
            D_obs[y] = D_obs[y] + (-1)**b * pB[b,y,1]
   
    Dsq_obs = np.zeros(nY, dtype=object)
    for y in range(nY):
        for b in range(nB):    
            Dsq_obs[y] = Dsq_obs[y] + pB[b,y,1]
            
    AB_obs = np.zeros((nX,nY), dtype=object)
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                    AB_obs[x,y] = AB_obs[x,y] + (-1)**a * (-1)**b * p[a,b,x,y,0]

    ABsq_obs = np.zeros((nX,nY), dtype=object)
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                    ABsq_obs[x,y] = ABsq_obs[x,y] + (-1)**a  * p[a,b,x,y,0]

    AsqB_obs = np.zeros((nX,nY), dtype=object)
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                    AsqB_obs[x,y] = AsqB_obs[x,y] + (-1)**b * p[a,b,x,y,0]
    
    AsqBsq_obs = np.zeros((nX,nY), dtype=object)
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                    AsqBsq_obs[x,y] = AsqBsq_obs[x,y] + p[a,b,x,y,0]
                    

    AD_obs = np.zeros((nX,nY), dtype=object)
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                    AD_obs[x,y] = AD_obs[x,y] + (-1)**a * (-1)**b * p[a,b,x,y,1]

    ADsq_obs = np.zeros((nX,nY), dtype=object)
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                    ADsq_obs[x,y] = ADsq_obs[x,y] + (-1)**a  * p[a,b,x,y,1]

    AsqD_obs = np.zeros((nX,nY), dtype=object)
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                    AsqD_obs[x,y] = AsqD_obs[x,y] + (-1)**b * p[a,b,x,y,1]
    
    AsqDsq_obs = np.zeros((nX,nY), dtype=object)
    for x in range(nX):
        for y in range(nY):
            for a in range(nA):
                for b in range(nB):
                    AsqDsq_obs[x,y] = AsqDsq_obs[x,y] + p[a,b,x,y,1]

    return A_obs,Asq_obs,B_obs,Bsq_obs,D_obs,Dsq_obs,AB_obs,ABsq_obs,AsqB_obs,AsqBsq_obs,\
        AD_obs,ADsq_obs,AsqD_obs,AsqDsq_obs
"""