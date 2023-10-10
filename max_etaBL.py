# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:10:46 2023
@author: Edwin
"""
"""
This code runs the NPA implementations for scenarios considered in the manuscript.

NOTE: This code requires the latest version of the package ncpol2sdpa that is being 
maintained by Peter Brown and is available on github.

Details about the implementation can be found in the Appendix of the manuscript.
"""
def get_momentequalities(A,B,D,etaA,etaBS,etaBL,chsh_or_zx):
    ### Get the moment equalities ###
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
    """
    Define the objective function. Since ncpol2sdpa minimizes the objective by default, we use -etaBL
    """
    return -etaBL


def get_subs(A,B,D):
    ### Substitutions used in the moment matrix ##
    
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
chsh_or_zx = 'chsh' ##This fixes the ideal implementation to be used. For CHSH implementation, set this variable equal to the string 'chsh'. For the BB84 implementation, set this variable equal to the string 'zx'.
binning = False ## Choose whether the outcomes are deterministically binned to +1 or if they are kept as seperate outcomes
visibility = 0.94

if chsh_or_zx == 'chsh':
    if binning:
        etaA_list = np.linspace(0.96,1,20)
    else:
        etaA_list = np.linspace(0.9,0.92,3).tolist() + np.linspace(0.92,0.94,4)[1:].tolist() + np.linspace(0.94,0.96,5)[1:].tolist()  + np.linspace(0.96,0.98,7)[1:].tolist() + \
            # np.linspace(0.98,0.99,7)[1:].tolist() + np.linspace(0.99,1,8)[1:].tolist()

if chsh_or_zx == 'zx':
    if binning:
        etaA_list = [1]#np.linspace(0.9586,1,10)
    else:
        etaA_list = [1] #np.linspace(0.8942,0.8955,5).tolist() + np.linspace(0.8955,1,10)[1:].tolist() 

### Define the observables used in the moment matrix ###

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


### Construct the operators to be used in the moment matrix ### 
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
sdp = ncp.SdpRelaxation(ncp.flatten([level4]), normalized=True, verbose=False) ## Set verbose = True to display the solver output. 
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
    """
    If you are using Peter Wittek originial ncpol2sdpa package, and not the latest version by Peter Brown, you will have to first convert the problem 
    to cvxpy and then solve using cvxpy. 
    """
    # prob = sdp.convert_to_cvxpy() 
    # prob.solve(verbose = False)
    sdp.solve()
    etaBL_list += [-sdp.primal]
    # etaBL_list += [-prob.value]
    print(etaA, -sdp.primal)
    # print(etaA, -prob.value)


plt.plot(etaA_list,etaBL_list,'r-' , linewidth=2)
plt.grid(color='r', linestyle='-', linewidth=1)
