'''
Created on Apr 1, 2021

@author: alienware
'''
import numpy as np
from numpy import linalg

def Get_j(M):
    j = (np.argmin(M[-1,:]) if min(M[-1,:])<0 else None )
    return j
#Get i 
def bound_div(v1,v2):
    Out = []
    for i in range(len(v1)):
        for j in range(len(v2)):
            if i == j:
                if v2[j] != 0 :
                    out = (v1[i]/v2[j] if v1[i]/v2[j]>0 else np.inf)
                    if v1[i]/v2[j]>0:
                else:
                    out = np.inf
                Out.append(out)
    return Out 
                
def Get_i(M):
    j = Get_j(M)
    i = (np.argmin(bound_div(M[0:-1,-1],M[0:-1,j]))if j is not None else None)
    return i 
#Pivot 
def Pivot(M:np.array,point = (np.inf,np.inf))-> np.array:
    i,j = ( (point[0],point[1]) if point != (np.inf,np.inf) else (Get_i(M),Get_j(M)))
    M[i,:] *= 1/M[i,j]
    for row in range(M.shape[0]):
        if row != i and M[row,j]!=0:
            M[row,:] += -M[row,j]/M[i,j] * M[i,:]
    return M 

def GET(v, tol = 1e-5):
    if abs(linalg.norm(v) - 1 ) < tol:
        return np.argmin(abs(np.array(v)-1))
    
def Result(O):
    temp = []
    OUT = np.zeros(O.shape[1])
    for col in range(O.shape[1]):
        temp.append(GET(O[:,col]))
    for idx in range(len(temp)):
            OUT[idx] = (O[:,-1][temp[idx]] if temp[idx] is not None else 0) 
    return OUT

def simplex(A,b,c):
    """
    The Simplex Method of an standard linear programming system
    @A: The constraint matrix the optimization problem is subject to 
    @b: The constants of the constriant 
    @c: The Objective vector 
    """
    M = np.zeros(np.array(A.shape)+1)
    M[0:A.shape[0],0:A.shape[1]] = A
    M[0:A.shape[0],A.shape[1]] = b
    M[A.shape[0],0:A.shape[1]] = c
    i,j =  (Get_i(M),Get_j(M))
    while j != None:
        M = Pivot(M, (i,j))
        i,j =  (Get_i(M),Get_j(M))
    argmins = Result(M)
    min_idx = np.array(M.shape)-1
    minimum = -1*M[min_idx[0],min_idx[1]]
    return argmins,minimum

if __name__ == "__main__":
    A = np.array([[30,1,1,0,0],[-30,1,0,1,0],[-30,1,0,0,1]])
    b = np.array([60,10,30])
    c = np.array([0,-1,0,0,0])
    #A = np.array([[30,40,60,-1,1,0],[60,10,30,-1,0,1],[-1,-1,-1,0,0,0],[1,1,1,0,0,0]])
    #b = np.array([0,0,-1,1])
    #c = np.array([0,0,0,-1,0,0])
    for i in simplex(A, b, c):print(i)
    print("DONE")
    
    