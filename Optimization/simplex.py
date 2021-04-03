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

def Idx_Pvt(v, tol = 1e-5):
    "Takes in a column, return None if it's not a unit vector, else Return the index of the pivot "
    if abs(linalg.norm(v) - 1 ) < tol:
        return np.argmin(abs(np.array(v)-1))
    
def simplex_split(A,c,O):
    """
    Input:
    @A: Original Matrix 
    @O: Output Matrix after simplex tableau operation
    OUTPUT:
    B: Basic Vectors of A
    D: NonBasic Vectors of A
    c_d: NonBasic Cost of A
    r_d: NonBasic Cost of O 
    """
    idx_vector = []
    for col in range(O.shape[1]): #Iterate through all columns
        idx_vector.append(Idx_Pvt(O[:,col])) #Determine the index of each column's pivots
    idx_vector.pop(-1)
    d_idx = (1* (np.array(idx_vector) == None))
    D,B = A.compress(d_idx,axis=1),A.compress(1* (np.array(idx_vector) != None),axis=1)
    c_d = c.compress(d_idx)
    c_b = c.compress(1* (np.array(idx_vector) != None))
    r_d = O[O.shape[0]-1,0:-1].compress(d_idx)
    return B,D,c_d,c_b,r_d
    
def Result(O):
    temp = []
    OUT = np.zeros(O.shape[1])
    for col in range(O.shape[1]):
        temp.append(Idx_Pvt(O[:,col]))
    for idx in range(len(temp)):
            OUT[idx] = (O[:,-1][temp[idx]] if temp[idx] is not None else 0) 
    return OUT
    
def simplex(A,b,c,dual:bool = False, results_show: bool = True, 
            trace:bool = False, trace_precision = 2):
    """
    The Simplex Method of an standard linear programming system
    @A: The constraint matrix the optimization problem is subject to 
    @b: The constants of the constriant 
    @c: The Objective vector 
    @dual: calculating the maximizer for the Dual Problem
    """
    M = np.zeros(np.array(A.shape)+1)
    M[0:A.shape[0],0:A.shape[1]] = A
    M[0:A.shape[0],A.shape[1]] = b
    M[A.shape[0],0:A.shape[1]] = c
    i,j =  (Get_i(M),Get_j(M))
    while j != None:
        temp = M
        M = Pivot(M, (i,j))
        if trace:
            print(temp.round(trace_precision))
            print("-->"*10+f"Pivoting on {i,j}"+"-->"*10)
            print(M.round(trace_precision))
            print("~"*60)
        i,j =  (Get_i(M),Get_j(M))
        
    argmins = Result(M)
    min_idx = np.array(M.shape)-1
    minimum = -1*M[min_idx[0],min_idx[1]]
    if dual:
        B,D,c_d,c_b,r_d = simplex_split(A,c,M)
        try:
            argmins_dual = np.linalg.solve(a = D.transpose(),b = np.array(c_d-r_d).transpose())
        except np.linalg.LinAlgError:
            try:
                argmins_dual = [c_d - r_d] @ np.linalg.inv(D)
            except:
                argmins_dual = c_b.transpose()@np.linalg.inv(B)
        if results_show: 
            print(f"""
The minimum of this optimization problem is achieved at ->>>{argmins}<<<-; 
Its dual argmins are ->>>{-1*argmins_dual}<<<-; 
The optimized value of the objective function is: ->>>{minimum}<<<-.
                   """)
        return argmins,argmins_dual,minimum
    else:
        return argmins,minimum
                
if __name__ == "__main__":
    A = np.array([[30,1,1,0,0],[-30,1,0,1,0],[-30,1,0,0,1]])
    b = np.array([60,10,30])
    c = np.array([0,-1,0,0,0])
    #A = np.array([[30,40,60,-1,1,0],[60,10,30,-1,0,1],[-1,-1,-1,0,0,0],[1,1,1,0,0,0]])
    #b = np.array([0,0,-1,1])
    #c = np.array([0,0,0,-1,0,0])
    #A = np.array([2,-1,7,1,0,0,1,3,4,0,1,0,3,6,1,0,0,1]).reshape(3,6)
    #b = np.array([0,9,3])
    #c = np.array([-2,-5,-1,0,0,0])
    for i in simplex(A, b, c,True,trace=True):print(i)
    print("DONE")
    
    