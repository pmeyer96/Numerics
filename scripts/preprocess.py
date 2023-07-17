import sympy as sp
import numpy as np
from scipy.linalg import null_space
import time

import sys


def preprocessing_neuron(A_0,z_0,c):
    m = A_0.shape[0]
    #Step 1
    J_0 = np.where(~A_0.any(axis= 0))[0]
    #Step 2
    b = np.zeros(z_0.shape)
    for i in J_0:
        b[i] = c - z_0[i]
    #Step 3
    z_it =z_0+b
    #Step 4
    k = 0
    c_vec = np.ones(z_0.shape) * c

    #print("Start", np.matmul(A_0, z_it))
    #print(c_vec)
    #Step 5
    while(np.count_nonzero(np.abs(z_it) -c_vec) > m):
    #calculate first m+1 indices of A
        J = np.where(np.abs(z_it) != c)[0]
        
        J_rel = J[:m+1]
        A_k = np.asanyarray(A_0[:,J_rel])
        
        b = null_space(A_k)[:,0]
        #print("Nullvector",b)
        alpha,_ = compute_alpha(z_it,b,J_rel,c)
        #print(alpha)
        # print("alpha times b", alpha*b)
        # print("z_rel",z_0[J_rel])
        j = 0
        for i in J_rel:
            z_it[i] = z_it[i] + alpha* b[j]
            j+=1     
        k+=1
        # print(k)
        # if np.any(np.abs(z_it) > c):
        #     print(np.any(np.abs(z_it)))
        #     print("Hier l äuft was falsch")
        #     print("alpha", alpha)
        #     print("z_it", z_it)
        #     print("b: ", b)
        #     print(k)
        #     return
        # print("Nullspace vector times A_"+str(k), np.matmul(A_k, b))
        # print("After "+str(k)+" Steps: ", np.matmul(A_0,z_it))
        # print("z_" + str(k), z_it)
    return z_it, k

def compute_alpha(z,b,J_rel, c):
    alpha_i = []
    j = 0
    
    for i in J_rel:
        if b[j] !=0 and -c-z[i] != 0 and c-z[i]!= 0:
            alpha_1 = (-c-z[i])/b[j]
            alpha_2 = (c-z[i])/b[j]
            alpha_i.append(min([alpha_1,alpha_2], key = abs))
            j+=1
        else:
            alpha_i.append(sys.float_info.max)
            j+=1

    alpha = min(alpha_i, key = abs)
    if alpha == sys.float_info.max:
        print("Alphas to chose from", alpha_i)
        print("b ist",b)
        print("z ist", z)
        print("shape of b", b.shape)
        print(len(J_rel))
    if alpha == 0:
        print("Alpha is zero")
        print(alpha_i)
    
    idx = alpha_i.index(alpha)
    return alpha, idx

def preprocessing_Neuron_p(A_0,z_0,c):
    print("c ist", c)
    m = A_0.shape[0]
    J_0 = np.where(~A_0.any(axis= 0))[0]
    #Step 2
    b = np.zeros(z_0.shape)
    for i in J_0:
        b[i] = c - z_0[i]
    #Step 3
    z_it =z_0+b
    #Step 4
    k = 0
    c_vec = np.ones(z_0.shape) * c
    d1 = np.count_nonzero(np.abs(z_it) -c_vec)
    sum = 0
    while(np.count_nonzero(np.abs(z_it) -c_vec) > m):
        k+=1
        J = np.where(np.abs(z_it) != c)[0]
        if(len(J) >= 2*m):
            J_rel = J[:2*m]
        else:
            J_rel = J

        A_k = np.asanyarray(A_0[:,J_rel])

        B = null_space(A_k)

        alpha, idx = compute_alpha(z_it, B[:,0], J_rel, c)
        z_it = add_z_alpha_b(z_it,alpha,B[:,0], J_rel)
        z_it[np.abs(z_it) < 1e-17] = 0
        counter = 0
        for i in range(len(J_rel)-m-1):
            B = reduce_basis(B,idx)
            alpha,idx = compute_alpha(z_it,B[:,0], J_rel,c)
            if np.all(B[:,0] == 0 ):
                print("Error!!")
            z_it_prime = z_it
            z_it = add_z_alpha_b(z_it, alpha, B[:,0], J_rel)
            counter+= 1

        d2 = np.count_nonzero(np.abs(z_it) -c_vec)
        print("k", k)
        print("Changed to c", d1- d2, " counter", counter)
        sum = sum + d1-d2
        d1 = d2
    return z_it, k, sum
        
        

def reduce_basis(B, idx):
    if B[idx,1] != 0:
        B_prime = np.reshape(B[:,0] - B[idx,0]/B[idx,1] * B[:,1], (B.shape[0],1))
    else:
        B_prime = np.reshape(B[:,1], (B.shape[0],1))
    for i in range(2,B.shape[1]):
        if B[idx,i] != 0:
            b_i = np.reshape((B[:,0] - B[idx,0]/B[idx,i] * B[:,i]),(B.shape[0],1))

        else:
            b_i = np.reshape(B[:,i],(B.shape[0],1))

        B_prime = np.concatenate((B_prime,b_i), axis = 1)
    
    B_prime[np.abs(B_prime) < 1e-15] = 0
    
    return B_prime

def add_z_alpha_b(z,alpha,b, J_rel):
    j = 0
    for i in J_rel:
        z[i] = z[i] + alpha* b[j]
        j+=1
    return z




A = np.random.rand(300,12000)
z = np.random.rand(12000)
c = np.abs(max(z, key = abs))
start_time = time.time()
z_it, k = preprocessing_neuron(A,z,c)
print("Algorithm orig needs seconds", (time.time()-start_time))
if np.any(np.abs(z_it) > c):
    print("algoirthm failed")
print("Iterations", k)
print(np.matmul(A,z)-np.matmul(A,z_it))




# A = np.random.rand(300,12000)
# z = np.random.rand(12000)
# c = np.abs(max(z, key = abs))
# c_vec = np.ones(z.shape) * c
# start_time = time.time()
# z_it,k, s = preprocessing_Neuron_p(A,z,c)
# print("Algorithm changed needs seconds", (time.time()-start_time))
# print("Zu c geänderte", s)
# print("Iterationen", k)

# print("Non max entries: ",np.count_nonzero(np.abs(z_it) -c_vec))
# print(np.matmul(A,z)-np.matmul(A,z_it))
