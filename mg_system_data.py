''' 
    Test problems for Neural Network of Ax=b or Linear Multigrid Solver
    Can either randomize A or x and compute solution vectors b

    12/2 Possibly it makes sense to leave A = [[2,-1],[-1,2]] so that it learns it just
    needs to inverse and multiply by b essentially to compute u
    Leaving A this way is the Laplacian
    If A were to stay the same each time, this certainly presents a difficult since each input is requiring a 
    different output

    Ex: A = [[1 2],[3 4]] x = [[1],[1]] then A*x = [[3],[7]] = b

'''

import numpy 
from math import *

def Laplacian(n, stencil=[-1, 2, -1], periodic=True):
    A = stencil[1] * numpy.eye(n) + stencil[2] * numpy.eye(n, k=1) + stencil[0] * numpy.eye(n, k=-1)
    if periodic:
        A[0,-1] += stencil[0]
        A[-1,0] += stencil[2]
    return A

def Helmholtz(n,k):
    return Laplacian(n)-numpy.identity(n)*k**2

def gen_data(gridsize, n, dim, equation, t): #input is number of training/testing samples desired, matrix size of A is gridsize x gridsize
    dataset = []
    solset = []
    if (equation == "laplacian"):
        A = Laplacian(gridsize)
    elif (equation == "helmholtz"):
        A = Helmholtz(gridsize, k=0.4)
    else:
        raise RuntimeError('Not such equation {}, choose either laplacian or helmoltz'.format(equation))
    #A_pinv = numpy.linalg.pinv(A)
    for _ in range(n):
        #k = random.randint(0,1)
       # k = 0 #for now 12/2
        #if (k == 0): #let A be [[2,-1,0],[-1,2,-1],...,[0,-1,2]]
            #will need an efficient way to do this fast
            #for now will let gridsize be 2x2 for getting started
            #u = numpy.random.random((gridsize,1))
           # u = numpy.random.uniform(low=-10, high=10, size=(gridsize,1)) #does range matter? possibily the larger the range, the easier for network?
        numpy.random.seed(_+t)
        u = numpy.random.rand(gridsize)
        u = u - numpy.mean(u)   
       # u_test = [1]*gridsize #FOR TEST OF AI=A FEB12 707PM
        #u = u_test      
        #print ('u = ', u)
#        if (equation == "helmholtz"):
##            c = [0]*(n-2)
##            a = [-1,1]
##            b = numpy.append(a, c)
#            b = [-1,1,0,0,0,0,0,0]
#        else:
        b = A @ u
        #print ('b = ', b)
        dataset.append(b)
        solset.append(u)
        #print (dataset)
    dataset = numpy.reshape(numpy.array(dataset),[n,1,gridsize,1])
    solset = numpy.reshape(numpy.array(solset),[n,1,gridsize,1])
        # if (k == 1): #randomly generate tridiagonal matrix? maybe should leave as 2,-1, so on and just random x vector
        #     continue
    return dataset, solset# A_pinv #returns input matrix of [A b] and solution array u]

def AI_data(gridsize,n,dim):
    dataset = []
    solset = []
    A = numpy.array(Laplacian(gridsize))
    I = numpy.array(numpy.identity(gridsize))
    print (I[4])
    for _ in range(n):
        r = numpy.random.randint(0,gridsize)
        b = A[r]
        u = I[r]
        dataset.append(b)
        solset.append(u)
    #print ('Aprime = ', A)
    A = numpy.reshape(numpy.array(dataset),[n,1,gridsize,1])
    I = numpy.reshape(numpy.array(solset),[n,1,gridsize,1])
    return A, I

#print (gen_data(8,1,1,"helmholtz",8)[0]) #shape is (n, gridsize, 1)
#print (AI_data(6,3,1))
