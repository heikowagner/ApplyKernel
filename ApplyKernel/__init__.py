from pyspark.mllib.regression import LabeledPoint
import numpy as np

def RadialKernel(x,y,sigma):
    return np.exp(-sum((x-y)**2)/(2*sigma**2))
    
    
def construct_K(Y_X,lamb,kernel, X_1=None):
    sp_X=Y_X.map(lambda x: x.features.toArray()).zipWithIndex()
    if X_1!=None:
        sp_X_1=X_1.zipWithIndex()
    else:
        sp_X_1=sp_X
    sp_Y=Y_X.map(lambda x: x.label).zipWithIndex().map(lambda(x,y) : (y,x) )
    grid=sp_X_1.cartesian(sp_X)
    #grid=sp_X.cartesian(sp_X_1)
    K=grid.map(lambda(x,y) : (x[1],kernel(x[0],y[0],lamb)) )
    return [sp_Y, K]

def construct_labeled(Y,K):
    def add_element(acc,x):
        if type(acc[1]) == list:
            return (min(acc[0],x[0]), acc[1] + [x[1]]  )
        else:
            return (min(acc[0],x[0]), [acc[1]] + [x[1]]  )
    jnd=Y.join(K).reduceByKey(lambda acc, x : add_element(acc,x) )
    labeled=jnd.map(lambda(y,x) : LabeledPoint(x[0], x[1])  )
    order=jnd.map(lambda (y,x): y)
    return [labeled, order]
            
class ApplyKernel:
    def __init__(self, method, kernel, lambd):
        self.method = method
        self.lambd = lambd
        self.trained = None
        self.Y_X_dat= None
        self.kernel=kernel

    def train(self, data, **kwargs):
        data_K= construct_K(data, self.lambd, self.kernel)
        self.Y_X_dat=data
        new_data=construct_labeled(data_K[0],data_K[1])
        self.trained=self.method.train( new_data[0] ,**kwargs)
        return self 
        
    def predict(self, data, **kwargs):
        data_K= construct_K(self.Y_X_dat, self.lambd, self.kernel, data)
        l_dat=construct_labeled(data_K[0], data_K[1])
        pred= self.trained.predict( l_dat[0].map(lambda p:  p.features.toArray()), **kwargs )
        return l_dat[1].zip(pred).sortByKey().map(lambda (x,y): y)
