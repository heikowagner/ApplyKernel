# ApplyKernel
A Python (PySpark) Package to extend linear mllib regression models to process non-linear data using Kernels.

Install the package via:
```
pip install git+https://github.com/heikowagner/ApplyKernel.git
```

# Usage
To train a model:
`TrainedModel= ApplyKernel(<Model>, <Kernel>, <Bandwith>).train(<LabeledPointVector>)`.

For prediction:
`TrainedModel.predict(<FeatureVector>)`

##Example:
```python
from ApplyKernel import ApplyKernel, RadialKernel

import numpy as np
import matplotlib.pyplot as plt

##Generate data
#Simulation
N=500
Y= np.random.randint(0,2,N)
degree=np.random.normal(0,1,N)*2*np.pi
X= [0+ (0.5 + Y*0.5)* np.cos(degree)+ np.random.normal(0,2,N)*0.05, 0 + (0.5 + Y*0.5)*np.sin(degree)+ np.random.normal(0,2,N)*0.05   ]

#plot data
plt.scatter(X[0], X[1], c=Y)
plt.show()

#Create LabeledPoint Vector
from pyspark.mllib.regression import LabeledPoint
X_par=sc.parallelize(np.transpose(X)).zipWithIndex().map(lambda(x,y) : (y,x) )
Y_par=sc.parallelize(np.transpose(Y)).zipWithIndex().map(lambda(x,y) : (y,x) )
Y_X= Y_par.join(X_par).map(lambda(y,x) : LabeledPoint(x[0], x[1])  )

from pyspark.mllib.regression import LinearRegressionModel, LinearRegressionWithSGD
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.classification import SVMModel, SVMWithSGD

##Train the Models
#KernelRegression= ApplyKernel(LinearRegressionWithSGD, RadialKernel, 0.5).train(Y_X)
#KernelLogit= ApplyKernel(LogisticRegressionWithLBFGS, RadialKernel, 0.5).train(Y_X)

KernelSVM= ApplyKernel(SVMWithSGD, RadialKernel, 0.5).train(Y_X)

##Simulate Test Set
N=200

Y_test= np.array( np.random.randint(0,2,N) )
degree=np.random.normal(0,1,N)*2*np.pi
X_test= np.array( [0+ (0.5 + Y_test*0.5)* np.cos(degree)+ np.random.normal(0,2,N)*0.05, 0 + (0.5 + Y_test*0.5)*np.sin(degree)+ np.random.normal(0,2,N)*0.05   ])
X_par= sc.parallelize( X_test.transpose() )

##Predict Group
Preds = KernelSVM.predict(X_par)

##Evaluate Model
sc_Y=sc.parallelize( Y_test ).zipWithIndex().map(lambda (x,y): (y,x))
labelsAndPreds=Preds.zipWithIndex().map(lambda (x,y): (y,x)).join( sc_Y ).map(lambda (x,y): y)

testErr = labelsAndPreds.filter(lambda (x,y): y != x).count() / float(labelsAndPreds.count())
print("Training Error = " + str(testErr))
plt.scatter(X_test[0], X_test[1], c=Preds.collect() )
plt.show()

```
