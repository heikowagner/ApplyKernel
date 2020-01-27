# ApplyKernel
A Phyton (PySpark) Package to extend linear mllib regression models to process non-linear data using Kernels.

Install the package via:
```
pip install git+https://github.com/heikowagner/ApplyKernel.git
```

# Usage
```python
import ApplyKernel

import numpy as np
import matplotlib.pyplot as plt

##Generate data
#Simulation
N=150

Y=np.random.randint(0,2,N)
X=np.array([np.random.normal(0,1,N)+Y*np.random.normal(3,1,N), np.random.normal(0,1,N)+Y*np.random.normal(3,1,N) ])

#plot data
plt.scatter(X[0], X[1], c=Y)
plt.show()
X_par=sc.parallelize(np.transpose(X)).zipWithIndex().map(lambda(x,y) : (y,x) )
Y_par=sc.parallelize(np.transpose(Y)).zipWithIndex().map(lambda(x,y) : (y,x) )
Y_X= Y_par.join(X_par).map(lambda(y,x) : LabeledPoint(x[0], x[1])  )
#print(Y_X.collect())

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.regression import LinearRegressionModel, LinearRegressionWithSGD
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.classification import SVMModel, SVMWithSGD

print( ApplyKernel(LinearRegressionWithSGD, RadialKernel, 0.5).train(Y_X).predict(Y_X.map(lambda x: x.features.toArray())).collect() )
print( ApplyKernel(SVMWithSGD, RadialKernel, 0.5).train(Y_X).predict(Y_X.map(lambda x: x.features.toArray())).collect() )
```
