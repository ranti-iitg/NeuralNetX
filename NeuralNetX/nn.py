from logistic import logistic
import numpy as np

def nn_scalarf(x, w): 
	return x * w

def nn_vectorf(x, w): 
	return logistic(x.dot(w.T))
	
def predictf(x,w): 
	return np.around(nn(x,w))