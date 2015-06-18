def gradient_scalarf(w, x, t):
	return 2 * x * (nn(x, w) - t)

def gradient_vectorf(w, x, t): 
	return (nn(x, w) - t).T * x
	
def updatef(w, x, t, learning_rate):
    return learning_rate * gradient_scalarf(w, x, t).sum()

def update_vectorf(w, x, t, learning_rate):
    return learning_rate * gradient_vectorf(w, x, t)