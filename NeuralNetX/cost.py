
def costL2f(y, t): 
	return ((t - y)**2).sum()

def cost_cross_entropy(y, t):
    return - np.sum(np.multiply(t, np.log(y)) + np.multiply((1-t), np.log(1-y)))
