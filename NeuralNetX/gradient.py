def gradientf(w, x, t):
	return 2 * x * (nn(x, w) - t)

def updatef(w, x, t, learning_rate):
    return learning_rate * gradientf(w, x, t).sum()
