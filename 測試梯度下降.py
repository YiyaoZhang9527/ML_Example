import numpy as np

class linear:

    def __init__(self,srcX,srcY
                 ,alpha=1
                 ,maxloop=5000
                 ,epsilon = 0):
        self.srcX = srcX
        self.srcY = srcY
        self.alpha = alpha
        self.maxloop = maxloop
        self.epsilon = epsilon

    def standarize(self,X):
        m, n = X.shape
        values = {}
        for j in range(n):
            features = X[:,j]
            meanVal = features.mean(axis=0)
            stdVal = features.std(axis=0)
            values[j] = [meanVal, stdVal]
            if stdVal != 0:
                X[:,j] = (features - meanVal) / stdVal
            else:
                X[:,j] = 0
        return X, values

    def data_processing(self):
        normalizedX ,normalizedValue = self.standarize(self.srcX)
        x = np.c_[np.ones(self.srcX.shape[0]),normalizedX]
        return x,normalizedValue,self.alpha,self.epsilon,self.maxloop,self.srcY

    def h(self,x,w):
        return np.dot(x,w)

    def loss(self,x,y,w):
        m = x.shape[0]
        return ((self.h(x,w)-y)**2).sum()/(2*m)

    def bgd_fit(self):
        x,Value,alpha,epsilon,maxloop,y = self.data_processing()
        m,n =  x.shape
        theta = np.zeros((n,1))
        costs = [self.loss(x,y,theta)]
        losslog = [np.inf]
        thetas = []

        for i in range(maxloop):
            theta = theta - alpha * 1.0 / m * np.dot(x.T,(self.h(x,theta)-y))
            cost = self.loss(x,y,theta)
            costs.append(cost)
            lossvar = abs(costs[-2]-costs[-1])
            losslog.append(lossvar)
            thetas.append(theta)

            if lossvar <= epsilon or np.isinf(lossvar) or np.isnan(lossvar):
                reduction_standardization = np.array([Value[i] for i in Value])
                prediction=np.dot(np.r_[np.ones(1),(np.array([70,2])-reduction_standardization[:,0])/reduction_standardization[:,-1]],theta)
                return prediction,theta,reduction_standardization,costs,thetas,Value,losslog,i

    def __del__(self):
        print('Linear regression is complete.')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    X = np.array([[62.47, 2.0], [65.78, 3.0], [58.05, 2.0], [52.09, 2.0], [74.98, 3.0], [55.87, 2.0], [90.66, 3.0], [113.68, 3.0], [97.92, 2.0], [46.33, 2.0], [134.55, 3.0], [151.15, 3.0], [63.01, 2.0], [65.66, 2.0], [108.81, 3.0], [66.19, 3.0], [54.1, 2.0], [73.44, 2.0], [51.78, 2.0], [92.42, 3.0], [59.13, 2.0], [49.49, 1.0], [51.68, 2.0], [52.87, 2.0], [69.46, 2.0], [76.41, 2.0], [63.1, 2.0], [197.37, 5.0], [93.53, 3.0], [91.35, 3.0], [103.49, 3.0], [45.12, 2.0], [59.59, 2.0], [174.66, 4.0], [35.8, 1.0], [91.35, 3.0], [55.07, 2.0], [119.44, 3.0], [65.85, 2.0], [72.05, 3.0], [85.98, 3.0], [103.29, 4.0], [184.05, 5.0], [90.87, 3.0], [38.83, 1.0], [51.65, 1.0], [50.14, 1.0]])
    Y = np.array([[213.0], [226.0], [179.0], [188.0], [215.0], [152.0], [290.0], [375.0], [305.0], [166.0], [385.0], [500.0], [195.0], [200.0], [310.0], [205.0], [158.0], [270.0], [150.0], [310.0], [180.0], [200.0], [155.0], [178.0], [303.0], [250.0], [218.0], [630.0], [326.0], [310.0], [530.0], [138.0], [230.0], [560.0], [115.0], [400.0], [140.0], [547.0], [240.0], [250.0], [315.0], [330.0], [680.0], [302.0], [130.0], [162.0], [140.0]])
    linearmodule = linear(srcX=X,srcY=Y,alpha=1,maxloop=5000,epsilon = 0)
    prediction,theta,reduction_standardization,costs,thetas,Value,losslog,time = linearmodule.bgd_fit()
    print(prediction,time)
    plt.plot(costs)
    plt.show()
    plt.close()

