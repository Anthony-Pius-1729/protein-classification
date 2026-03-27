import numpy as np

class ProteinClassifier:
    def __init__(self, x_train, y_train, x_test, y_test, x_valid, y_valid):
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._x_valid = x_valid
        self._y_valid = y_valid
        
        ## Parameters
        self._w = np.zeros(self._x_train.shape[1])
        self._B = 0.01
        self._alpha = 0.001
        
    def sigmoid(self, z):
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    
    def train_cross_entropy (self):
        """
        Train the model using a cross entropy loss function
        """
    
        
        iterations = 10000
        loss = []
        
        
        for it in range(iterations):
            N = len(self._x_train)
        
            Z = self._x_train @ self._w + self._B
            
            y_hat = self.sigmoid(Z)
            y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
            pure_error = y_hat - self._y_train
            dw = np.dot(self._x_train.T, pure_error)/N
            db = np.mean(pure_error)
            
            ith_loss = -np.mean(self._y_train * np.log(y_hat) + (1 - self._y_train)*np.log(1 - y_hat))
            loss.append(ith_loss)
            
            self._w = self._w - (self._alpha) * dw
            self._B = self._B - (self._alpha) * db
            
            
        
        print(f"Weights: {self._w}")
        
        