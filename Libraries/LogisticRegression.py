import numpy as np
# Logistic Regression class
class LogisticRegression():

    # Initialization
    def __init__(self,method = 'Gradient Descent', learning_rate = 0.1, n_iterations = 1000, alpha=1.0):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.theta = None

    # Sigmoid function
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Fit 
    def fit(self, X, y):
        # Get number of rows and columns of training data
        m, n=X.shape

        # Add bias => Xo = 1 to training data
        X_b = np.c_[np.ones((m,1)), X]

        # Get theta by gradient descent (calculate paramerts to apply it within function in predict)
        if self.method == 'Gradient Descent':
            # It is n+1 because we have theta 0 
            self.theta = np.random.randn(n + 1)   
            for iteration in range(self.n_iterations):
                # Calculate z score then hypothesis
                z = X_b @ self.theta
                h_theta = self._sigmoid(z)
                gradients = 1/m * X_b.T @ (h_theta - y)
                self.theta -= self.learning_rate * gradients

        elif self.method == 'Ridge':
            # It is n+1 because we have theta 0 
            self.theta = np.random.randn(n + 1)   
            for iteration in range(self.n_iterations):
                 # Calculate z score then hypothesis
                z = X_b @ self.theta
                h_theta = self._sigmoid(z)
                gradients = 1/m * X_b.T @ (h_theta - y)
                # Add regularization term and make sure not included theta 0 
                gradients[1:] += (self.alpha / m) * self.theta[1:]
                self.theta -= self.learning_rate * gradients  

        elif self.method == 'Lasso':
            # It is n+1 because we have theta 0 
            self.theta = np.random.randn(n+1)
            for _ in range(self.n_iterations):
                h_theta = X_b @ self.theta
                gradients = 1/m * X_b.T @ (h_theta - y)
                # Add regularization term and make sure not included theta 0 and never update bias dervative
                gradients[1:] += (self.alpha / m) * np.sign(self.theta[1:])
                self.theta -= self.learning_rate * gradients        

        else :
            raise ValueError('Unknown method.')
        
        # Bias value and other theta values 
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

    # Predict probabilities
    # This function is ready to use in soft oting later
    def predict_proba(self, X):
        # Get number of rows of test data
        m = X.shape[0]
        # Add bias => Xo = 1 to test data
        X_b = np.c_[np.ones((m,1)), X]
        # Apply linear regression equation to get z score
        z = X_b @ self.theta
        # Apply sigmoid function to get probabilities
        y_proba = self._sigmoid(z)
        return y_proba
       
    # Predict classes
    def predict(self, X):
        # get class labels based on probability threshold of 0.5 to be y = 1 or y = 0
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    # Score 
    def score(self, X_new, y):
        # Return accuracy score
        y_pred = self.predict(X_new)
        return np.mean(y_pred == y)