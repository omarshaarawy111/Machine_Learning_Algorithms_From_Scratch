import numpy as np
# Linear Regression class with (OLS, Normal Equation and Gradient Descent)
class LinearRegression():

    # Initialization
    def __init__(self, method = 'OLS', learning_rate = 0.1, n_iterations = 1000, alpha=1.0):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.theta = None

    # Fit 
    def fit(self, X, y):
        # Get number of rows and columns of training data
        m, n=X.shape

        # Add bias => Xo = 1 to training data
        X_b = np.c_[np.ones((m,1)), X]

        # Get theta by each method (calculate paramerts to apply it within function in predict)
        if self.method == 'OLS':
            self.theta = np.linalg.solve(X_b.T @ X_b, X_b.T @ y)

        elif self.method == 'Normal':
            self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

        elif self.method == 'Gradient Descent':
            # It is n+1 because we have theta 0 
            self.theta = np.random.randn(n + 1)   
            for iteration in range(self.n_iterations):
                h_theta = X_b @ self.theta
                gradients = 2/m * X_b.T @ (h_theta - y)
                self.theta -= self.learning_rate * gradients

        elif self.method == 'Ridge':
            # I matrix 
            I = np.eye(n+1)
            # No regularization of bias
            I[0,0] = 0  
            self.theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y

        elif self.method == 'Ridge-Gradient Descent':
            # It is n+1 because we have theta 0 
            self.theta = np.random.randn(n + 1)   
            for iteration in range(self.n_iterations):
                h_theta = X_b @ self.theta
                # Add regularization term and make sure not included theta 0 
                gradients = 2/m * ( X_b.T @ (h_theta - y) + self.alpha * np.r_[0, self.theta[1:]] )
                self.theta -= self.learning_rate * gradients  

        elif self.method == 'Lasso-Gradient Descent':
            # It is n+1 because we have theta 0 
            self.theta = np.random.randn(n+1)
            for _ in range(self.n_iterations):
                h_theta = X_b @ self.theta
                gradients = 2/m * X_b.T @ (h_theta - y)
                # Add regularization term nd make sure not included theta 0 and never update bias dervative
                gradients[1:] += self.alpha * np.sign(self.theta[1:])
                self.theta -= self.learning_rate * gradients        

        else :
            raise ValueError('Unknown method.')
        
        # Bias value and other theta values 
        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]
        
    # Predict
    def predict(self, X):
        # Get number of rows of test data
        m = X.shape[0]

        # Add bias => Xo = 1 to test data
        X_b = np.c_[np.ones((m,1)), X]

        # Apply linear regression equation
        y_predict = X_b @ self.theta
        return y_predict
    
    # Score 
    def score(self, X_new, y):
        # Return R^2 score
        y_pred = self.predict(X_new)
        ss_total = np.sum((y - np.mean(y)) **2)
        ss_residual = np.sum((y - y_pred) **2)
        return 1 - ss_residual / ss_total