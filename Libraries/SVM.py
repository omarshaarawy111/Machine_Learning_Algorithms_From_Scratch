import numpy as np
# SVM class for Classification and Regression
# Kernal functions for SVM 
# It will be used for classification and regression 
# Linear Kernel
def linear_kernel(x1, x2):
    return np.dot(x1, x2)   

# Polynomial Kernel
def polynomial_kernel(x1, x2 , gamma=1, degree=3, coef0=1):
    return ((gamma * np.dot(x1, x2)) + coef0) ** degree

# Radial Basis Function (RBF) Kernel 
def rbf_kernel(x1, x2, gamma=1):
    distance = np.linalg.norm(x1 - x2) ** 2
    # Know that gamm = 1 / (2 * sigma^2)      
    return np.exp(-gamma * distance)

# Sigmoid Kernel
def sigmoid_kernel(x1, x2, gamma=0.01, coef0=0):
    return np.tanh((gamma * np.dot(x1, x2)) + coef0)

# Classification and Regression using SVM
# Classification class
class SVC():

    # Initialization
    # Tol is threshold for stopping criteria 
    def __init__(self, c = 1.0, kernel = 'Linear', degree=3, gamma=1, coef0=1, tol=1e-3, max_iter=1000, decision_function_shape='ovr'):
        self.c = c
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.models = [] # To store sub-models for multiclass
        self.classes = None
        
        # For each case of kernal is assigned by the retun value of the function to the varaibale kernal and assign the all parameters
        # We make varaible as function 
        if kernel == 'Linear':
            self.kernel = lambda x1, x2: linear_kernel(x1, x2)

        elif kernel == 'Polynomial':
            self.kernel = lambda x1, x2: polynomial_kernel(x1, x2, gamma=self.gamma, degree=self.degree, coef0=self.coef0)    

        elif kernel == 'RBF':
            self.kernel = lambda x1, x2: rbf_kernel(x1, x2, gamma=self.gamma)

        elif kernel == 'Sigmoid':
            self.kernel = lambda x1, x2: sigmoid_kernel(x1, x2, gamma=self.gamma, coef0=self.coef0)

        else:
            raise ValueError("Unknown kernel.")    

    # Internal helper for binary training (SMO)
    def _fit_binary(self, X, y):
        m, n = X.shape
        theta = np.zeros(m)
        b = 0

        # Compute the Kernel matrix
        K_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K_matrix[i, j] = self.kernel(X[i], X[j])   

        # SMO Algorithm
        for iteration in range(self.max_iter):      
            alpha_prev = np.copy(theta)

            for i in range(m):
                f_xi = np.sum(theta * y * K_matrix[:, i]) + b
                E_i = f_xi - y[i]

                if (y[i] * E_i < -self.tol and theta[i] < self.c) or (y[i] * E_i > self.tol and theta[i] > 0):
                    j = np.random.randint(0, m)
                    while j == i:
                        j = np.random.randint(0, m)

                    f_xj = np.sum(theta * y * K_matrix[:, j]) + b
                    E_j = f_xj - y[j]

                    alpha_i_old, alpha_j_old = theta[i], theta[j]

                    if y[i] != y[j]:
                        L, H = max(0, alpha_j_old - alpha_i_old), min(self.c, self.c + alpha_j_old - alpha_i_old)
                    else:
                        L, H = max(0, alpha_i_old + alpha_j_old - self.c), min(self.c, alpha_i_old + alpha_j_old)

                    if L == H: continue

                    eta = 2.0 * K_matrix[i, j] - K_matrix[i, i] - K_matrix[j, j]
                    if eta >= 0: continue

                    theta[j] -= (y[j] * (E_i - E_j)) / eta
                    theta[j] = np.clip(theta[j], L, H)

                    if abs(theta[j] - alpha_j_old) < 1e-5: continue

                    theta[i] += y[i] * y[j] * (alpha_j_old - theta[j])

                    b1 = b - E_i - y[i] * (theta[i] - alpha_i_old) * K_matrix[i, i] - y[j] * (theta[j] - alpha_j_old) * K_matrix[i, j]   
                    b2 = b - E_j - y[i] * (theta[i] - alpha_i_old) * K_matrix[i, j] - y[j] * (theta[j] - alpha_j_old) * K_matrix[j, j]

                    if 0 < theta[i] < self.c: b = b1
                    elif 0 < theta[j] < self.c: b = b2
                    else: b = (b1 + b2) / 2

            if np.linalg.norm(theta - alpha_prev) < self.tol:
                break
        
        return {"theta": theta, "b": b, "X": X, "y": y}
    
    # Fit 
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        self.models = []

        if n_classes <= 2:
            binary_y = np.where(y == self.classes[0], -1, 1)
            self.models.append(self._fit_binary(X, binary_y))
            
        elif self.decision_function_shape == 'ovr':
            # One vs Rest
            
            for c in self.classes:
                binary_y = np.where(y == c, 1, -1)
                self.models.append(self._fit_binary(X, binary_y))

        elif self.decision_function_shape == 'ovo':
            # One vs One
            
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    idx = np.where((y == self.classes[i]) | (y == self.classes[j]))
                    X_sub, y_sub = X[idx], y[idx]
                    binary_y = np.where(y_sub == self.classes[i], 1, -1)
                    model = self._fit_binary(X_sub, binary_y)
                    model['cls_pair'] = (self.classes[i], self.classes[j])
                    self.models.append(model)
                    
        # Bias value and other theta values 
        self.intercept_ = self.models[0]['b']   
        self.coef_ = self.models[0]['theta']                     

    # Decision function for a single model
    def _get_score(self, X, model):
        scores = []
        for x in X:
            score = np.sum(model['theta'] * model['y'] * [self.kernel(x_tr, x) for x_tr in model['X']]) + model['b']
            scores.append(score)
        return np.array(scores)

    # Predict             
    def predict(self, X):
        if len(self.classes) <= 2:
            scores = self._get_score(X, self.models[0])
            return np.where(scores >= 0, self.classes[1], self.classes[0])

        if self.decision_function_shape == 'ovr':
            # Highest confidence wins
            all_scores = np.array([self._get_score(X, m) for m in self.models])
            return self.classes[np.argmax(all_scores, axis=0)]

        elif self.decision_function_shape == 'ovo':
            # Voting system
            votes = np.zeros((len(X), len(self.classes)))
            for m in self.models:
                scores = self._get_score(X, m)
                preds = np.where(scores >= 0, m['cls_pair'][0], m['cls_pair'][1])
                for idx, p in enumerate(preds):
                    votes[idx, np.where(self.classes == p)[0][0]] += 1
            return self.classes[np.argmax(votes, axis=1)]

    # Score 
    def score(self, X_new, y):
        # Return accuracy score of the model
        y_pred = self.predict(X_new)
        return np.mean(y_pred == y) 

# Regression remains largely the same but usually doesn't use OVO/OVR
class SVR():
    def __init__(self, c=1.0, epsilon=0.1, kernel='Linear', degree=3, gamma=1, coef0=1, lr=0.001, max_iter=1000):
        self.c = c
        self.epsilon = epsilon
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.lr = lr
        self.max_iter = max_iter

        if kernel == 'Linear':
            self.kernel = lambda x1, x2: linear_kernel(x1, x2)
        elif kernel == 'Polynomial':
            self.kernel = lambda x1, x2: polynomial_kernel(x1, x2, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
        elif kernel == 'RBF':
            self.kernel = lambda x1, x2: rbf_kernel(x1, x2, gamma=self.gamma)
        elif kernel == 'Sigmoid':
            self.kernel = lambda x1, x2: sigmoid_kernel(x1, x2, gamma=self.gamma, coef0=self.coef0)
        else:
            raise ValueError("Unknown kernel.")

    def fit(self, X, y):
        self.X = X
        self.y = y
        m = len(X)
        self.alpha = np.zeros(m)
        self.b = 0

        self.K_matrix = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                self.K_matrix[i, j] = self.kernel(X[i], X[j])

        for _ in range(self.max_iter):
            for i in range(m):
                pred = np.sum(self.alpha * self.K_matrix[:, i]) + self.b
                error = pred - self.y[i]
                if abs(error) > self.epsilon:
                    self.alpha[i] -= self.lr * error
                    self.alpha[i] = np.clip(self.alpha[i], -self.c, self.c)
                    self.b -= self.lr * error

        # Bias value and other theta values 
        self.intercept_ = self.models[0]['b']   
        self.coef_ = self.models[0]['theta'] 
        
    def predict(self, X):
        predictions = []
        for x in X:
            pred = np.sum(self.alpha * [self.kernel(x_tr, x) for x_tr in self.X]) + self.b
            predictions.append(pred)
        return np.array(predictions)

    def score(self, X_new, y):
        # Return R^2 score 
        y_pred = self.predict(X_new)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - ss_residual / ss_total
