import numpy as np
# Naive Bayes base class
class BaseNB():

    # Initialization
    def __init__(self, method='gaussian', alpha=1.0):
        self.method = method
        self.alpha = alpha

    # Guassian likelihood function
    # For the Bernoulli and Multinomial functions they are just counting
    def guassian_likelihood_log_prob(self, X, mean, var):
        # Here we log result to avoid muliplication and convert it to summation for each class
        log_probs = np.zeros((self.n_classes, X.shape[0]))
        for c in range(self.n_classes):
            # Add stability to avoid zero division
            log_probs[c, :] = -0.5 * np.sum(np.log(2 * np.pi * self.var[c] + 1e-9) + ((X - self.mean[c]) ** 2) / (self.var[c] + 1e-9), axis=1)
        return log_probs
    
    # Fit 
    def fit(self, X, y):
        # In learning phase we calculate prior probabilities and likelihood parameters and we neglect evidence cause it is constant for all classes so saving time
        # For perior probabilities we calculate all formula cause its parts are already paramters
        # We calculate part of likelihood based on method which is parameter terms which add later to the full formula of likelihood
        # Intialize parameters
        # We collect unique classes
        self.classes = np.unique(y)
        # We calcualte the number of classes first
        self.n_classes = len(np.unique(y))
        # Get number of rows and columns of training data
        self.n_samples, self.n_features = X.shape
        # Intilaize prior probabilities
        self.classes_prior = np.zeros(self.n_classes)

        # Get likelihood parameters based on method
        # Intialize paramerts then will fill them in the general array for each class (filtering)
        # Intialize Guassian parameters
        if self.method == 'gaussian':
            self.mean = np.zeros((self.n_classes, self.n_features))
            self.var = np.zeros((self.n_classes, self.n_features))

        # intialize Bernoulli and Multinomial parameters
        elif self.method == 'bernoulli' or self.method == 'multinomial':
            self.feature_count = np.zeros((self.n_classes, self.n_features))

        else:
            raise ValueError("Method not found.")

        # Loop over each class to calculate prior probability and likelihood parameters to get pesterior probability later 
        for idx, c in enumerate(self.classes):
            # Indexing is very important cause prior probabilities and likelihood parameters are arrays so we sort in order
            # Filtering samples belong to current class then reduce sample space
            X_c = X[y == c]

            # Calculate prior probability which will be count of samples belong to class / total samples belong to all classes
            self.classes_prior[idx] = float(X_c.shape[0] / self.n_samples)

            # Calculate likelihood parameters based on method for each class for all features
            # Guassian likelihood Parameters
            if self.method == 'gaussian':
                # axis = 0 for mean and variance calculation along rows (features)
                self.mean[idx] = X_c.mean(axis=0)
                # Add stability to avoid zero division
                self.var[idx] = X_c.var(axis=0) + 1e-9  

            # Multinomial likelihood Parameters
            elif self.method == 'multinomial':
                # Add alpha for laplace smoothing to avoid deviding zero or being zero cause of one of more probabilities being zero
                # Multinomianl is current feature count / smaples count for all features in class 
                # Here we devide by the sum of occurrences smaples in class
                self.feature_count[idx] = (X_c.sum(axis=0) + self.alpha) / (X_c.sum() + self.alpha * self.n_features)

            # Bernoulli likelihood Parameters
            elif self.method == 'bernoulli':    
                # Bernoulli is current feature count / samples number for currnet features in class
                # We devide by the count of occurences of the feature in class
                self.feature_count[idx] = (X_c.sum(axis=0) + self.alpha) / (X_c.shape[0] + 2 * self.alpha)

        return self

    # Predict    
    def predict(self, X):
        # Calculate log posterior probability for each class which will be log prior probability + log likelihood
        # Intialize log posterior and perior array
        log_posteriors = np.zeros((self.n_classes, X.shape[0]))
        
        # Loop over each class to calculate log posterior probability
        for idx, c in enumerate(self.classes):
            # Get log prior probability
            # We don't need log_periors and log_likelihood to be arrays permentant matrix just temporary variable carry array every time
            # Add stability to avoid zero addition
            log_perior = np.log(self.classes_prior[idx] + 1e-9)

            # Get log likelihood based on method and apply the full formula of likelihood
            # Gaussian likelihood
            if self.method == 'gaussian':   
                log_likelihood_all = self.guassian_likelihood_log_prob(X, self.mean[idx], self.var[idx])
                log_likelihood = log_likelihood_all[idx, :]

            # Multinomial likelihood
            elif self.method == 'multinomial':
                # Here is the multinomial version of feature count
                # Log probability calculation and we multiplication with new data so when word appears more times then its probability will be ehigher times and instead of power we use multiplication cause of log
                # Use clip(min=1e-15) to prevent log(0) which leads to -inf or NaN
                # We only clip the lower bound because the formula only involves log(probs)
                # Even if probs reach 1.0, log(1.0) is 0, which is mathematically stable
                # We choose 1e-15 as it is one of the safest smallest numbers in computer which is closest to zero without being zero and it is also small enough to not affect the probabilities significantly
                probs = np.clip(self.feature_count[idx], 1e-15, 1.0)
                log_likelihood = X @ np.log(probs.T)

            # Bernoulli likelihood  
            elif self.method == 'bernoulli':
                # Here is the bernoulli version of feature count
                # Log probability calculation and we multiplication with new data so when word appears or disappear then it affects its probability so we use addition of p and 1-p parts to cover all cases instead of mulitplication cause of log
                # Use clip(1e-15, 1 - 1e-15) to prevent log(0) in both parts of the formula:
                # 1. np.log(probs) -> requires probs > 0
                # 2. np.log(1 - probs) -> requires probs < 1
                # 3- We clip to less then one so practically no 100 % probabaility in real world also prevent more than one which leads to negative or NaN
                # We choose 1e-15 as it is one of the safest smallest numbers in computer which is closest to zero without being zero and it is also small enough to not affect the probabilities significantly
                probs = np.clip(self.feature_count[idx], 1e-15, 1 - 1e-15)
                log_likelihood = X @ np.log(probs) + (1 - X) @ np.log(1 - probs)

            # Final calculation of log posterior probability  
            log_posteriors[idx, :] = log_perior + log_likelihood  

        # Return the class with the highest log posterior probability
        return self.classes[np.argmax(log_posteriors, axis=0)]
    
    # Score
    def score(self, X, y):
        # Return accuracy score 
        return np.mean(self.predict(X) == y) 

# Guassian Naive Bayes class
class GaussianNB(BaseNB):
    def __init__(self):
        super().__init__(method='gaussian')

# Multinomial Naive Bayes class
class MultinomialNB(BaseNB):    
    def __init__(self, alpha=1.0):
        # Here we need to add alpha parameter for child so we can pass it to the parent
        super().__init__(method='multinomial', alpha=alpha)

# Bernoulli Naive Bayes class
class BernoulliNB(BaseNB):      
    def __init__(self, alpha=1.0):
        # Here we need to add alpha parameter for child so we can pass it to the parent
        super().__init__(method='bernoulli', alpha=alpha)