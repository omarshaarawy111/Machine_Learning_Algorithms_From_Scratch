import numpy as np
# KNN class for classification and regression
class KNN():

    # Initialization
    def __init__(self,task = 'Classification', k = 3, distance_metric = 'Euclidean'):
        self.task = task
        self.k = k
        self.distance_metric = distance_metric

    # distance function
    def _distance(self, a, b):
        if self.distance_metric == 'Euclidean':
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.distance_metric == 'Manhattan':
            return np.sum(np.abs(a - b))
        else:
            raise ValueError('Unknown distance metric.')

    # Fit 
    def fit(self, x, y):
        # Just store training data
        self.X_train = x       
        self.y_train = y


    # Predict 
    def predict(self, X):
        y_pred = []
        for x in X:
            # Calculate distances between X_new and all training samples
            distances = np.array([self._distance(x, xi) for xi in self.X_train])
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            # Get labels of k nearest neighbors
            k_labels = self.y_train[k_indices]

            # Classification usign majority vote
            if self.task == 'Classification':
                # Count occurrences of each class
                classes, counts = np.unique(k_labels, return_counts=True)
                # Take the class with maximum count
                pred = classes[np.argmax(counts)]
                y_pred.append(pred)

            elif self.task == 'Regression':
                # Take the mean of k labels
                pred = np.mean(k_labels)
                y_pred.append(pred)

            else:
                raise ValueError('Unknown task type.')    

        return np.array(y_pred)    
    
    # Predict probabilities
    # This function is ready to use in soft oting later
    def predict_proba(self, X):
        probas = []
        for x in X:
            distances = np.array([self._distance(x, xi) for xi in self.X_train])
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            classes, counts = np.unique(k_labels, return_counts=True)
            probs = np.zeros(len(np.unique(self.y_train)))
            for c, count in zip(classes, counts):
                idx = np.where(np.unique(self.y_train) == c)[0][0]
                probs[idx] = count / self.k
            probas.append(probs)
        return np.array(probas)  
       
    # Score 
    def score(self, X_new, y):
            y_pred = self.predict(X_new)

            # For classification, Return accuracy
            if self.task == 'Classification':
                return np.mean(y_pred == y)
            
            # For regression, Return R^2 score
            elif self.task == 'Regression':
                # R^2 score
                ss_total = np.sum((y - np.mean(y))**2)
                ss_res = np.sum((y - y_pred)**2)
                return 1 - ss_res / ss_total