from Libraries.LinearRegression import *
from Libraries.LogisticRegression import *
from Libraries.KNN import *
from Libraries.SVM import *
from Libraries.NaiveBayes import *
from Libraries.DecisionTree import *
from Libraries.RandomForest import *
from Libraries.Voting import *
from Libraries.AdaBoost import *
from Libraries.GradientBoost import *
import sys, importlib, shutil
shutil.rmtree("Libraries/__pycache__", ignore_errors=True)
sys.modules.pop("Libraries.RandomForest", None)
importlib.invalidate_caches()

import numpy as np
# We need this copy for cloning base estimator
import copy
# For parallelism and working with multiple of cores 
from joblib import Parallel, delayed

# Bagging class for Classification and Regression
# We work with the base class then inheritence happen to build the classification or regression bagging
class BaggingBase():

    # Initialization
    def __init__(self, base_estimator = None, n_estimators = 10, random_state = None, n_jobs = None):
        # Here we gonna define the base estimator later in classification or regression
        # Others is the same default paramerts of API
        # n_jobs=None or 1 is the same and -1 means using all cores
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = []


    # We have helper functions
    # Bootstrapping samples function
    def _bootstrap_sample(self, X, y):
        # Here we fetch the number of samples from x 
        # X is combination of samples x features
        n_samples = X.shape[0]

        # Every time we select random numbers of indice with enabling replacement option so we can use same indice in more models
        # So we can get the sample 0, 1 or multiple of times 
        # Here we choose randomly from all samples and we need the size of bootstrapped data = size of orginal data that is why replace option should be applied
        indices = np.random.choice(n_samples, n_samples, replace=True)

        # Here we return the bootstrapped data
        return X[indices], y[indices]

    # Clone function
    def _clone(self):
        # Here we use the library of copy to return clone of base_estimator
        # The idea here to siolate object per model
        # It is only for learning phase
        return copy.deepcopy(self.base_estimator)
    
    # Modeling function
    # We will have one fit function per every model then combine all fits in general fit function for all memebers
    # Fit single model
    def _fit_single_model(self, X, y):
        # First we recieve the bootstrapped data to model it from given X and y
        X_samples, y_samples = self._bootstrap_sample(X, y)
        
        # Get cloned copy of base_estimator
        # Every time we gonna pass the object of base estimator
        cloned_model = self._clone()

        # We use the fit function for specific single model which was already built from scratch before
        cloned_model.fit(X_samples, y_samples)

        # Every time we return the fitted model
        return cloned_model
    
    # Fit all models function
    # Here we gather all fitted models 
    def fit(self, X, y):
        # We set random state wuth the number user entered and seed it
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Loop over estimator number collecting all fitted models in one place to be ready for test phase
        # We call _fit_single_model every time
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_model)(X, y) for estimator in range(self.n_estimators)
        )    
        
        # We return nothing
        return self
    
    # Predict probabilities
    # This function is ready to use in soft oting later
    def predict_proba(self, X):
        probas = Parallel(n_jobs=self.n_jobs)(
            delayed(model.predict_proba)(X) for model in self.models
        )
        probas = np.array(probas)
        return np.mean(probas, axis=0)

    # The same structure of fit will be built for predict
    # We will have one predict function per every model then combine all predicts in general predict function for all memebers
    # Predict single model
    def _predict_single_model(self, cloned_model, X):
        # We use the predict function for specific single model which was already built from scratch before
        # We use test data
        return cloned_model.predict(X)

    # Precit all models function
    # Here we gather all precited models 
    def predict(self, X):
        # Loop over model collecting all predictions in one place to be ready for aggregation later using cpu cores (parallelism)
        # We call _predict_single_model every time
        # to apply predict we need to concatenate with fit first this is gathered in self.models
        # Here every fitted model work with all samples 
        # So each sample have multiple predictions for all models
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_single_model)(model, X) for model in self.models
        )    
        
        # Preditction as numpy array
        predictions = np.array(predictions)
        
        # After collecting predictions we are ready for aggregation
        return self._aggregate(predictions)

    # Aggregation function    
    # Aggregation to decide the final output
    # Here we create just abstract so after inheritence each of classification and regression has its own methodoly of aggregation for final prediction
    def _aggregate(self, predictions):
        raise NotImplementedError  

# Bagging ensemble Classifier class
class BaggingClassifier(BaggingBase):

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters
    def __init__(self, base_estimator=None, n_estimators=10, random_state=None, n_jobs=1):
        
        # Default estimator for classification as API
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier()

        super().__init__(base_estimator, n_estimators, random_state, n_jobs)

    # Aggregation function    
    def _aggregate(self, predictions):

        # Get number of samples as every sample wich will be columns as every column represents sample and indexes represent models  
        n_samples = predictions.shape[1]

        # Gather all predictions in one array and make sure its data type the same as the source predictions array
        # Intialize
        final_predictions = np.zeros(n_samples, dtype=predictions.dtype)

        # Apply majority voting 
        # Loop over all samples to get the most voted label
        for sample in range(n_samples):
            # Featch models of samples values
            # Every sample represents column
            votes = predictions[:, sample]

            # Here we get unique classes then counting them
            # We return two related arrays of classes and labels 
            # One for unique values and other for unique counts
            # So return the index of laregest count will be equivelant to the label of largest count
            classes, counts = np.unique(votes, return_counts=True)
            majority_voting = classes[np.argmax(counts)]
            final_predictions[sample] = majority_voting

        # Return the final predictions of the test data
        return final_predictions
    
    # Score
    def score(self, X, y):
        # Return accuracy score 
        return np.mean(self.predict(X) == y)

# Bagging ensemble Tree Regressor class
class BaggingRegressor(BaggingBase):

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters
    def __init__(self, base_estimator=None, n_estimators=10, random_state=None, n_jobs=1):
        
        # Default estimator for classification as API
        if base_estimator is None:
            base_estimator = DecisionTreeRegressor()

        super().__init__(base_estimator, n_estimators, random_state, n_jobs)

    # Aggregation function 
    def _aggregate(self, predictions):
        # We just return the mean of every sample wich will be columns as every column represents sample and indexes represent trees
        return np.mean(predictions, axis=0)  
  
    # Score
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)