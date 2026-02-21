# Known that random forest is bagging + decision trees but with some edits 
# So academiacally we just inherit bagging + decision trees and add edits
# Known that random forest work with only one base estimator (Descision Tree) cloning it within bootsreapped data with replacement
# But for API Design it is seperated from bagging and no inherit happended and we just duplicate bagging functions from scratch
from Libraries.DecisionTree import *
from Libraries.Bagging import *

import numpy as np
# For parallelism and working with multiple of cores 
from joblib import Parallel, delayed

# Random Forest class for Classification and Regression
# We work with the base class then inheritence happen to build the classification or regression random forest
class RandomForestBase():

    # Intialization 
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True,  random_state=None, n_jobs=None):
        # Here no pruning happen and only prepruning paramaters is for capacity regulazation only
        # Here we intialize random forest trees properties there is mandatory and optional propeties based on use case and also your capacity regularization decision   
        # min_samples_split = 2, min_samples_leaf = 1 these are mandatory cause they are logically the min amount to split or work rather than it isn't logical
        # min_impurity_decrease = 0.0, ccp_alpha = 0.0 becasue the default is no pruning till you decide
        # There are common factors that will be passed from random fores to called trees like max_depth, min_samples_split, min_samples_leaf, max_features which was none there but here will be intialized, random_state and n_jobs
        # We can specifiy max features to use per decsion tree with three ways (log2, sqrt, number)
        # Here base estimator is desicion tree and no need to specifiy it
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
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
    
    # Function of selecting max features technique for each tree
    def _get_max_features(self, n_features):
        # Here we pass the cirteria of selecting max features to use in each tree and we return the number of features to use in each tree based on the criteria
        # For SQRT the number
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        
        # For the Log2 number
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        
        # For the number only
        elif self.max_features is None:
            return n_features
        
        # This is the case of another option
        return self.max_features
    
    # Here we create just abstract so after inheritence each of classification and regression has its own methodoly of make new estimator of tree or predict
    # Make estimator function to make new tree with the criteria of max features to use in each tree
    def _make_estimator(self, max_features):
        raise NotImplementedError

    # Modeling function
    # We will have one fit function per every tree then combine all fits in general fit function for all trees
    # Fit single tree
    def _fit_single_model(self, X, y):
        if self.bootstrap:
            # If bootstrap is true 
            # First we recieve the bootstrapped data to model it from given X and y
            X_samples, y_samples = self._bootstrap_sample(X, y)
        else:
            # If bootstrap is false we just work with the original data
            X_samples, y_samples = X, y

        # Get number of features to pass it to make tree
        n_features = X_samples.shape[1]

        # Create tree estimator
        # Every time we gonna pass the max_features using get_max_features function
        tree = self._make_estimator(self._get_max_features(n_features))

        # We use the fit function for specific single tree which was already built from scratch before
        tree.fit(X_samples, y_samples)

        # Every time we return the fitted model
        return tree
    
    # Fit all models function
    # Here we gather all fitted models 
    def fit(self, X, y):
        # We set random state wuth the number user entered and seed it
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Loop over trees number collecting all fitted models in one place to be ready for test phase
        # We call _fit_single_model every time
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_model)(X, y) for tree in range(self.n_estimators)
        )    
        
        # We return nothing
        return self


    # Precit function
    # Here no _predict_single_model as bagging we just use one fucntion only
    # Here we gather all precited models 
    def predict(self, X):
        # Loop over model collecting all predictions in one place to be ready for aggregation later using cpu cores (parallelism)
        # to apply predict we need to concatenate with fit first this is gathered in self.models
        # Here every fitted tree work with all samples 
        # So each sample have multiple predictions for all trees
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(model.predict)(X) for model in self.models
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

# Random Forest Classifier class
class RandomForestClassifier(RandomForestBase):

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters    
    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
        n_jobs=None
    ):
       
        # That is the only parameter i take from user and the rect will be passed to the parent 
        self.criterion = criterion
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs
        )

    # Helper functions
    # Make estimator function and the idea is very simple we just call tree classification class taking it is result with determined number of max_features and other essential paramerts
    def _make_estimator(self, max_features):
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_features
        )
    
    # Aggregation function    
    def _aggregate(self, predictions):

        # Get number of samples as every sample wich will be columns as every column represents sample and indexes represent trees
        n_samples = predictions.shape[1]

        # Gather all predictions in one array and make sure its data type the same as the source predictions array
        # Intialize
        final_predictions = np.zeros(n_samples, dtype=predictions.dtype)

        # Apply majority voting 
        # Loop over all samples to get the most voted label
        for sample in range(n_samples):
            # Featch trees of samples values
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
    
# Random Forest Regressor class
class RandomForestRegressor(RandomForestBase):

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters    
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        random_state=None,
        n_jobs=None
    ):
       
       # No need to wrtie criterion = 'mse' as default because in our implementation of decision tree regression we just work with one criterion which is mse and no need to specify it as user input
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=n_jobs
        )

    # Helper functions
    # Make estimator function and the idea is very simple we just call tree regression class taking it is result with determined number of max_features and other essential paramerts
    def _make_estimator(self, max_features):
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=max_features
        )
    
    # Aggregation function    
    def _aggregate(self, predictions):

        # We just return the mean of every sample wich will be columns as every column represents sample and indexes represent models
        return np.mean(predictions, axis=0)  
    
    # Score
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)