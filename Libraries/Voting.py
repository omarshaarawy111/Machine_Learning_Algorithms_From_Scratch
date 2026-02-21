import numpy as np
# For parallelism and working with multiple of cores 
from joblib import Parallel, delayed

class VotingBase():

    # Initialization
    def __init__(self, estimators = None, n_jobs = None):
        # Here we gonna define the estimators used later in classification or regression
        # Others is the same default paramerts of API
        # n_jobs=None or 1 is the same and -1 means using all cores
        self.estimators = estimators
        self.n_jobs = n_jobs
        self.models = []

   
    # Modeling function
    # We will have one fit function per every model then combine all fits in general fit function for all memebers
    # Fit single model
    def _fit_single_model(self,current_estimator, X, y):      
        # Get estimator from the list of estimators and fit it directly 
        # No need to create helper function like _make_estimator or _clone
        # Every time we gonna pass the object of current estimator
        # We use the fit function for specific single model which was already built from scratch before
        # Here no bottstrapped data so we work with X and y directly
        current_estimator.fit(X, y)

        # Every time we return the fitted model
        return current_estimator
    
    # Fit all models function
    # Here we gather all fitted models 
    def fit(self, X, y):
        # No random state exist
        # Loop over estimator number collecting all fitted models in one place to be ready for test phase
        # We call it n_estimators but it is actually a list of estimators not a number
        # We call _fit_single_model every time
        self.models = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_model)(curtrent_estimator, X, y) for curtrent_estimator in self.estimators
        )    
        
        # We return nothing
        return self
    
    # The same structure of fit will be built for predict
    # We will have one predict function per every model then combine all predicts in general predict function for all memebers
    # Predict single model
    def _predict_single_model(self, current_estimator, X):
        # We use the predict function for specific single model which was already built from scratch before
        # We use test data
        return current_estimator.predict(X)
    
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
    
# Voting ensemble Classifier class
class VotingClassifier(VotingBase):

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters
    def __init__(self, estimators = None, n_jobs=1, voting = 'hard'):
        
         # Here we handle if not estimator we raise and error
        if estimators is None:
            raise ValueError('No estimators have been passed.')
        
        super().__init__(estimators, n_jobs)

        # Here handling error of voting know that default is hard voting
        if voting not in ("hard", "soft"):
            raise ValueError("Voting must be 'hard' or 'soft'.")
        
        self.voting = voting

    # Aggregation function  
    # This represents hard voting which is the same as bagging 
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
    
    # Predict function
    def predict(self, X):
        if self.voting == "soft":
            # Get probabilities from all models
            probas = Parallel(n_jobs=self.n_jobs)(
                delayed(model.predict_proba)(X) for model in self.models
            )

            # shape is 3D: (n_models, n_samples, n_classes)
            probas = np.array(probas) 

            # Average probabilities across models
            # shape is 2D: (n_samples, n_classes)
            avg_proba = np.mean(probas, axis=0)  

            # Step 3: Choose class with highest probability
            # Shape is 1D: (n_samples,) and value is the class label with highest average probability
            final_labels = np.argmax(avg_proba, axis=1)  

            return final_labels
        
        # Hard voting logic is the same depends on base class
        return super().predict(X)

    # Score
    def score(self, X, y):
        # Return accuracy score 
        return np.mean(self.predict(X) == y)

# Voting ensemble Tree Regressor class
class VotingRegressor(VotingBase):

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters
    def __init__(self, estimators = None, n_jobs=1):
        
        # Here we handle if not estimator we raise and error
        if estimators is None:
            raise ValueError('No estimators have been passed.')

        super().__init__(estimators, n_jobs)

    # Aggregation function 
    def _aggregate(self, predictions):
        # We just return the mean of every sample wich will be columns as every column represents sample and indexes represent models
        return np.mean(predictions, axis=0)  
  
    # Score
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)