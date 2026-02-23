# Known that ada boost is boosting with decsion trees as base estimator 
# We have multiple of weak learners to have a strong learner at the end working sequentially and each weak learner is trained on the previous one errors
from Libraries.DecisionTree import *

import numpy as np
# We need this copy for cloning decision tree estimator
import copy

# We work with the base class then inheritence happen to build the classification or regression tree
class AdaBoost():
    
    # Initialization
    def __init__(self, base_estimator, n_estimators=50, learning_rate=1.0):
        # Later we will decide base estimator is classification or regression tree 
        # Stopping condtions will be no update in weights if error is 0 or 1 
        self.base_estimator = base_estimator
        # Others is the same default paramerts of API
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # Storage of leaners and their weights
        self.estimators_ = []
        self.estimator_weights_ = []

    # Fit
    def fit(self, X, y):
        # Get number of samples
        n_samples = X.shape[0]

        # As usaul we fit means we collect paramerts to use it later
        # We need at the end estimators and their weights to use it in prediction --> (except MMSE) no work with hypothesis 
        # Intialize weights for all samples which will be 1 / number of samples and gonna updated each iterarion
        # Matrix of weights for each sample 
        # Here we will use np.full to give the whole number of samples n_samples the same weight 
        sample_weights = np.full(n_samples, 1 / n_samples)

        # Loop over all learners
        for learner in range(self.n_estimators):
            
           # Create clone of learner to fit the data and update the weights
           estimator = copy.deepcopy(self.base_estimator)

           # Fit the cloned estimator with sample weights
           # Here decsion trees will work with passing sample weights not counter 
           estimator.fit(X, y, sample_weight=sample_weights)

           # Update sample weights based on the error of the current learner and Calculate error (Alpha) using _boost function which will be different based on classification or regression 
           sample_weights, alpha = self._boost(X, y, sample_weights, estimator)

           # Stopping condition (no more updates on weights) => we return none in the case of error is 0 or 1 in the _boost function and we break the loop here
           if sample_weights is None:
               break

           # Append new model to estimators_ with the equivelant weight in estimatr_weights_ to be used in prediction --> (except MMSE) no work with hypothesis 
           self.estimators_.append(estimator)
           self.estimator_weights_.append(alpha) 

        return self

    # Boost function
    # This function will be responsible for calculating the error of the current learner and updating the sample weights based on that error   
    # Here we create just abstract so after inheritence each of classification and regression has its own methodoly of aggregation for final prediction
    def _boost(self, X, y, sample_weights, estimator):
        raise NotImplementedError
        
# Ada Boost Classifier class
class AdaBoostClassifier(AdaBoost):
    
    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters       
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0):

        # Handle the error of estimator
        # If estimator is None or not equal decision tree classifier object or max_depth is not 1
        # is_instance is used to check object type 
        if estimator is None or not isinstance(estimator, DecisionTreeClassifier) or (hasattr(estimator, 'max_depth') and estimator.max_depth != 1):
            # Here is the correct learner
            base_estim_to_use = DecisionTreeClassifier(max_depth=1)
        else:
            base_estim_to_use = estimator

        super().__init__(base_estim_to_use, n_estimators, learning_rate)  

    # Boost function
    # It is for classification
    # It is the most important part in ada boost and it is the one which update the weights and the error for each learner
    def _boost(self, X, y, sample_weights, estimator):
        # Here we will return sample weight and performance of stump (PS) (Alpha) (Amount of say)
        # For binary classification we need to convert labels to -1 and 1 for the mathematical calculation of error and weight update
        # We train our learner
        y_pred = estimator.predict(X)

        # Identify incorrect predictions
        # When training y not equal to actual y
        # We do this so that we can calculate the error (TE) of the learner and update the weights accordingly
        # Make sure type is int at the end
        # Matrix of 1 for incorrect predictions and 0 for correct predictions
        incorrect = (y_pred != y).astype(int)

        # Get the number of unique classes (k) which will be used in the amount of say calculation
        classes = np.unique(y)
        n_classes = len(np.unique(y))

        # Calculate the error of the learner (TE) which is the sum of weights of incorrect predictions divided by the sum of all weights
        # Take care that incorrect total is sum of all samples which were incorrect
        total_error = np.dot(sample_weights, incorrect) / np.sum(sample_weights)

        # Stopping condition (error is 0 or  1 which means the learner is perfect or completely wrong and no more updates on weights)
        # Amount of say will be the highest when error is 0 and the lowest when error is 1
        if total_error <= 0: return None, 1.0 
        if total_error >= 1.0 - (1.0 / n_classes): return None, None

        # Calculate the amount of say (Alpha) which is the logarithm of the ratio of correct predictions to incorrect predictions multiplied by the learning rate
        # Add satbility term to avoid division by zero and log of zero (1e-9)
        alpha = self.learning_rate * (np.log((1.0 - total_error) / (total_error + 1e-9)) + np.log(n_classes - 1.0))

        # Update weight and for software implementation it is different than the mathematical one as we just make e power amout of say * number of incorrect
        # So we higher the incorrect weights and lower the correct ones as alpha will be the same but the trick that incorrect = 0 or positive number
        sample_weights *= np.exp(alpha * incorrect)

        # Nomalize weights to sum to 1
        sample_weights /= np.sum(sample_weights)

        return sample_weights, alpha
    
    # Predict probabilities
    # This function is ready to use in soft oting later
    def predict_proba(self, X):
        # Loop over learners and calculate the weighted average of their predictions to get the final probability estimates for each class then get the max of them to get the final prediction
        # SAMME method
        # Unlike text books in both cases wether it is binary or multi class we will use the same method of weighted average of probabilities and then get the max of them to get the final prediction
        # We get the number of classes from first estimator as all estimators will have the same number of classes
        n_classes = self.estimators_[0].n_classes_

        # Make probabilities matrix initialized with zeros for each class and each sample
        # Samples represent as indexes and classes represent as columns
        probas = np.zeros((X.shape[0], n_classes))
        # We gather all aplha values to get the total alpha for normalization of probabilities
        total_alpha = np.sum(self.estimator_weights_)

        # Here all alphas are total not for specificlabel so we muliply probabilities of each classes / estimator to make each class have the percentage of this alpha and we add old to new     
        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            # We get the total number of probabilities for each sample per all estimators
            probas += (alpha / total_alpha) * estimator.predict_proba(X)
        return probas

    # Predict
    def predict(self, X):
        # We just return the max of the probabilities of all labels to get the final prediction
        # We make axis=1 to ge the max of rows to get the max probability for each sample across all classes
        return np.argmax(self.predict_proba(X), axis=1)  
    
    # Score
    def score(self, X, y):
        # Return accuracy score 
        return np.mean(self.predict(X) == y)

# Ada Boost Regressor class
class AdaBoostRegressor(AdaBoost):  

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters 
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0):

        # Handle the error of estimator
        # If estimator is None or not equal decision tree regressor object or max_depth is not 1
        # is_instance is used to check object type 
        if estimator is None or not isinstance(estimator, DecisionTreeRegressor) or (hasattr(estimator, 'max_depth') and estimator.max_depth != 1):
            # Here is the correct learner
            base_estim_to_use = DecisionTreeRegressor(max_depth=1)
        else:
            base_estim_to_use = estimator

        super().__init__(base_estim_to_use, n_estimators, learning_rate) 

    # Boost function
    # It is for regression
    # It is the most important part in ada boost and it is the one which update the weights and the error for each learner
    def _boost(self, X, y, sample_weights, estimator):
        # Here we will return sample weight and performance of stump (PS) (Alpha) (Amount of say)
        # The steps of regression boostig is different than classification 
        # We train our learner
        y_pred = estimator.predict(X)   

        # Calculate residuals
        residuals = np.abs(y - y_pred)

        # Calculate max error
        total_error = np.max(residuals)

        # Stopping condition (error is 0)
        # Amount of say will be the highest when error is 0 
        if total_error <= 0: return None, 1.0 

        # Calculate the relative error
        # Add satbility term to avoid division by zero 
        e_rel = residuals / (total_error + 1e-9)
        
        # Calculate weighted error
        weighted_error = np.sum(sample_weights * e_rel)

        # Stopping criteria (error is highes) --> weighted error is greater than or equal to 0.5 which means the learner is completely wrong and no more updates on weights
        if weighted_error >= 0.5: return None, None

        # Calculate Beta
        beta = weighted_error / (1.0 - weighted_error)

        # Calculate the amount of say (Alpha)
        # Add satbility term to avoid division by zero 
        alpha = self.learning_rate * np.log(1.0 / (beta + 1e-9))

        # Update weights
        # So based on e_rel when you are close the sample weight will be lower and when you are far the sample weight will be higher 
        # When beta close to zero that means lower e_rel and when beta close to 1 that means higher e_rel
        sample_weights *= np.power(beta, 1 - e_rel)

        # Nomalize weights to sum to 1
        sample_weights /= np.sum(sample_weights)

        return sample_weights, alpha

    # Predict
    def predict(self, X):
            # We just return the average of predictions instead of getting max probabiliy in classification
            # Collecting predictions from estimators in array
            predictions = np.array([estimator.predict(X) for estimator in self.estimators_]).T

            # Applying the weight (alpha) to each tree's prediction
            # Return the weighted average of predictions for each sample
            return np.average(predictions, axis=1, weights=self.estimator_weights_)
    
    # Score
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)