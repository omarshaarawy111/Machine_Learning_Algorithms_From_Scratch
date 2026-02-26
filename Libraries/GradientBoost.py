# Known that gradient boost is boosting with decsion trees as base estimator 
# We have multiple of weak learners to have a strong learner at the end working sequentially and each weak learner is trained on the previous one errors
from Libraries.DecisionTree import *

import numpy as np
# We need this copy for cloning decision tree estimator
import copy

# We work with the base class then inheritence happen to build the classification or regression tree
class GradientBoostBase:

    # Initialization
    def __init__(self, base_estimator, n_estimators=100, learning_rate=0.1, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, tol=1e-6, random_state=None):
        # Later we will decide base estimator is classification or regression tree 
        # Stopping condition will be the number of weak learners or enable early stopping (error has no more improvement after k iterations)
        self.base_estimator = base_estimator
        # Others is the same default paramerts of API
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        # Early stopping parameters
        self.early_stopping = early_stopping
        # Percentage of data to be vlaidation set
        self.validation_fraction = validation_fraction
        # The number of times (patience) till no change in erorr happen then early stop comes
        self.n_iter_no_change = n_iter_no_change
        # The min tolerance which means the min improvement and it is very close to zero
        self.tol = tol
        self.random_state = random_state
        # Storage of leaners
        self.estimators_ = []
        # For early stopping
        # Train loss and validation loss array
        self.train_loss_ = []
        self.val_loss_ = []
        # The best number of learners before we early stop
        self.best_iteration_ = None

    # Here we create just abstract so after inheritence each of classification and regression has its own methodoly of aggregation for final prediction
    def fit(self, X):
        raise NotImplementedError
    
    # Helper function
    def _clone_estimator(self):
        # Create clone of learner to fit the data 
        return copy.deepcopy(self.base_estimator)

    # This train val split for early stopping
    def _train_val_split(self, X, y):
        # Here we will depend on indices not data
        # get number of samples
        m = X.shape[0]

        # Seed based on random_state value to ensure it is the same when we run the code again
        rng = np.random.RandomState(self.random_state)

        # Get the number of samples as np array with evenly spaced values within a specified interval
        idx = np.arange(m)

        # Shuffle array 
        # We shuffle indices itself rather than X beacause it is more memory-efficient
        rng.shuffle(idx)
        
        # Portion of validation set to be used in Loss functions
        n_val = int(np.floor(self.validation_fraction * m))

        # if the portions is less than one make it at least one
        n_val = max(1, n_val)

        # Filter orginal train dataset by the validation indexes
        # So retrieve the first 0:n_val as validation and the rest as train n_val:end
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

        # If after split the train size is 0 then no Early stopping happen
        if train_idx.size == 0:
            # If too small dataset, fallback: no validation
            return X, y, None, None

        # Here we return indices of training and valdiation sets    
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    
# Gradient Boost Classifier class
class GradientBoostClassifier(GradientBoostBase):

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters       
    def __init__(self, estimator=None, n_estimators=100, learning_rate=0.1, 
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, tol=1e-6, random_state=None):

        # Gradient boost depends on Decision Tree Regressor as we pass numbers (residuals) and we get numbers (average)
        # Handle the error of estimator
        # If estimator is None or not equal decision tree classifier object or max_depth is not 1
        # is_instance is used to check object type 
        if estimator is None or not isinstance(estimator, DecisionTreeClassifier):
            # Here is the correct learner
            base_estim_to_use = DecisionTreeClassifier(max_depth=3)
        else:
            base_estim_to_use = estimator

        super().__init__(base_estim_to_use, n_estimators, learning_rate,
                         early_stopping, validation_fraction, n_iter_no_change, tol, random_state)

    # Helper functions
    # Case of binary classification
    # Sigmoid funtion to convert scoreF to P and in predictions to change final scoreF to P then to label
    def _sigmoid(self, z):
         # We add clip to make sure that z won't reach infinity or negative infinity for stability
         z = np.clip(z, -35, 35)
         return 1.0 / (1.0 + np.exp(-z))
    
    # Case of multi classification
    # Softmax funtion to convert scoreF to P
    def _softmax(self, F):
        # We convert all logits (log odds) (scoreF) to probabilites
        # We subtracted from max for stability trick
        F = F - np.max(F, axis=1, keepdims=True)

        # Then convert it to exp
        expF = np.exp(F)

        # Add stability of 1e-9 to make sure we won't devide by zero
        # Then devide it by sum of all learners highest F
        return expF / (np.sum(expF, axis=1, keepdims=True) + 1e-9)
    
    # Case of early stopping
    # We apply logloss for classification
    # So logloss for binary and logloss for multi
    def _log_loss_binary(self, y01, p1):
        # We add clip to make sure that p1 has limits using stability value
        # y01 refers to y = 0  and y = 1 and p1 to refer to probability of y = 1
        p1 = np.clip(p1, 1e-9, 1.0 - 1e-9)
        
        # The logistic formula is -1/n sum(y * log (p) + (1 - y) * log (1 - p)) to make sure we cover the abscene and the presence of the y
        # Logloss function
        return -np.mean(y01 * np.log(p1) + (1 - y01) * np.log(1 - p1))

    def _log_loss_multiclass(self, Y_onehot, P):
        # We add clip to make sure that p1 has limits using stability value
        P = np.clip(P, 1e-9, 1.0)

        # The logistic formula is -1/n sum(y * log (p))
        # Cross entorpy log loss function
        # Logloss function
        return -np.mean(np.sum(Y_onehot * np.log(P), axis=1))

    # Fit
    def fit(self, X, y):
        # Use numpy array to point to the data itself for memory effieciency
        X = np.asarray(X)
        y = np.asarray(y)

        # Encode classes to numbers (label encoder)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        # Get the number of classess
        self.n_classes_ = len(self.classes_)
        
        # Early stopping split if it enabled
        if self.early_stopping:
            # Once we apply the early stoppin then we pass x and classess to be splitted 
            # The function is for indices level and it returns data
            X_train, y_train, X_val, y_val = self._train_val_split(X, y_enc)
        else:
            # No validation set
            X_train, y_train, X_val, y_val = X, y_enc, None, None

        # For early stopping    
        self.estimators_ = []
        self.train_loss_ = []
        self.val_loss_ = []
        self.best_iteration_ = None
        
        # Fit binary classes and apply early stop
        if self.n_classes_ == 2:
            # Step 1
            # y_train is {0,1}
            # The probability that class y = 1 happen
            p0 = np.mean(y_train)

            # For stability
            p0 = np.clip(p0, 1e-12, 1.0 - 1e-12)

            # F0 = log(p0/(1-p0))
            # Log odds which will be F0
            # The most naive prediction
            self.init_ = np.log(p0 / (1.0 - p0))

            # Current scores on train in array which will be init_
            # Every time we have f0
            # Create array with the same shape of train rows to store init_ and the dtype is float
            # I make like column in table equivelant to each sample that is why we have the same shape
            F_train = np.full(X_train.shape[0], self.init_, dtype=float)

            # Convert F --> P 
            # I make like column in table equivelant to each sample that is why we have the same shape
            p_train = self._sigmoid(F_train)

            # For validation tracking
            if X_val is not None:
                # Create array with the same shape of train rows to store init_
                # I make like column in table equivelant to each sample that is why we have the same shape
                F_val = np.full(X_val.shape[0], self.init_, dtype=float)
                # Convert F --> P 
                # I make like column in table equivelant to each sample that is why we have the same shape
                p_val = self._sigmoid(F_val)
            
            # For loss purposes we have the best validation ever and no_improve counter which will be later
            best_val = np.inf
            no_improve = 0
            
            # Loop over leaners
            for learner in range(self.n_estimators):
                # Calculate residuals (negative gradient): r = y - p
                # Here we assign .astype to make sure flaot - float because p will be float [0:1]
                residuals = y_train.astype(float) - p_train

                # Create clone of learner to fit the data and update the weights
                estimator = self._clone_estimator()
                estimator.fit(X_train, residuals)

                # Update scores F 
                # Fnew = Fold + learning_rate * leaf_value (average) which will be the ouput of the decision tree regressor (predict)
                F_train += self.learning_rate * estimator.predict(X_train)
                # Convert F to P
                p_train = self._sigmoid(F_train)

                # Store estimator
                self.estimators_.append(estimator)

                # Track losses
                tr_loss = self._log_loss_binary(y_train, p_train)

                # Store losses
                self.train_loss_.append(tr_loss)

                if X_val is not None:
                    # All we did on train we did on evluate to compare
                    F_val += self.learning_rate * estimator.predict(X_val)
                    p_val = self._sigmoid(F_val)
                    va_loss = self._log_loss_binary(y_val, p_val)
                    self.val_loss_.append(va_loss)

                    # Early stopping check
                    if va_loss + self.tol < best_val:
                        # First time the best_val will be va_loss because it is equal infinity and for sure the condition gonna happen
                        best_val = va_loss
                        
                        # Best iteration is m + 1 cause m starts at zero
                        self.best_iteration_ = m + 1 
                        
                        # no_improve is the counter of n_iter_no_change (patience)
                        # Track the iteration with no change
                        # So logic to make it zero becasue it is the best firt time
                        no_improve = 0
                    else:
                        # We give chances
                        no_improve += 1

                        # When pateince run out
                        if no_improve >= self.n_iter_no_change:
                            # Roll back to best iteration
                            if self.best_iteration_ is not None:
                                # We stop and we will update the whole lists
                                self.estimators_ = self.estimators_[:self.best_iteration_]
                                self.val_loss_ = self.val_loss_[:self.best_iteration_]
                                self.train_loss_ = self.train_loss_[:self.best_iteration_]
                            break

            return self

        # For multi classification
        # Softmax method
        K = self.n_classes_

        # One-hot for training labels
        # Here we seperate work per class
        # We shape new y_train as matrix of zeros (m, k)
        Y_train = np.zeros((X_train.shape[0], K), dtype=float)

        # Fill all cells with dummy var = 1.0
        Y_train[np.arange(X_train.shape[0]), y_train] = 1.0

        # Initialize scores with log priors: F0_k = log(pi_k)
        priors = np.mean(Y_train, axis=0)
        priors = np.clip(priors, 1e-12, 1.0)
        self.init_ = np.log(priors)

        # Current scores on train: (n, K)
        F_train = np.tile(self.init_, (X_train.shape[0], 1))
        P_train = self._softmax(F_train)

        # Validation
        if X_val is not None:
            Y_val = np.zeros((X_val.shape[0], K), dtype=float)
            Y_val[np.arange(X_val.shape[0]), y_val] = 1.0
            F_val = np.tile(self.init_, (X_val.shape[0], 1))
            P_val = self._softmax(F_val)

        best_val = np.inf
        no_improve = 0

        for m in range(self.n_estimators):
            # Residuals per class: R = Y - P
            R = Y_train - P_train

            trees_this_iter = []
            for k in range(K):
                estimator = self._clone_estimator()
                estimator.fit(X_train, R[:, k])
                trees_this_iter.append(estimator)

            # Update scores
            for k in range(K):
                F_train[:, k] += self.learning_rate * trees_this_iter[k].predict(X_train)

            P_train = self._softmax(F_train)

            # Store this iteration's trees
            self.estimators_.append(trees_this_iter)

            # Track losses
            tr_loss = self._log_loss_multiclass(Y_train, P_train)
            self.train_loss_.append(tr_loss)

            if X_val is not None:
                for k in range(K):
                    F_val[:, k] += self.learning_rate * trees_this_iter[k].predict(X_val)
                P_val = self._softmax(F_val)
                va_loss = self._log_loss_multiclass(Y_val, P_val)
                self.val_loss_.append(va_loss)

                # Early stopping
                if va_loss + self.tol < best_val:
                    best_val = va_loss
                    self.best_iteration_ = m + 1
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.n_iter_no_change:
                        if self.best_iteration_ is not None:
                            self.estimators_ = self.estimators_[:self.best_iteration_]
                            self.val_loss_ = self.val_loss_[:self.best_iteration_]
                            self.train_loss_ = self.train_loss_[:self.best_iteration_]
                        break

        return self

    # Predict probabilities
    def predict_proba(self, X):
        X = np.asarray(X)

        # Binary
        if self.n_classes_ == 2:
            F = np.full(X.shape[0], self.init_, dtype=float)
            for estimator in self.estimators_:
                F += self.learning_rate * estimator.predict(X)
            p1 = self._sigmoid(F)
            return np.column_stack([1.0 - p1, p1])

        # Multi class
        K = self.n_classes_
        F = np.tile(self.init_, (X.shape[0], 1))
        for trees_this_iter in self.estimators_:
            for k in range(K):
                F[:, k] += self.learning_rate * trees_this_iter[k].predict(X)
        return self._softmax(F)

    # Predict
    def predict(self, X):
        probas = self.predict_proba(X)
        pred_idx = np.argmax(probas, axis=1)
        return self.classes_[pred_idx]

    # Score
    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)

class GradientBoostRegressor(GradientBoostBase):
    
    # Initialization
    def __init__(self, estimator=None, n_estimators=100, learning_rate=0.1,
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
                 tol=1e-6, random_state=None):
        
        if estimator is None or not isinstance(estimator, DecisionTreeRegressor):
            base_estim_to_use = DecisionTreeRegressor(max_depth=3)
        else:
            base_estim_to_use = estimator
        
        super().__init__(base_estim_to_use, n_estimators, learning_rate,
                         early_stopping, validation_fraction, n_iter_no_change,
                         tol, random_state)
        
        self.init_ = None

    def _mse(self, y, y_pred):
        return np.mean((y - y_pred) ** 2)

    # Fit
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).astype(float)

        # Early stopping split (optional)
        if self.early_stopping:
            X_train, y_train, X_val, y_val = self._train_val_split(X, y)
        else:
            X_train, y_train, X_val, y_val = X, y, None, None

        # F0 = mean(y) for squared error
        self.init_ = np.mean(y_train)

        # Current prediction
        y_pred_train = np.full(X_train.shape[0], self.init_, dtype=float)

        if X_val is not None:
            y_pred_val = np.full(X_val.shape[0], self.init_, dtype=float)

        self.estimators_ = []
        self.train_loss_ = []
        self.val_loss_ = []
        self.best_iteration_ = None

        best_val = np.inf
        no_improve = 0

        for m in range(self.n_estimators):
            # Residuals = y - prediction
            residuals = y_train - y_pred_train

            estimator = self._clone_estimator()
            estimator.fit(X_train, residuals)

            # Update predictions
            y_pred_train += self.learning_rate * estimator.predict(X_train)

            self.estimators_.append(estimator)

            # Track losses
            tr_loss = self._mse(y_train, y_pred_train)
            self.train_loss_.append(tr_loss)

            if X_val is not None:
                y_pred_val += self.learning_rate * estimator.predict(X_val)
                va_loss = self._mse(y_val, y_pred_val)
                self.val_loss_.append(va_loss)

                # Early stopping
                if va_loss + self.tol < best_val:
                    best_val = va_loss
                    self.best_iteration_ = m + 1
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.n_iter_no_change:
                        if self.best_iteration_ is not None:
                            self.estimators_ = self.estimators_[:self.best_iteration_]
                            self.val_loss_ = self.val_loss_[:self.best_iteration_]
                            self.train_loss_ = self.train_loss_[:self.best_iteration_]
                        break

        return self

    # Predict
    def predict(self, X):
        X = np.asarray(X)
        y_pred = np.full(X.shape[0], self.init_, dtype=float)
        for estimator in self.estimators_:
            y_pred += self.learning_rate * estimator.predict(X)
        return y_pred

    # Score
    def score(self, X, y):
        y = np.asarray(y).astype(float)
        return np.mean((self.predict(X) - y) ** 2)  # MSE
