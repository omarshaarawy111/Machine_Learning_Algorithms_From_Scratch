# Known that exteme gradient boost is boosting with greedy tress as base estimator 
# So we need to get decision tress
# We have multiple of weak learners to have a strong learner at the end working sequentially and each weak learner is trained on the previous one errors
from Libraries.DecisionTree import *
import numpy as np
import copy

# Here is modified version of Decsion Tree (Greedy Tree) and Gradient Boost (Extreme Gradient Boost)


# XGBoost Tree Regressor class (Greedy Tree)
# All we need just regressor as we work with optimal leaf weights not samples
# This is the greedy regression tree used inside Extreme Gradient Boosting
class XGBTreeRegressor(BaseDecisionTree):

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features=None, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, min_child_weight=1.0):
        # Here we intialize tree properties there is mandatory and optional propeties based on use case
        # Note that criterion here is not gini/entropy/mse, it is XGBoost gain based on extreme gradients and hessians
        # Here we don't work with min_impurity_decreace nor ccp_alpha so we set them to zero
        super().__init__(criterion="xgb",
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         max_features=max_features,
                         min_impurity_decrease=0.0,
                         ccp_alpha=0.0)

        # Regularization on leaf weights (L2)
        self.reg_lambda = reg_lambda
        # Regularization on leaf weights (L1)
        # This is used less than L2
        self.reg_alpha = reg_alpha
        # Penalization term on each split 
        # This is regularization but for tree structure
        self.gamma = gamma
        # Minimum hessian sum per leaf (to avoid unreliable splits)
        self.min_child_weight = min_child_weight

        # Root node
        self.root = None

    # Helper functions
    # L1 regularization for reg_alpha
    def _soft_threshold(self, G):
        # We adjust the gain based on these conditions
        # We adjust as the existence of alpha make us shrink the gain wether it is for root, left or right
        # If reg_alpha is zero then no shrink happens
        a = self.reg_alpha
        if a <= 0:
            return G

        # Shrink happen
        if G > a:
            return G - a
        if G < -a:
            return G + a
        return 0.0

    # Similarity score function
    # This is used to calculate the gain of split
    def _similarity(self, G, H):
        # This score is term of calcualting the gain
        # We add reg_lambda to avoid overfitting 
        # Add stability to avoid zero devision
        G_adj = self._soft_threshold(G)
        return (G_adj ** 2) / (H + self.reg_lambda + 1e-9)

    # Gain function
    # This is the criterion to choose the best feature and the best threshold
    def _gain(self, G, H, GL, HL, GR, HR):
        # gamma penalizes each split to avoid overfitting in tree structure
        return 0.5 * (self._similarity(GL, HL) + self._similarity(GR, HR) - self._similarity(G, H)) - self.gamma

    # Here the value leaf is the optimal weight of this leaf
    # This is not average like gradient boost when calling decision tree
    def _leaf_value(self, g, h):
        # Summation of extreme gradients and hessians inside the leaf
        G = np.sum(g)
        H = np.sum(h)

        # Apply L1 shrink if it is enabled
        G_adj = self._soft_threshold(G)

        # Optimal leaf weight as w = -G/(H + lambda)
        # Add stability to avoid zero devision
        w = - G_adj / (H + self.reg_lambda + 1e-9)
        return w, None

    # Best split function 
    # Here it is the same skelton as decsion tree but with different methodology
    def _best_split(self, X, g, h, features=None):
        # Here we need to have the best split by having samples and features
        # This function to get the best feature with all related properties and later in _build_tree we will assign samples actually
        # My target is to reach the best gain based on extreme gradients and 
        # here the same concept of applying internal and global splits

        # Intialize the best gain first
        # - infinity is the safest option always
        best_gain = -np.inf

        # My target is to retrieve best feature with best threshold
        split_feature, split_threshold = None, None

        # Number of features
        # Here if we apply portion of features of not
        n_features = X.shape[1]
        if features is None:
            features = np.arange(n_features)

        # Parent stats
        G = np.sum(g)
        H = np.sum(h)

        # Gains score list
        gains_score = []

        # Loop over features
        for feature in features:
            # First we decalre the thresholds for internal splits
            # Thresholds come from features itself so coming from x
            # As we know acadeemically that continuos values (num=numerical features) we take average but here we take the midpoint between each pairs
            # And for discrete values (Categorical features) we take the unique labels
            # Preprocessing for data happen and eventually we got numbers wether it is categorical features or numerical one
            # So in both cases sickit learn API take the midpoint as midpoint covers all cases and get less number of thresholds
            # We need the all samples of X within this feature
            # We got sorted non repeatable array (Core)
            values = np.unique(X[:, feature])

            # If the feature has only one unique value then no split possible
            if values.shape[0] < 2:
                continue

            # Midpoints fetched and stored in array
            # Midpoint = (sum of each pair) / 2 
            # Start at 0,1 index then 1,2 and so on
            thresholds = (values[:-1] + values[1:]) / 2

            # Loop over thresholds to see the highest one
            for threshold in thresholds:
                # Samples role
                # We have left and right children
                # Left samples gonna be less or equal threshold
                # Right samples gonna be other wise
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                # If one child has no samples then immediately stop the current threshold
                # Sum will be applied to samples means when no sapmples exsist
                # sum() == 0
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                # Child stats
                GL = np.sum(g[left_mask])
                HL = np.sum(h[left_mask])
                # Of course the rest will be in the right
                GR = G - GL
                HR = H - HL

                # Stop case
                # Stopping criteria (pre prunning)
                # min_child_weight is minimum hessian sum in each leaf
                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue

                # Gain score
                gain = self._gain(G, H, GL, HL, GR, HR)

                # Here instead
                if gain > best_gain:
                    # By deafult first time we pass the condition as min value of gain is 0 which is bigger than -1
                    best_gain = gain
                    split_feature = feature
                    split_threshold = threshold

        # Add the best gain of the current feature to the score list then later we choose the highest score from it
            gains_score.append([best_gain, split_feature, split_threshold])
            
        # Get the highest score
        max_gain = max(gains_score, key = lambda x:x[0])

        # Return the all properties of the best features
        # Return best_gain we will use it as stopping criteria for min impurity decrease
        # Return split_feature to work with and build tree
        # Return split_threshold to assign features in tree
        return max_gain[0], max_gain[1], max_gain[2]

    # For building our tree
    # This is the most important function with the help of other helper functions
    def _build_tree(self, X, g, h, depth):
        # Ofcourse we need to retrieve smaples and fatures count
        n_samples, n_features = X.shape

        # Stop case
        # Stopping criteria (pre prunning)
        # Preprunning within node
        # The cases of inaccurate number of samples in the node or exceed the max depth is hyper paramter stopping cirteria
        if n_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value, leaf_dist = self._leaf_value(g, h)
            return Node(value=leaf_value, value_dist=leaf_dist)

        # This part is related to option of choosing portion of features as here we don't want to work with all features
        # Note that it is just hyper parametr (optional)
        features_idx = np.arange(n_features)
        if self.max_features is not None:
            # Make it random number from predefined range 
            # Replace  = False so we won't duplicate feature
            features_idx = np.random.choice(
                n_features, self.max_features, replace=False
                )

        # Step 1 ---> find the best feature with all properties
        best_gain, split_feature, split_threshold = self._best_split(X, g, h, features_idx)

        # Stop case
        # Stopping criteria (pre prunning)
        # Preprunning within leaves
        # If no split feature found OR gain is not positive then leaf node
        # Here the prepruning of gamma happen as we envolve gamma in our calcualtion then if it is bigger than or equal the gain 
        # So logically best gain gonna beneth zero
        if split_feature is None or best_gain <= 0:
            leaf_value, leaf_dist = self._leaf_value(g, h)
            return Node(value=leaf_value, value_dist=leaf_dist)

        # Step 2 ---> assign samples in tree
        # Now we work with the best split (best feature + best threshold) after test and assign samples
        left_mask = X[:, split_feature] <= split_threshold
        right_mask = ~left_mask

        # Stop case
        # Stopping criteria (pre prunning)
        # The cases of inaccurate number of samples in the leaf (after split)
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            leaf_value, leaf_dist = self._leaf_value(g, h)
            return Node(value=leaf_value, value_dist=leaf_dist)

        # Recusive case
        left = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth + 1)

        # We return Node bacause we need all information about tree so we can move through it in prediction knowing conditions
        return Node(split_feature, split_threshold, left, right)

    # Fit
    def fit(self, X, g, h):
        # Use numpy array (asarray) to point to the data itself for memory efficiency
        X = np.asarray(X)
        g = np.asarray(g).astype(float)
        h = np.asarray(h).astype(float)

        # The fit is building our tree and save it in root as symbol of our journey starting with pre prunning hyper paramerts
        self.root = self._build_tree(X, g, h, 0)

    # Predict
    # It returns leaf weights (numbers)
    def predict(self, X):
        return super().predict(X)

# We work with the base class then inheritence happen to build the classifier or regressor
class XGBoostBase:

    # Initialization
    def __init__(self, base_estimator, n_estimators=100, learning_rate=0.1,
                 early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, tol=1e-6, random_state=None):
        # Later we will decide base estimator is greedy regression tree
        # Stopping condition will be the number of weak learners or enable early stopping (error has no more improvement after k iterations)
        self.base_estimator = base_estimator
        # Others is the same default paramerts of API
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

        # Early stopping parameters
        self.early_stopping = early_stopping
        # Percentage of data to be vlaidation set
        self.validation_fraction = validation_fraction
        # The number of times (patience) till no change in erorr happen then early stop comes
        self.n_iter_no_change = n_iter_no_change
        # The min tolerance which means the min improvement and it is very close to zero
        self.tol = tol

        # Storage of leaners
        self.estimators_ = []
        # For early stopping
        # Train loss and validation loss array
        self.train_loss_ = []
        self.val_loss_ = []
        # The best number of learners before we early stop
        self.best_iteration_ = None

    # Here we create just abstract so after inheritence each of classification and regression has its own methodoly of aggregation for final prediction
    def fit(self, X, y):
        raise NotImplementedError

    # Helper function
    def _clone_estimator(self):
        # Create clone of learner to fit the data
        return copy.deepcopy(self.base_estimator)

    # Train val split function
    # This train val split for early stopping
    def _train_val_split(self, X, y):
        # Here we will depend on indices not data
        # get number of samples
        m = X.shape[0]

        # Seed based on random_state value to ensure it is the same when we run the code again
        rng = np.random.RandomState(self.random_state)

        # Get the number of samples as np array with evenly spaced values within a specified interval
        # It is just indexing (creating indices)
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


# XGBoost Classifier class
class XGBoostClassifier(XGBoostBase):

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters
    def __init__(self,estimator=None, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, max_features=None, 
                 reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, min_child_weight=1.0,
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
                 tol=1e-6, random_state=None):

        # Extreme gradient boost classification depends on XGBTreeRegressor as base learner
        # Because we pass extreme gradients and hessians as numbers and we get numbers (leaf weights) as predictions
        # Handle the error of estimator
        # If estimator is None or not equal XGBTreeRegressor object then use default
        if estimator is None or not isinstance(estimator, XGBTreeRegressor):
            base_estim_to_use = XGBTreeRegressor(max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            gamma=gamma,
            min_child_weight=min_child_weight)

        else:
            base_estim_to_use = estimator

        super().__init__(base_estim_to_use, n_estimators, learning_rate,
                         early_stopping, validation_fraction, n_iter_no_change,
                         tol, random_state)

        # Init_ will be the naive prediction in classification
        self.init_ = None
        # Store original labels
        self.classes_ = None
        # Store number of classes
        self.n_classes_ = None

    # Helper functions
    # Functions that help to convert scoreZ to P
    # Case of binary classification
    # Sigmoid funtion to convert scoreZ to P and in predictions to change final scoreZ to P then to label
    def _sigmoid(self, z):
        # We add clip to make sure that z won't reach infinity or negative infinity for stability
        z = np.clip(z, -35, 35)
        return 1.0 / (1.0 + np.exp(-z))

    # Case of multi classification
    # Softmax funtion to convert scoreZ to P
    def _softmax(self, Z):
        # We convert all logits (log odds) (scoreZ) to probabilites
        # We subtracted from max for stability trick
        Z = Z - np.max(Z, axis=1, keepdims=True)

        # Then convert it to exp
        expZ = np.exp(Z)

        # This is normalization to make sure sum of all probabilities after conversion from Z is = 1
        return expZ / (np.sum(expZ, axis=1, keepdims=True) + 1e-9)

    # Case of early stopping
    # We apply logloss for classification
    # So logloss for binary and logloss for multi
    # Logloss function
    def _log_loss_binary(self, y01, p1):
        # We add clip to make sure that p1 has limits using stability value
        p1 = np.clip(p1, 1e-9, 1.0 - 1e-9)
        return -np.mean(y01 * np.log(p1) + (1 - y01) * np.log(1 - p1))

    # Softmax cross entropy loss
    def _log_loss_multiclass(self, Y_onehot, P):
        # We add clip to make sure that p has limits using stability value
        P = np.clip(P, 1e-9, 1.0)
        return -np.mean(np.sum(Y_onehot * np.log(P), axis=1))

    # Fit
    def fit(self, X, y):
        # Use numpy array (asarray) to point to the data itself for memory effieciency
        X = np.asarray(X)
        y = np.asarray(y)

        # Encode classes to numbers (label encoder)
        self.classes_, y_enc = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)

        # Early stopping split if it enabled
        if self.early_stopping:
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
            # Intialize for both train and validation
            # y_train is {0,1}
            # The probability that class y = 1 happen
            p0 = np.mean(y_train)
            p0 = np.clip(p0, 1e-12, 1.0 - 1e-12)

            # Z0 = log(p0/(1-p0))
            # Log odds which will be Z0
            # The most naive prediction
            self.init_ = np.log(p0 / (1.0 - p0))

            # Current scores on train (Z and P) for current learner
            Z_train = np.full(X_train.shape[0], self.init_, dtype=float)
            p_train = self._sigmoid(Z_train)

            # For validation tracking
            if X_val is not None:
                # Intialize validation set
                # Create array with the same shape of train rows to store init_ for predictions of validation
                # We need of course F and p in our loss
                # The purpose is loss scale
                Z_val = np.full(X_val.shape[0], self.init_, dtype=float)
                p_val = self._sigmoid(Z_val)

            best_val = np.inf
            no_improve = 0

            # Loop over learners
            for m in range(self.n_estimators):
                # Step 2
                # Calculate extreme gradients and hessians for logistic loss:
                # g = p - y
                # h = p(1 - p)
                g = p_train - y_train.astype(float)
                h = p_train * (1.0 - p_train)

                # Step 3
                # Create clone of learner to fit the data
                # It will be XGBTreeRegressor as we pass extreme gradients and hessians as numbers
                estimator = self._clone_estimator()
                estimator.fit(X_train, g, h)

                # Step 4
                # Update scores Z
                # Znew = Zold + learning_rate * leaf_weight
                Z_train += self.learning_rate * estimator.predict(X_train)
                p_train = self._sigmoid(Z_train)

                # Store estimator
                self.estimators_.append(estimator)

                # Step 5
                # Evaluation step
                # Track losses
                tr_loss = self._log_loss_binary(y_train, p_train)
                self.train_loss_.append(tr_loss)

                if X_val is not None:
                    # Here we need to precit
                    # It is concident that train fomrula looks like test 
                    # Here we don't build we just collect
                    # We don't duplicate training work
                    # But no fitting just predict
                    Z_val += self.learning_rate * estimator.predict(X_val)
                    p_val = self._sigmoid(Z_val)
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
                        # So logic to make it zero becasue it is the best and we record again
                        no_improve = 0

                    else:
                        # We give chances
                        no_improve += 1

                        # When pateince run out
                        if no_improve >= self.n_iter_no_change:
                            # Roll back to best iteration
                            if self.best_iteration_ is not None:
                                # We stop and we will update the whole lists
                                # Index at all other lists there
                                self.estimators_ = self.estimators_[:self.best_iteration_]
                                self.val_loss_ = self.val_loss_[:self.best_iteration_]
                                self.train_loss_ = self.train_loss_[:self.best_iteration_]
                            break

            return self

        # For multi classification
        # Softmax method
        K = self.n_classes_

        # One-hot for training labels
        Y_train = np.zeros((X_train.shape[0], K), dtype=float)
        Y_train[np.arange(X_train.shape[0]), y_train] = 1.0

        # Step 1
        # Initialize scores with log priors: Z0_k = log(pi_k)
        priors = np.mean(Y_train, axis=0)
        priors = np.clip(priors, 1e-12, 1.0)
        self.init_ = np.log(priors)

        # Current scores on train: (n, K)
        Z_train = np.tile(self.init_, (X_train.shape[0], 1))
        P_train = self._softmax(Z_train)

        # For validation tracking
        # Intialize validation set
        # We need of course F and p in our loss
        # Create array with the same shape of train rows to store init_ for predictions of validation
        # The purpose is loss scale 
        if X_val is not None:
            # Intialize validation set
            # We need of course F and p in our loss
            # Create array with the same shape of train rows to store init_ for predictions of validation
            # The purpose is loss scale
            Y_val = np.zeros((X_val.shape[0], K), dtype=float)
            Y_val[np.arange(X_val.shape[0]), y_val] = 1.0
            # Fill all cells with dummy var  = 1
            Z_val = np.tile(self.init_, (X_val.shape[0], 1))
            P_val = self._softmax(Z_val)

        best_val = np.inf
        no_improve = 0

        # Loop over learners
        for m in range(self.n_estimators):
            # Step 2
            # Calculate extreme gradients and hessians per class:
            # g_k = p_k - y_k
            # h_k = p_k(1 - p_k)
            G_mat = P_train - Y_train
            H_mat = P_train * (1.0 - P_train)

            # Step 3
            # We create trees = number of classes per iteration
            trees_this_iter = []
            for k in range(K):
                estimator = self._clone_estimator()
                estimator.fit(X_train, G_mat[:, k], H_mat[:, k])
                trees_this_iter.append(estimator)

            # Step 4
            # Update scores Z
            for k in range(K):
                Z_train[:, k] += self.learning_rate * trees_this_iter[k].predict(X_train)

            # Step 5
            # Convert Z to P to start new learner
            P_train = self._softmax(Z_train)

            # Store this iteration's trees
            self.estimators_.append(trees_this_iter)

            # Step 6
            # Evaluation step
            tr_loss = self._log_loss_multiclass(Y_train, P_train)
            self.train_loss_.append(tr_loss)

            if X_val is not None:
                # Here we need to precit
                # It is concident that train fomrula looks like test 
                # Here we don't build we just collect
                # We don't duplicate training work
                # But no fitting just predict
                # Make sure we work on k trees per iteration so we predict Zks
                for k in range(K):
                    Z_val[:, k] += self.learning_rate * trees_this_iter[k].predict(X_val)

                # Convert Z to P       
                P_val = self._softmax(Z_val)
                # Evaluation
                va_loss = self._log_loss_multiclass(Y_val, P_val)
                self.val_loss_.append(va_loss)
             
                # Early stopping
                # The same as binary
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
            # Get all scores intialization
            Z = np.full(X.shape[0], self.init_, dtype=float)

            # Loop over learners
            for estimator in self.estimators_:
                Z += self.learning_rate * estimator.predict(X)

            # Get P
            p1 = self._sigmoid(Z)

            # In code binary classification has also one hot encoder so we have two columns 
            # For y = 0 and y = 1 and we combine them to get 2 columns so we return P0 and P1
            return np.column_stack([1.0 - p1, p1])

        # Multi class
        K = self.n_classes_
        Z = np.tile(self.init_, (X.shape[0], 1))

        # Loop over learners
        for trees_this_iter in self.estimators_:
            # Loop over trees
            for k in range(K):
                Z[:, k] += self.learning_rate * trees_this_iter[k].predict(X)

        # Get P
        return self._softmax(Z)

    # Predict
    def predict(self, X):
        # We get probabilities first
        probas = self.predict_proba(X)

        # We get the max probability per sample
        pred_idx = np.argmax(probas, axis=1)

        # Return the label
        return self.classes_[pred_idx]

    # Score
    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)
    
# XGBoost Regressor class
class XGBoostRegressor(XGBoostBase):

    # Intialization
    # We pass known numbers of paramters to parent class and also at the same time get known numbers of paramters
    def __init__(self,estimator=None, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, min_samples_leaf=1, max_features=None, 
                 reg_lambda=1.0, reg_alpha=0.0, gamma=0.0, min_child_weight=1.0,
                 early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
                 tol=1e-6, random_state=None):

        # Extreme gradient boost classification depends on XGBTreeRegressor as base learner
        # Because we pass extreme gradients and hessians as numbers and we get numbers (leaf weights) as predictions
        # Handle the error of estimator
        # If estimator is None or not equal XGBTreeRegressor object then use default
        if estimator is None or not isinstance(estimator, XGBTreeRegressor):
            base_estim_to_use = XGBTreeRegressor(max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            gamma=gamma,
            min_child_weight=min_child_weight)

        super().__init__(base_estim_to_use, n_estimators, learning_rate,
                         early_stopping, validation_fraction, n_iter_no_change,
                         tol, random_state)

        # Init_ will be the naive prediction which is mean(y) in squared error regression
        self.init_ = None

    # Case of early stopping
    # We apply MSE for regression as loss function
    # MSE function
    def _mse(self, y, y_pred):
        # Mean Squared Error = mean((y - y_pred)^2)
        return np.mean((y - y_pred) ** 2)

    # Fit
    def fit(self, X, y):
        # Use numpy array (asarray) to point to the data itself for memory efficiency
        X = np.asarray(X)
        # Ensure y is float because regression and predictions are continuous numbers
        y = np.asarray(y).astype(float)

        # Early stopping split (optional)
        # If enabled then we split data into train and validation sets
        if self.early_stopping:
            X_train, y_train, X_val, y_val = self._train_val_split(X, y)
        else:
            # No validation set
            X_train, y_train, X_val, y_val = X, y, None, None

        # Step 1
        # Initialization for regression with squared error:
        # F0 = mean(y_train) which is the most naive prediction (constant baseline)
        self.init_ = np.mean(y_train)

        # Current prediction on train
        # Create array with the same shape of train rows to store init_ for predictions
        # I make like column in table equivalent to each sample that is why we have the same shape
        y_pred_train = np.full(X_train.shape[0], self.init_, dtype=float)

        # For validation tracking
        if X_val is not None:
            # Initialize validation predictions with the same init_
            # The purpose is loss scale (baseline) and cumulative prediction updates
            y_pred_val = np.full(X_val.shape[0], self.init_, dtype=float)

        # Reset storages for re-fit
        self.estimators_ = []
        self.train_loss_ = []
        self.val_loss_ = []
        self.best_iteration_ = None

        # For loss purposes we have the best validation ever and no_improve counter which will be later
        best_val = np.inf
        no_improve = 0

        # Loop over learners
        for m in range(self.n_estimators):
            # Step 2
            # Calculate extreme gradients and hessians for squared error:
            # g = y_pred - y
            # h = 1
            g = y_pred_train - y_train
            h = np.ones_like(g, dtype=float)

            # Step 3
            # Create clone of learner to fit the data
            # It will be XGBTreeRegressor as we pass extreme gradients and hessians as numbers
            estimator = self._clone_estimator()
            estimator.fit(X_train, g, h)

            # Step 4
            # Update predictions:
            # y_pred_new = y_pred_old + learning_rate * tree_prediction
            # Where tree_prediction is leaf weight for the leaf that each sample falls into
            y_pred_train += self.learning_rate * estimator.predict(X_train)

            # Store estimator
            self.estimators_.append(estimator)

            # Step 5
            # Evaluation step
            # Not like academic there is no evaluation before learner 1
            # Track training loss (MSE)
            tr_loss = self._mse(y_train, y_pred_train)
            self.train_loss_.append(tr_loss)

            if X_val is not None:
                # Here we need to predict
                # It is coincident that train formula looks like test
                # Here we don't build we just collect
                # We don't duplicate training work
                # But no fitting just predict
                y_pred_val += self.learning_rate * estimator.predict(X_val)

                # Evaluation on validation (MSE)
                va_loss = self._mse(y_val, y_pred_val)
                self.val_loss_.append(va_loss)

                # Early stopping check
                # The same as classifier:
                # If validation improves by at least tol then reset patience
                if va_loss + self.tol < best_val:
                    best_val = va_loss
                    # Best iteration is m + 1 cause m starts at zero
                    self.best_iteration_ = m + 1
                    # Reset counter because this is the best so far
                    no_improve = 0
                else:
                    # No improvement
                    no_improve += 1

                    # When patience runs out
                    if no_improve >= self.n_iter_no_change:
                        # Roll back to best iteration
                        if self.best_iteration_ is not None:
                            self.estimators_ = self.estimators_[:self.best_iteration_]
                            self.val_loss_ = self.val_loss_[:self.best_iteration_]
                            self.train_loss_ = self.train_loss_[:self.best_iteration_]
                        break

        return self

    # Predict
    def predict(self, X):
        # Convert X to numpy
        X = np.asarray(X)

        # Start from baseline init_
        # Create array with the same shape of X rows to store init_ for predictions
        y_pred = np.full(X.shape[0], self.init_, dtype=float)

        # Loop over learners and accumulate their contributions
        for estimator in self.estimators_:
            y_pred += self.learning_rate * estimator.predict(X)

        return y_pred

    # Score
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2) 