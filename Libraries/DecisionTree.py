import numpy as np
from collections import Counter

# Decision Tree class for Classification and Regression
# We start with the core class which is related to node 
class Node():

    # Initialization
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None, value_dist = None):
        # We need to intailize everything of node properties (features, threshold, left, right, value)
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 
        # For classifier predict_proba
        self.value_dist = value_dist
        
# We work with the base class then inheritence happen to build the classification or regression tree
class BaseDecisionTree():

    # Initialization
    def __init__(self, criterion, max_depth = None, min_samples_split = 2, min_samples_leaf = 1, max_features = None, min_impurity_decrease = 0.0, ccp_alpha = 0.0):
        # Here we intialize tree properties there is mandatory and optional propeties based on use case and also your prunning decision    
        # min_samples_split = 2, min_samples_leaf = 1 these are mandatory cause they are logically the min amount to split or work rather than it isn't logical
        # min_impurity_decrease = 0.0, ccp_alpha = 0.0 becasue the default is no pruning till you decide
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.root = None

    # Helper functions and will be called in build_tree later
    # Impurity functions
    # Categorical impurities
    # Gini
    def _gini(self, y, sample_weight):
        # We calculate probabilities based on the sum of weights for each class
        total_w = np.sum(sample_weight)
        if total_w == 0: return 0
        probs_sq = 0
        for c in np.unique(y):
            p = np.sum(sample_weight[y == c]) / total_w
            probs_sq += p ** 2
        return 1 - probs_sq
    
    # Numerical impurity
    # Entropy
    def _entropy(self, y, sample_weight):
        # We calculate probabilities based on the sum of weights for each class
        total_w = np.sum(sample_weight)
        if total_w == 0: return 0
        ent = 0
        for c in np.unique(y):
            p = np.sum(sample_weight[y == c]) / total_w
            ent -= p * np.log2(p + 1e-9)
        return ent
    
    def _mse(self, y, sample_weight):
        # Get the weighted MSE
        if np.sum(sample_weight) == 0: return 0
        weighted_mean = np.average(y, weights=sample_weight)
        return np.average((y - weighted_mean) ** 2, weights=sample_weight)
    
    # Accumlative function to make all criterion in one place
    def _impurity(self, y, sample_weight):
        if self.criterion == "gini":
            return self._gini(y, sample_weight)
        
        elif self.criterion == "entropy":
            return self._entropy(y, sample_weight)
        
        elif self.criterion == "mse":
            return self._mse(y, sample_weight)

    # Best split function
    def _best_split(self, X, y, sample_weight, features=None):
        # Here we need to have the best split by having samples and features
        # This function to get the best feature with all related properties and later in _build_tree we will assign samples actually
        # My target is to reach the best info gain or gini gain 
        # We have internal splits for every feature to choose the best gain 
        # Then compare it with others features to decide which is the best among all internal and features split to choose this feature
        # Intialize the best gain first
        best_gain = -1

        # My target is to retrieve best feature with best threshold
        split_feature, split_threshold = None, None

        # Calcualte the parent impurity so later we calcualte info gain or gini gain 
        parent_impurity = self._impurity(y, sample_weight)

        # Number of features
        n_features = X.shape[1]
        if features is None:
            features = np.arange(n_features)

        # Gains score list
        gains_score = []
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
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                # Target role to calculate gian info or gini info
                left_y, left_w = y[left_mask], sample_weight[left_mask]
                right_y, right_w = y[right_mask], sample_weight[right_mask]

                # total number of weights
                total_w = np.sum(sample_weight) 

                # Average weighted impurity
                weighted_average_impurity = (
                    np.sum(left_w) / total_w * self._impurity(left_y, left_w)
                    + np.sum(right_w) / total_w * self._impurity(right_y, right_w)
                )

                # Gain info = Gini info
                gain = parent_impurity - weighted_average_impurity

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

    # Check the purity function
    def _is_pure(self, y):
        # We could know by having only one label 
        # This fucntion is exclusive for the classification only and not for regression as we never reach the exact value
        return len(np.unique(y)) == 1

    # Here we create just abstract so after inheritence each of classification and regression has its own methodoly of leaf_value for prediction later
    def _leaf_value(self, y, sample_weight):
        raise NotImplementedError    

    # For building our tree 
    # This is the most important function with the help of other helper functions
    def _build_tree(self, X, y, sample_weight, depth):  
        # Ofcourse we need to retrieve smaples and fatures count
        n_samples, n_features = X.shape

        # Stop case
        # Stopping criteria (pre prunning)
        # Preprunning within node
        # The case of leaf node is automatic stopping criteria
        # The cases of inaccurate number of samples in the node or exceed the max depth is hyper paramter stopping cirteria
        if self._is_pure(y) or n_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value, leaf_dist = self._leaf_value(y, sample_weight)
            return Node(value=leaf_value, value_dist=leaf_dist)

        # This part is related to random forest as here we don't want to work with all features to smaller correlation between tress, higher variance and cut the dominance of one feature
        # Note that it is just hyper parametr
        # Indexing feature for future use
        features_idx = np.arange(n_features)
        # Make sure we adjust that parametr
        if self.max_features is not None:
            # Make it random number from predefined range 
            # Replace  = False so we won't duplicate feature
            features_idx = np.random.choice(
                n_features, self.max_features, replace=False
            )

        # Step 1 ---> find the best feature with all properties
        # Return the all properties of the best features from best_split function
        # Return best_gain we will use it as stopping criteria for min impurity decrease
        # Return split_feature to work with and build tree
        # Return split_threshold to assign features in tree
        best_gain, split_feature, split_threshold = self._best_split(X, y, sample_weight, features_idx)

        # Stop case
        # Stopping criteria (pre prunning)
        # Preprunning within leaves
        # The case of non split feature is automatic stopping criteria
        # The cases of min impurity decrease is hyper paramter stopping cirteria
        if split_feature is None or best_gain < self.min_impurity_decrease:
            leaf_value, leaf_dist = self._leaf_value(y, sample_weight)
            return Node(value=leaf_value, value_dist=leaf_dist)

        # Step 2 ---> assign samples in tree
        # Samples role
        # We have left and right children
        # Left samples gonna be less or equal threshold
        # Right samples gonna be other wise
        # Now we work with the best split (best feature + best threshold) after test and assign samples
        left_mask = X[:, split_feature] <= split_threshold
        right_mask = ~left_mask

        # Stop case
        # Stopping criteria (pre prunning)
        # The cases of inaccurate number of samples in the leaf (after split)
        # We can't gathering all stopping conditions at one place as each step has its own stopping criteria
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf :
            leaf_value, leaf_dist = self._leaf_value(y, sample_weight)
            return Node(value=leaf_value, value_dist=leaf_dist)

        # Recusive case
        left = self._build_tree(X[left_mask], y[left_mask], sample_weight[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], sample_weight[right_mask], depth + 1)

        # We return Node  bacause we need all information about tree so we can move through it in prediction knowing conditions
        return Node(split_feature, split_threshold, left, right)   

    # Post prunning function
    def _prune(self, node):
        # We pass the tree for the function to post prune
        if node.left is None or node.right is None:
            return self._impurity(np.array([node.value]), np.array([1.0])), 1

        left_err, left_leaves = self._prune(node.left)
        right_err, right_leaves = self._prune(node.right)

        subtree_error = left_err + right_err
        leaf_error = self._impurity(np.array([node.value]), np.array([1.0]))

        if leaf_error + self.ccp_alpha <= subtree_error:
            node.left = None
            node.right = None
            return leaf_error, 1

        return subtree_error, left_leaves + right_leaves

    # Fit
    def fit(self, X, y, sample_weight=None):
        # We initialize weights as ones if not provided for standard tree operation
        # This is very critical for using within boosting as we work with weights not counts 
        # So this will be enabled in the case of weights and versa
        if sample_weight is None:
            # We create ones as weights if there is no weights as work with ones is the same as work with counts
            sample_weight = np.ones(X.shape[0])

        # The fit is building our tree and save it in root as symbol of our journey starting with pre prunning hyper paramerts
        self.root = self._build_tree(X, y, sample_weight, 0)
        
        # The decision of post pruning based on hyper  cc_aplpha greater than zero
        if self.ccp_alpha > 0:
            self._prune(self.root)

    # Predict 
    def predict(self, X):
        # Each observation has the journey till we reach the suitable leaf node based on condtions
        def traverse(x, node):
            if node.value is not None:
                # Stop case
                # We reach leaf node
                return node.value
            
            # Recursive case
            # I keep moving till i reach the value of node which is the leaf node (end of journey)
            if x[node.feature] <= node.threshold:
                return traverse(x, node.left)
            return traverse(x, node.right)
        
        # We just loop on new observation one by one
        return np.array([traverse(x, self.root) for x in X])

    # Simple visual tree print function
    def print_tree(self, node=None, indent=""):
        # It is simple simulation of tree
        if node is None:
            # Get tree
            node = self.root
            
        if node.value is not None:
            # Stop case
            # For leaf node ---> it has value of course
            print(indent + "Leaf:", np.round(node.value, 2))
        else:
            # Recursive case
            # Here the 2nd importance of getting threshold for drawing
            print(indent + f"X[{node.feature}] <= {node.threshold.round(2)}")
            self.print_tree(node.left, indent + "  ")
            self.print_tree(node.right, indent + "  ")

# Decsion Tree Classifier class
class DecisionTreeClassifier(BaseDecisionTree):

    # Intialization
    # We pass unknown numbers of paramters to parent class and also at the same time get unknown numbers of paramters
    def __init__(self, criterion="gini", **kwargs):
        super().__init__(criterion=criterion, **kwargs)

    # Fit
    def fit(self, X, y, sample_weight=None):
        # Store classes information for predict_proba (API style)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Standard fit from parent
        super().fit(X, y, sample_weight)

    # Here the value leaf is the most weighted majority class
    def _leaf_value(self, y, sample_weight):
        # We retrieve class with the highest sum of weights
        weighted_counts = np.zeros(self.n_classes_)
        for i, c in enumerate(self.classes_):
            weighted_counts[i] = np.sum(sample_weight[y == c])
            
        value = self.classes_[np.argmax(weighted_counts)]

        # For classifier predict_proba we need weighted distribution of classes in leaf
        value_dist = weighted_counts / (np.sum(weighted_counts) + 1e-9)

        return value, value_dist

    # Predict probabilities
    # This function is ready to use in soft oting later
    def predict_proba(self, X):
        def traverse_prob(x, node):
            if node.value is not None:
                return node.value_dist  
            if x[node.feature] <= node.threshold:
                return traverse_prob(x, node.left)
            return traverse_prob(x, node.right)
        probas = []
        for x in X:
            probas.append(traverse_prob(x, self.root))
        return np.array(probas)  
    
    # Score
    def score(self, X, y):
        # Return accuracy score 
        return np.mean(self.predict(X) == y)


# Decsion Tree Regressor class
class DecisionTreeRegressor(BaseDecisionTree):

    # Intialization
    # We pass unknown numbers of paramters to parent class and also at the same time get unknown numbers of paramters
    def __init__(self, **kwargs):
        super().__init__(criterion="mse", **kwargs)

    # Here the value leaf is the weighted mean of y values
    def _leaf_value(self, y, sample_weight):
        return np.average(y, weights=sample_weight), None

    # Score
    def score(self, X, y):
        return np.mean((self.predict(X) - y) ** 2)