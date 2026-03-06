# This is unspuervised learning algorithm which means we don't have labels 
# We don't have classifcation or regression 
# Here we make fature selction so wether we recast our original dataset or keep it
import numpy as np

class PCA():

    # Intialization
    def __init__(self, n_components=None):
        # Here only one parameter from user which is n_components
        self.n_components = n_components

        # Paramters of fit phase
        self.components = None
        self.mean = None
        self.std = None
        self.explained_variance = None

    # Fit
    def fit(self, X):
        # Convert X to array
        # Always remember to convert to nparray
        X = np.array(X)

        # Step 1 --> standarization
        # Get mean first
        self.mean = np.mean(X, axis=0)

        # Get standard deviation
        self.std = np.std(X, axis=0)

        # Scale X 
        # Add stability to avoid zero devision
        # Standard scaler  = (x - mean) / std
        X_scaled = (X - self.mean) / (self.std + 1e-9)

        # Step 2 --> get covariance matrix
        # We pass x_scaled and it process the rest like getting mean of x_scaled and sum
        # rowvar is true by default to make each row is var and column as observation
        # So we disabled it and do versa
        cov_matrix = np.cov(X_scaled, rowvar=False)

        # Step 3 --> eigen decomposition
        # Get eigen vectos and eigen values
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigen values with its eigen vectors
        # -1 is for descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        # So eigen values here is samples vs e 
        # Academilcally we create feature vector as filter of eigen vectors but here no
        eigenvectors = eigenvectors[:,sorted_idx]

        # known that all exaplined varinace with all principal components will appear at this stage through eigenvalues
        self.explained_variance = eigenvalues

        # Step 4 
        # Create feature vector through compoments selection
        # Select the n components i have decided
        if self.n_components is not None:
              # Handle if paramter is in % case
              # Here we don't mean percentage of features 
              # We mean principle components which can succeed this percentage of explained varaince ratio
              if 0 < self.n_components < 1:
                # First we got total varaince from eigen values
                total_variance = np.sum(eigenvalues)
                # Get variance ratio for each principle component
                # Pc : ratio
                variance_ratio = eigenvalues / total_variance
                # Third make it acculative as we go further in out array we cover more area of desired %%!
                cumulative = np.cumsum(variance_ratio)
                # Go the index bigger than or equal the desired threshold and add one as indexes start with 0
                self.n_components = np.argmax(cumulative >= self.n_components) + 1
                
              # Get all samples with selected principle components
              eigenvectors = eigenvectors[:, :self.n_components]
              
        self.components = eigenvectors

    # Transform
    # Known that fit is collection info and transform is applied so where the application here
    # Application is recast our scaled data
    def transform(self, X):
        # Step 5 --> recast dataset with the rule Z = XW where X is scaled and W is the feature vector of sleected eigen vectors
        # Step 5.1 --> standarization
        # Work with training data paramerts (mean and standard deviation)
        # We never calcualte it again so that avoiding data leakage
        X_scaled = (X - self.mean) / (self.std + 1e-9)

        # Step 5.2 apply Z rule
        return np.dot(X_scaled, self.components)
    
    # Fit and transform 
    def fit_transform(self, X):
        # Here we have fit and transform sperated as fit_transform it is for first data epoch (training data)
        # But then i will assign test data to work with the same fitting of model and transform shouldnot got new calcualtion on old data (training) so that no data leakage happened
        # Call fit first then return transform
        self.fit(X)
        return self.transform(X)
    
    # Inverse transform
    # This fucntion make reverse calculations to get the original data again
    def inverse_transform(self, Z):
        # Here we pass Z which was Z = XW and get X_scaled again then the original data
        # Get X_scaled
        # So it will be Z.W^-1
        X_scaled = np.dot(Z, self.components.T)

        # After scale get the original data
        # Don't forget to add stability so that we avoid zero multiplication 
        X = X_scaled * (self.std + 1e-9) + self.mean
        return X