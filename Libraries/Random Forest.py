# Known that random forest is bagging + decision trees but with some edits 
# So we just inherit bagging + decision trees and add edits
# Known that random forest work with only one base estimator (Descision Tree) cloning it within bootsreapped data with replacement
from Libraries.Bagging import *

# Random Forest class for Classification and Regression
# We work with the base class then inheritence happen to build the classification or regression random forest
class RandomForestBase(BaggingBase):
