import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from MDLP import MDLP_Discretizer

def main():

    ######### USE-CASE EXAMPLE #############

    #read dataset
    dataset = datasets.load_iris()
    X, y = dataset['data'], dataset['target']
    feature_names, class_names = dataset['feature_names'], dataset['target_names']
    numeric_features = np.arange(X.shape[1])  # all fetures in this dataset are numeric. These will be discretized

    #Split between training and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    #Initialize discretizer object and fit to training data
    discretizer = MDLP_Discretizer(features=numeric_features)
    discretizer.fit(X_train, y_train)
    X_train_discretized = discretizer.transform(X_train)

    #apply same discretization to test set
    X_test_discretized = discretizer.transform(X_test)

    #Print a slice of original and discretized data
    print 'Original dataset:\n%s' % str(X_train[0:5])
    print 'Discretized dataset:\n%s' % str(X_train_discretized[0:5])

    #see how feature 0 was discretized
    print 'Feature: %s' % feature_names[0]
    print 'Interval cut-points: %s' % str(discretizer._cuts[0])
    print 'Bin descriptions: %s' % str(discretizer._bin_descriptions[0])

if __name__ == '__main__':
    main()