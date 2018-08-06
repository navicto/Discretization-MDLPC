from __future__ import division
__author__ = 'Victor Ruiz, vmr11@pitt.edu'
import numpy as np
from Entropy import entropy_numpy, cut_point_information_gain_numpy
from math import log
from sklearn.base import TransformerMixin
from sklearn import datasets
from sklearn.model_selection import train_test_split

def previous_item(a, val):
    idx = np.where(a == val)[0][0] - 1
    return a[idx]

class MDLP_Discretizer(TransformerMixin):
    def __init__(self, features=None, raw_data_shape=None):
        '''
        initializes discretizer object:
            saves raw copy of data and creates self._data with only features to discretize and class
            computes initial entropy (before any splitting)
            self._features = features to be discretized
            self._classes = unique classes in raw_data
            self._class_name = label of class in pandas dataframe
            self._data = partition of data with only features of interest and class
            self._cuts = dictionary with cut points for each feature
        :param X: pandas dataframe with data to discretize
        :param class_label: name of the column containing class in input dataframe
        :param features: if !None, features that the user wants to discretize specifically
        :return:
        '''
        #Initialize descriptions of discretizatino bins
        self._bin_descriptions = {}

        #Create array with attr indices to discretize
        if features is None:  # Assume all columns are numeric and need to be discretized
            if raw_data_shape is None:
                raise Exception("If feautes=None, raw_data_shape must be a non-empty tuple")
            self._col_idx = range(raw_data_shape[1])
        else:
            if not isinstance(features, np.ndarray):
                features = np.array(features)
            if np.issubdtype(features.dtype, np.integer):
                self._col_idx = features
            elif np.issubdtype(features.dtype, np.bool):  # features passed as mask
                if raw_data_shape is None:
                    raise Exception('If features is a boolean array, raw_data_shape must be != None')
                if len(features) != self._data_raw.shape[1]:
                    raise Exception('Column boolean mask must be of dimensions (NColumns,)')
                self._col_idx = np.where(features)
            else:
                raise Exception('features argument must a np.array of column indices or a boolean mask')

    def fit(self, X, y):
        self._data_raw = X  # copy of original input data
        self._class_labels = y.reshape(-1, 1)  # make sure class labels is a column vector
        self._classes = np.unique(self._class_labels)


        if len(self._col_idx) != self._data_raw.shape[1]:  # some columns will not be discretized
            self._ignore_col_idx = np.array([var for var in range(self._data_raw.shape[1]) if var not in self._col_idx])

        # initialize feature bins cut points
        self._cuts = {f: [] for f in self._col_idx}

        # pre-compute all boundary points in dataset
        self._boundaries = self.compute_boundary_points_all_features()

        # get cuts for all features
        self.all_features_accepted_cutpoints()

        #generate bin string descriptions
        self.generate_bin_descriptions()

        #Generate one-hot encoding schema

        return self

    def transform(self, X, inplace=False):
        if inplace:
            discretized = X
        else:
            discretized = X.copy()
        discretized = self.apply_cutpoints(discretized)
        return discretized
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, inplace=True)

    def MDLPC_criterion(self, X, y, feature_idx, cut_point):
        '''
        Determines whether a partition is accepted according to the MDLPC criterion
        :param feature: feature of interest
        :param cut_point: proposed cut_point
        :param partition_index: index of the sample (dataframe partition) in the interval of interest
        :return: True/False, whether to accept the partition
        '''
        #get dataframe only with desired attribute and class columns, and split by cut_point
        left_mask = X <= cut_point
        right_mask = X > cut_point

        #compute information gain obtained when splitting data at cut_point
        cut_point_gain = cut_point_information_gain_numpy(X, y, cut_point)
        #compute delta term in MDLPC criterion
        N = len(X) # number of examples in current partition
        partition_entropy = entropy_numpy(y)
        k = len(np.unique(y))
        k_left = len(np.unique(y[left_mask]))
        k_right = len(np.unique(y[right_mask]))
        entropy_left = entropy_numpy(y[left_mask])  # entropy of partition
        entropy_right = entropy_numpy(y[right_mask])
        delta = log(3 ** k, 2) - (k * partition_entropy) + (k_left * entropy_left) + (k_right * entropy_right)

        #to split or not to split
        gain_threshold = (log(N - 1, 2) + delta) / N

        if cut_point_gain > gain_threshold:
            return True
        else:
            return False

    def feature_boundary_points(self, values):
        '''
        Given an attribute, find all potential cut_points (boundary points)
        :param feature: feature of interest
        :param partition_index: indices of rows for which feature value falls whithin interval of interest
        :return: array with potential cut_points
        '''

        missing_mask = np.isnan(values)
        data_partition = np.concatenate([values[:, np.newaxis], self._class_labels], axis=1)
        data_partition = data_partition[~missing_mask]
        #sort data by values
        data_partition = data_partition[data_partition[:, 0].argsort()]

        #Get unique values in column
        unique_vals = np.unique(data_partition[:, 0])  # each of this could be a bin boundary
        #Find if when feature changes there are different class values
        boundaries = []
        for i in range(1, unique_vals.size):  # By definition first unique value cannot be a boundary
            previous_val_idx = np.where(data_partition[:, 0] == unique_vals[i-1])[0]
            current_val_idx = np.where(data_partition[:, 0] == unique_vals[i])[0]
            merged_classes = np.union1d(data_partition[previous_val_idx, 1], data_partition[current_val_idx, 1])
            if merged_classes.size > 1:
                boundaries += [unique_vals[i]]
        boundaries_offset = np.array([previous_item(unique_vals, var) for var in boundaries])
        return (np.array(boundaries) + boundaries_offset) / 2

    def compute_boundary_points_all_features(self):
        '''
        Computes all possible boundary points for each attribute in self._features (features to discretize)
        :return:
        '''
        def padded_cutpoints_array(arr, N):
            cutpoints = self.feature_boundary_points(arr)
            padding = np.array([np.nan] * (N - len(cutpoints)))
            return np.concatenate([cutpoints, padding])

        boundaries = np.empty(self._data_raw.shape)
        boundaries[:, self._col_idx] = np.apply_along_axis(padded_cutpoints_array, 0, self._data_raw[:, self._col_idx], self._data_raw.shape[0])
        mask = np.all(np.isnan(boundaries), axis=1)
        return boundaries[~mask]

    def boundaries_in_partition(self, X, feature_idx):
        '''
        From the collection of all cut points for all features, find cut points that fall within a feature-partition's
        attribute-values' range
        :param data: data partition (pandas dataframe)
        :param feature: attribute of interest
        :return: points within feature's range
        '''
        range_min, range_max = (X.min(), X.max())
        mask = np.logical_and((self._boundaries[:, feature_idx] > range_min), (self._boundaries[:, feature_idx] < range_max))
        return np.unique(self._boundaries[:, feature_idx][mask])

    def best_cut_point(self, X, y, feature_idx):
        '''
        Selects the best cut point for a feature in a data partition based on information gain
        :param data: data partition (pandas dataframe)
        :param feature: target attribute
        :return: value of cut point with highest information gain (if many, picks first). None if no candidates
        '''
        candidates = self.boundaries_in_partition(X, feature_idx=feature_idx)
        if candidates.size == 0:
            return None
        gains = [(cut, cut_point_information_gain_numpy(X, y, cut_point=cut)) for cut in candidates]
        gains = sorted(gains, key=lambda x: x[1], reverse=True)

        return gains[0][0] #return cut point

    def single_feature_accepted_cutpoints(self, X, y, feature_idx):
        '''
        Computes the cuts for binning a feature according to the MDLP criterion
        :param feature: attribute of interest
        :param partition_index: index of examples in data partition for which cuts are required
        :return: list of cuts for binning feature in partition covered by partition_index
        '''

        #Delte missing data
        mask = np.isnan(X)
        X = X[~mask]
        y = y[~mask]

        #stop if constant or null feature values
        if len(np.unique(X)) < 2:
            return
        #determine whether to cut and where
        cut_candidate = self.best_cut_point(X, y, feature_idx)
        if cut_candidate == None:
            return
        decision = self.MDLPC_criterion(X, y, feature_idx, cut_candidate)

        # partition masks
        left_mask = X <= cut_candidate
        right_mask = X > cut_candidate

        #apply decision
        if not decision:
            return  # if partition wasn't accepted, there's nothing else to do
        if decision:
            #now we have two new partitions that need to be examined
            left_partition = X[left_mask]
            right_partition = X[right_mask]
            if (left_partition.size == 0) or (right_partition.size == 0):
                return #extreme point selected, don't partition
            self._cuts[feature_idx] += [cut_candidate]  # accept partition
            self.single_feature_accepted_cutpoints(left_partition, y[left_mask], feature_idx)
            self.single_feature_accepted_cutpoints(right_partition, y[right_mask], feature_idx)
            #order cutpoints in ascending order
            self._cuts[feature_idx] = sorted(self._cuts[feature_idx])
            return

    def all_features_accepted_cutpoints(self):
        '''
        Computes cut points for all numeric features (the ones in self._features)
        :return:
        '''
        for attr in self._col_idx:
            self.single_feature_accepted_cutpoints(X=self._data_raw[:, attr], y=self._class_labels, feature_idx=attr)
        return

    def generate_bin_descriptions(self):
        '''
        Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
        the bins
        :param out_data_path: path to save discretized data
        :param out_bins_path: path to save bins description
        :return:
        '''
        bin_label_collection = {}
        for attr in self._col_idx:
            if len(self._cuts[attr]) == 0:
                bin_label_collection[attr] = ['All']
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                start_bin_indices = range(0, len(cuts) - 1)
                bin_labels = ['%s_to_%s' % (str(cuts[i]), str(cuts[i+1])) for i in start_bin_indices]
                bin_label_collection[attr] = bin_labels
                self._bin_descriptions[attr] = {i: bin_labels[i] for i in range(len(bin_labels))}


    def apply_cutpoints(self, data):
        '''
        Discretizes data by applying bins according to self._cuts. Saves a new, discretized file, and a description of
        the bins
        :param out_data_path: path to save discretized data
        :param out_bins_path: path to save bins description
        :return:
        '''
        for attr in self._col_idx:
            if len(self._cuts[attr]) == 0:
                # data[:, attr] = 'All'
                data[:, attr] = 0
            else:
                cuts = [-np.inf] + self._cuts[attr] + [np.inf]
                discretized_col = np.digitize(x=data[:, attr], bins=cuts, right=False).astype('float') - 1
                discretized_col[np.isnan(data[:, attr])] = np.nan
                data[:, attr] = discretized_col
        return data