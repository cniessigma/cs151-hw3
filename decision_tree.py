import scipy.stats as stats
import numpy as np
from data import Data


def read_data(fdir, fname, vector_size):
	"""
	Opens the file using that Data class members and reads in the data.
	Data is stores as a list of tuples: [(val1, val2), (val1, val2)]
	Tuple value 1 is the vector (list)
	Tuple value 2 is the label for the vector (int)
	"""
	with open(fdir+fname, 'r') as f:
		data = f.readlines()
	data = [ x.strip().split(' ') for x in data ] # strip \n and split using ' '
	data = [ [float(y) for y in x] for x in data ] # convert all values to ints
	data = [ (np.array(x[:vector_size]), x[vector_size]) for x in data ] # convert to tuple
	return data

class D_Node(object):
	"""
	A decision tree node. Contains a list of the node's data set, calculates its entropy,
	and can be used to generate the most information gaining split. Assumes all data vectors
	are the same size
	"""
	def __init__(self, data_set, build_subtree = False, depth = 1):
		self.data_set = data_set
		self.depth = depth
		self.vector_size = len(data_set[0][0])
		self.left_child = None
		self.right_child = None
		self.entropy = self._calculate_entropy()

		if not self.is_pure:
			self.feature, self.threshold = self._select_feature_and_threshold()
			if build_subtree:
				less_than = [tup for tup in self.data_set if tup[0][self.feature] <= self.threshold]
				greater_than = [tup for tup in self.data_set if tup[0][self.feature] > self.threshold]
				self.left_child = D_Node(less_than, build_subtree = True, depth = depth + 1)
				self.right_child = D_Node(greater_than, build_subtree = True, depth = depth + 1)
		




	def _select_feature_and_threshold(self):
		thresholds = [self._best_threshold_for_feature(i) for i in range(self.vector_size)]
		min_entropy_threshold = None
		best_feature = None
		for feature_index, threshold_tuple in enumerate(thresholds):
			if threshold_tuple[0] is None or threshold_tuple[1] is None:
				continue

			curr_threshold, curr_entropy = threshold_tuple

			if best_feature is None:
				min_entropy_threshold = threshold_tuple
				best_feature = feature_index
			elif curr_entropy < min_entropy_threshold[1]:
				min_entropy_threshold = threshold_tuple
				best_feature = feature_index


		return best_feature, min_entropy_threshold[0]

	def _calculate_entropy(self, data_set_in = None):
		"""
		Calculates the entropy of this node's dataset, as well as determining whether
		or not the data set is pure. If it's pure, then it's a leaf node.
		"""
		if data_set_in is None:
			data_set = self.data_set
		else:
			data_set = data_set_in

		# Count the occurences of each label in the dataset
		unique_labels, counts = np.unique([tup[1] for tup in data_set], return_counts=True)
		if len(unique_labels) > 1 and data_set_in is None:
			self.is_pure = False
		elif len(unique_labels == 1) and data_set_in is None:
			self.is_pure = True
			self.decision = unique_labels[0]

		# Normalize the list of counts into a list of probabilities.
		probabilities = counts.astype('float') / len(data_set)
		return stats.entropy(probabilities)

	def _best_threshold_for_feature(self, feature_index):
		"""
		Calculates which threshold minimizes the entropy of the resultant subsets,
		given that we are splitting based on feature 'feature_index'
		"""
		feature_values = [(vector[0][feature_index], vector[1]) for vector in self.data_set]
		unique_values = np.unique([tup[0] for tup in feature_values])

		# TODO? ADD FIX FOR EXACT DATAPOINT WITH DIFFERENT LABELS, IE
		# [([2], 1), ([2], 0), ([1], 0), ([1], 0)]
		best_threshold = None
		min_entropy = None
		for i in range(1, len(unique_values)):
			threshold = (unique_values[i] + unique_values[i - 1]) / 2.0
			less_than = [tup for tup in feature_values if tup[0] <= threshold]
			greater_than = [tup for tup in feature_values if tup[0] > threshold]
			p_less_than = len(less_than) / float(len(feature_values))
			p_greater_than = len(greater_than) / float(len(feature_values))
			cond_entropy = p_less_than*self._calculate_entropy(less_than) + p_greater_than*self._calculate_entropy(greater_than)

			if best_threshold is None:
				best_threshold = threshold
				min_entropy = cond_entropy
			elif cond_entropy < min_entropy:
				best_threshold = threshold
				min_entropy = cond_entropy
		

		return (best_threshold, min_entropy)

	def __str__(self):
		if self.is_pure:
			return '%d: Selecting %d' % (self.depth, self.decision)
		else:
			return '%d: Is x_%d <= %f' % (self.depth, self.feature, self.threshold) 


class D_Tree(object):
	"""
	A full decision tree, with the root 
	"""
	def __init__(self, data_set):
		self.data_set = data_set
		self.root = D_Node(data_set, build_subtree = True)



if __name__ == '__main__':
	data = read_data(Data.fdir, Data.training, Data.vector_size)
	tree = D_Tree(data)
	q = [tree.root]
	while q:
		n = q[0]
		if n.depth > 3:
			break
		print str(n)
		q.append(n.left_child)
		q.append(n.right_child)
		q.pop(0)



