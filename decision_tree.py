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
	def __init__(self, data_set, build_subtree = False, depth = 1, tree = None):
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
			self.decision = max(unique_labels, key= lambda l : counts[unique_labels.tolist().index(l)])
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

	def classify(self, data_in):
		if self.is_pure:
			return self.decision
		if data_in[self.feature] <= self.threshold:
			return self.left_child.classify(data_in)
		else:
			return self.right_child.classify(data_in)

	def __str__(self):
		if self.is_pure:
			return '%d: Selecting %d' % (self.depth, self.decision)
		else:
			return '%d: Is Feature %d <= %f (Majority %d)' % (self.depth, self.feature + 1, self.threshold, self.decision) 


class D_Tree(object):
	"""
	A full decision tree, with the root 
	"""
	def __init__(self, data_set):
		self.data_set = data_set
		self.root = D_Node(data_set, build_subtree = True, tree = self)

	def classify(self, data_in):
		n = self.root
		while n:
			if n.is_pure:
				return n.decision
			if data_in[n.feature] <= n.threshold:
				n = n.left_child
			else:
				n = n.right_child



	def error(self, validation_data, pruning_node = None):
		total = len(validation_data)
		hits = 0.0
		misses = 0.0
		for vector, label in validation_data:
			n = self.root
			while n:
				# If this is the node we are testing for pruning,
				# stop the classification here and predict the majority
				# label in n
				if n is pruning_node and n is not None:
					if n.decision == label:
						hits += 1.0
					else:
						misses += 1.0
					break
				if n.is_pure:
					if n.decision == label:
						hits += 1.0
					else:
						misses += 1.0
					break
				if vector[n.feature] <= n.threshold:
					n = n.left_child
				else:
					n = n.right_child
		return (misses / total)


	def prune_tree(self, validation_data, prune_once = True):
		bfs_queue = [tree.root]
		tree_error = self.error(validation_data)
		while bfs_queue:
			n = bfs_queue[0]

			pruned_error = self.error(validation_data, n)

			if pruned_error < tree_error:
				print str(n)
				print pruned_error, '<', tree_error
				n.is_pure = True
				n.left_child = None
				n.right_child = None
				tree_error = pruned_error
				if prune_once:
					return

			if not n.is_pure:
				bfs_queue.append(n.left_child)
				bfs_queue.append(n.right_child)

			bfs_queue.pop(0)





if __name__ == '__main__':
	training_data = read_data(Data.fdir, Data.training, Data.vector_size)
	test_data = read_data(Data.fdir, Data.testing, Data.vector_size)
	validation_data = read_data(Data.fdir, Data.validate, Data.vector_size)

	tree = D_Tree(training_data)

	print 'Training Error: %f' % tree.error(training_data)
	print 'Testing Error: %f' % tree.error(test_data)

	tree.prune_tree(validation_data)

	print 'Training Error: %f' % tree.error(training_data)
	print 'Testing Error: %f' % tree.error(test_data)
	
	tree.prune_tree(validation_data)

	print 'Training Error: %f' % tree.error(training_data)
	print 'Testing Error: %f' % tree.error(test_data)




