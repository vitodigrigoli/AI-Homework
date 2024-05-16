import numpy as np

class DecisionNode:
	''' Nodo di decisione che rappresenta un nodo in un albero decisionale '''

	def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
		'''
		feature_index = indice delle feature utilizzata per dividere i nodi figli
		threshold = soglia per la feature al nodo corrente per dividere i dati
		left, right = nodi figli del nodo corrente
		value = valore di classe se il nodo è un nodo foglia
		'''
		self.feature_index = feature_index
		self.threshold = threshold
		self.left = left
		self.right = right
		self.value = value


class DecisionTree:
	'''
	Albero Decisionale
	'''

	def __init__(self, max_depth=10):
		self.max_depth = max_depth
		self.root = None

	def _most_common_label(self, y):
		'''
		Ritorna la label più comune in un array di labels
		'''
		if len(y) == 0:
			return None

		counts = np.bincount(y)
		return np.argmax(counts)

	def _gini(self, y):
		_, counts = np.unique(y, return_counts=True)
		probabilities = counts / counts.sum()
		return 1 - np.sum(np.square(probabilities))

	def _information_gain(self, y, feature_column, split_thresh):
		'''
		Calcola il guadagno di informazione per un determinato split
		'''
		# suddivisione dei dati
		left_idxs = feature_column < split_thresh
		right_idxs = feature_column >= split_thresh

		# percentuale di samples per ogni divisione
		left_count, right_count = np.sum(left_idxs), np.sum(right_idxs)
		total_count = len(y)

		if left_count == 0 or right_count == 0:
			return 0

		p_left, p_right = left_count/total_count, right_count/total_count

		# calcola il guadagno di informazione
		gain = self._gini(y) - (p_left * self._gini(y[left_idxs])) + (p_right * self._gini(y[right_idxs]))

		return gain

	def _best_split(self, X, y, n_features):
		'''
		Trova il miglior split per i dati di input
		'''
		best_feature, best_thresh = None, None
		best_gain = -1

		for feature_index in range(n_features):
			thresholds = np.unique(X[:, feature_index])

			for threshold in thresholds:
				gain = self._information_gain(y, X[:, feature_index], threshold)
				if gain > best_gain:
					best_gain = gain
					best_feature = feature_index
					best_thresh = threshold

		return best_feature, best_thresh

	def _build_tree(self, X, y, depth=0):
		n_sample, n_features = X.shape

		# controllo per creare un nodo foglia
		if n_sample == 0 or depth == self.max_depth:
			leaf_value = self._most_common_label(y)
			return DecisionNode(value=leaf_value)

		# calcolo della migliore suddivisione
		best_feature, best_thresh = self._best_split(X, y, n_features)

		if best_feature == None:
			leaf_value = self._most_common_label(y)
			return DecisionNode(value=leaf_value)

		# Trova gli indici per dividere i dati in nodi figli
		left_idxs = X[:, best_feature] < best_thresh
		right_idxs = X[:, best_feature] >= best_thresh

		# Ricorsivamente costruisce i nodi figli
		left_subtree = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
		right_subtree = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)

		return DecisionNode(feature_index=best_feature, threshold=best_thresh, left=left_subtree, right=right_subtree)

	def fit(self, X, y):
		'''
		Addestra l'albero decisionale usando i dati di training X, y
		'''
		self.root = self._build_tree(X, y)
  
	def _traverse_tree(self, x, node: DecisionNode):
		'''
  		Percorre l'albero in ogni punto e ritorna la classe
    	'''

		if node.value is not None:
			return node.value

		if x[node.feature_index] < node.threshold:
			return self._traverse_tree(x, node.left)
		else:
			return self._traverse_tree(x, node.right)
   

	def predict(self, X):
		"""
  		Predice le classi per i dati X usando l'albero addestrato
    	"""
		return [self._traverse_tree(x, self.root) for x in X]
