import numpy as np
from collections import Counter

# * decision tree functions/classes
# class data structure for tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        return self.value is not None

# calculate entropy of entire training set
def calculate_entropy(Y):
    freqCount = np.bincount(Y)
    ps = freqCount/len(Y)
    return -np.sum([p * np.log2(p) for p in ps if p>0])

# calculate information gain to decide construction of decision tree
def info_gain(y,  X_col, threshold):
    system_entropy = calculate_entropy(y)

    left_idxs, right_idxs = split(X_col, threshold)

    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0

    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    e_l, e_r = calculate_entropy(y[left_idxs]), calculate_entropy(y[right_idxs])
    child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

    # calculate the IG
    information_gain = system_entropy - child_entropy
    return information_gain

# split our data (used when we create child nodes)
def split(X_column, split_thresh):
    left_indexes = np.argwhere(X_column <= split_thresh).flatten()
    right_indexes = np.argwhere(X_column > split_thresh).flatten()
    return left_indexes, right_indexes

# find the highest IG and that will be our best split
def best_split(X, Y, feat_indexes):
    best_gain = -1
    split_index, split_threshold = None, None

    for feat_idx in feat_indexes:
        X_column = X[:, feat_idx]
        thresholds = np.unique(X_column)

        for thresh in thresholds:
            gain = info_gain(Y, X_column, thresh)

            if gain > best_gain:
                best_gain = gain
                split_index = feat_idx
                split_threshold = thresh

    return split_index, split_threshold

# pick the label that occurs the most and return that label (for leaf nodes)
def common_label(y):
    counter = Counter(y)
    value = counter.most_common(1)[0][0]
    return value

# # train decision tree model on given training feature data (X), training labels (Y), and 
# # train until provided max depth (-1 if we want to continue until IG is 0 or we run out of features)
def DT_train_binary(X,Y,max_depth):
    if (len(X) != len(Y)):
        raise ValueError('Param 1 and 2 require same length arrays..')
        
    global select_num_feats
    global min_samples_split
    global root_node

    min_samples_split = 2
    select_num_feats = None
    root_node = None

    # we fit our data and generate our tree
    def fit(X, Y):
        global select_num_feats
        global min_samples_split
        global root_node

        select_num_feats = X.shape[1] if not select_num_feats else min(X.shape[1, select_num_feats])
        root_node = create_tree(X, Y, depth=1)

    # generate our tree on X feature data and Y data labels with a depth counter
    def create_tree(X, Y, depth=1):
        # * check stopping criteria
        if (max_depth != -1 and depth >= max_depth):
            leaf_value = common_label(Y)
            return Node(value=leaf_value)
        
        num_samples, num_features = X.shape
        num_labels = len(np.unique(Y))

        # * check stopping criteria
        if (num_labels == 1 or num_samples<=min_samples_split):
            leaf_value = common_label(Y)
            return Node(value=leaf_value)
        
        # * find best split
        feat_indexs = np.random.choice(num_features, select_num_feats, replace=False)
        best_feature, best_threshold = best_split(X, Y, feat_indexs)
        
        # * create child nodes
        left_indexes, right_indexes = split(X[:, best_feature], best_threshold)
        left = create_tree(X[left_indexes, :], Y[left_indexes], depth + 1)
        right = create_tree(X[right_indexes, :], Y[right_indexes], depth + 1)

        return Node(best_feature, best_threshold, left, right)
    
    fit(X, Y)

    return root_node

# check how closely the predictions from decision tree, DT made on
# feature data X and compare with data labels Y
# returns accuracy on how many correct predictions model made
def DT_test_binary(X,Y,DT):
    predictions = DT_make_prediction(X, DT)
    accuracy = np.sum((Y == predictions)) / len(Y)
    return accuracy

def RF_test_random_forest(X, Y, RF):
    final_predictions, echtree_preds = RF_make_prediction(X, RF)

    for index, pred in enumerate(echtree_preds):
        tree_pred = accuracy = np.sum((Y == pred)) / len(Y)
        print(f"DT {index}: {tree_pred}")

    accuracy = np.sum(Y == final_predictions) / len(Y)
    return accuracy

# takes feature data X and a trained DT as Node
# and traverses the tree until it reaches a classification leaf node
def traverse_tree(X, Node: Node):
    if Node is None:
        return None
    if Node.is_leaf_node():
        return Node.value
    if X[Node.feature] <= Node.threshold:
        return traverse_tree(X, Node.left)
    return traverse_tree(X, Node.right)

# * prediction functions (DT, RF)
def DT_make_prediction(X,DT):
    return np.array([traverse_tree(x, DT) for x in X])

def RF_make_prediction(X, RF):
    individual_predictions = np.array([DT_make_prediction(X, tree) for tree in trees])
    tree_predictions = np.swapaxes(individual_predictions, 0, 1)
    final_predictions = np.array([common_label(pred) for pred in tree_predictions])
    return final_predictions, individual_predictions

# * random forest functions
def RF_build_random_forest(X, Y, max_depth, num_of_trees):
    global select_num_feats
    global min_samples_split
    global trees

    min_samples_split = 2
    select_num_feats = None
    trees = []

    def prep_samples(X, Y):
        num_samples = X.shape[0]
        subset_size = int(num_samples * 0.10)
        indexes = np.random.choice(num_samples, subset_size, replace=True)        
        return X[indexes], Y[indexes]

    def fitRF(X, Y):
        global select_num_feats
        global min_samples_split
        global trees

        for _ in range(num_of_trees):
            X_samples, Y_samples = prep_samples(X, Y)
            tree = DT_train_binary(X_samples, Y_samples, max_depth)

            trees.append(tree)

    fitRF(X, Y)

    return trees














# ! past code iterations, does not WORK 

# #  label_points, feature_vectors = [], []
# #     totalEntropy = CalculateTotalEntropy(Y)
# #     featEntropy, proportionCnt = CalculateEntropy(X, Y)
# #     # print(X, Y)
# #     training_set_IG = CalculateIG(totalEntropy, featEntropy, proportionCnt)

# #     currentDepth = 0
# #     maxIG = 0

# #     for key, value in training_set_IG.items():
# #         maxIG = max(maxIG, value)
# #         maxIGPair = (key, value)
# #     featKey, featVal = maxIGPair
# #     root = TreeNode(featKey)
# #     print(training_set_IG)
# #     # stacked_data = np.hstack((X,Y.reshape(-1,1)))
# #     # rows_to_delete = np.where(X[:, featKey] == 1)
# #     # new_training_set = np.delete(stacked_data, rows_to_delete, axis = 0)

# #     currentDepth = maxDepth(root)

# #     if (currentDepth == max_depth):
# #         inorder_rec(root)

# #     if maxIG == 0 or (len(X) == 0):
# #         inorder_rec(root)

# #     if (classifyData(featEntropy[featKey])):
# #         root.right = TreeNode(1)

# #         stacked_data = np.hstack((X,Y.reshape(-1,1)))
# #         rows_to_delete = np.where(X[:, featKey] == 1)
# #         new_training_set = np.delete(stacked_data, rows_to_delete, axis = 0)
# #         for row in new_training_set:
# #             label_points.append(row[-1])
# #             feature_vectors.append(row[:-1])
            
# #         feature_vectors = np.array(feature_vectors, dtype=np.float32)
# #         label_points = np.array(label_points, dtype=np.int32)        

# #         root.left = DT_train_binary(feature_vectors, label_points, max_depth)
# #     else:
# #         root.left = TreeNode(0)

# #         stacked_data = np.hstack((X,Y.reshape(-1,1)))
# #         rows_to_delete = np.where(X[:, featKey] == 0)
# #         new_training_set = np.delete(stacked_data, rows_to_delete, axis = 0)
# #         for row in new_training_set:
# #             label_points.append(row[-1])
# #             feature_vectors.append(row[:-1])
            
# #         feature_vectors = np.array(feature_vectors, dtype=np.float32)
# #         label_points = np.array(label_points, dtype=np.int32)        

# #         root.right = DT_train_binary(feature_vectors, label_points, max_depth)

# # # calculate entropy of each feature inside training set
# # def CalculateEntropy(X, Y):
# #     featEntropy = {}
# #     proportionCnt = {}

# #     if len(X) == 0:
# #         return

# #     _, cols = np.shape(X)

# #     for i in range(0, cols, 1):
# #         featureColumn = X[:,i]
# #         featureLabelCol = np.vstack((featureColumn, Y)).T

# #         boolNoMask = featureLabelCol[:, 0] == 0
# #         boolYesMask = featureLabelCol[:, 0] == 1
# #         noSet = featureLabelCol[boolNoMask]
# #         yesSet = featureLabelCol[boolYesMask]

# #         totalNo = len(noSet)
# #         totalYes = len(yesSet)

# #         correctNoLabelCt = np.count_nonzero(noSet[:,1] == 1)
# #         incorrectNoLabelCt = totalNo - correctNoLabelCt

# #         correctYesLabelCt = np.count_nonzero(yesSet[:,1] == 1)
# #         incorrectYesLabelCt = totalYes - correctYesLabelCt

# #         noEntropy = entropyFormula(correctNoLabelCt, incorrectNoLabelCt, totalNo)
# #         yesEntropy = entropyFormula(correctYesLabelCt, incorrectYesLabelCt, totalYes)
        
# #         featEntropy[i] = (noEntropy, yesEntropy)
# #         proportionCnt[i] = (totalNo, totalYes)
    
# #     return featEntropy, proportionCnt