import numpy as np
import math as m

# class approach for implementing tree for decision tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root: TreeNode) -> int:
    stack = [ [root, 1] ]
    maxDepth = 0

    while stack:
        node, depth = stack.pop()

        if node: 
            maxDepth = max(depth, maxDepth)
            stack.append( [node.left, 1 + depth] )
            stack.append( [node.right, 1 + depth] )

    return maxDepth


# helper function to calculate entropy for features/training set
def entropyFormula(yesVals, noVals, totalVals):
    if yesVals == 0 or noVals == 0:
        return 0

    return -(yesVals/totalVals) * m.log2(yesVals/totalVals) - (noVals/totalVals) * m.log2(noVals/totalVals)

# calculate entropy of entire training set
def CalculateTotalEntropy(Y):
    totalLabels = len(Y)
    yesLabels = np.count_nonzero(Y == 1)
    noLabels = np.count_nonzero(Y == 0)

    return entropyFormula(yesLabels, noLabels, totalLabels)

# calculate entropy of each feature inside training set
def CalculateFeatureEntropy(X, Y):
    featEntropy = {}
    proportionCnt = {}

    _, cols = np.shape(X)
    
    for i in range(0, cols, 1):
        featureColumn = X[:,i]
        featureLabelCol = np.vstack((featureColumn, Y)).T

        boolNoMask = featureLabelCol[:, 0] == 0
        boolYesMask = featureLabelCol[:, 0] == 1
        noSet = featureLabelCol[boolNoMask]
        yesSet = featureLabelCol[boolYesMask]

        totalNo = len(noSet)
        totalYes = len(yesSet)

        correctNoLabelCt = np.count_nonzero(noSet[:,1] == 1)
        incorrectNoLabelCt = totalNo - correctNoLabelCt

        correctYesLabelCt = np.count_nonzero(yesSet[:,1] == 1)
        incorrectYesLabelCt = totalYes - correctYesLabelCt

        noEntropy = entropyFormula(correctNoLabelCt, incorrectNoLabelCt, totalNo)
        yesEntropy = entropyFormula(correctYesLabelCt, incorrectYesLabelCt, totalYes)
        
        featEntropy[i] = (noEntropy, yesEntropy)
        proportionCnt[i] = (totalNo, totalYes)
    
    return featEntropy, proportionCnt

# calculate information gain to decide construction of decision tree
def CalculateIG(totalEntropy, featEntropyList, proportionCnt):
    IG_result = {}

    for i in range( len(featEntropyList) ):
        totalNo, totalYes = proportionCnt[i][0], proportionCnt[i][1]
        noEntropy, yesEntropy = featEntropyList[i][0], featEntropyList[i][1]

        total_training_data = totalYes + totalNo
        IG_result[i] = totalEntropy - ( ( (totalYes/total_training_data) * yesEntropy) + ( (totalNo/total_training_data) * noEntropy ) )
    
    return IG_result

# train decision tree model on given training feature data (X), training labels (Y), and 
# train until provided max depth (-1 if we want to continue until IG is 0 or we run out of features)

# * REQUIRES EQUAL LENGTH BETWEEN TRAINING FEATURE DATA AND TRAINING LABELS
def DT_train_binary(X,Y,max_depth):
    if (len(X) != len(Y)):
        raise ValueError('Param 1 and 2 require same length arrays..')
    
    totalEntropy = CalculateTotalEntropy(Y)
    featEntropyList, proportionCnt = CalculateFeatureEntropy(X, Y)
    training_set_IG = CalculateIG(totalEntropy, featEntropyList, proportionCnt)

    root = TreeNode(None)
    currentDepth = 0

    #TODO: implement tree using classes/nodes. use recursion to build tree and keep track of current depth of tree.


    if max_depth != -1:
        while currentDepth != max_depth:
            currentDepth = maxDepth(root)

    #TODO: implement logic where we keep learning until we run out of features or IG is 0
    

def DT_test_binary(X,Y,DT):
    print("fart1")

def DT_make_prediction(x,DT):
    print("fart2")


#grad section
def DT_train_real(X,Y,max_depth):
    print("fart3")

def DT_test_real(X,Y,DT):
    print("fart4")

