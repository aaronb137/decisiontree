import numpy as np
import math as m

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def entropyFormula(yesVals, noVals, totalVals):
    if yesVals == 0 or noVals == 0:
        return 0

    return -(yesVals/totalVals) * m.log2(yesVals/totalVals) - (noVals/totalVals) * m.log2(noVals/totalVals)

def CalculateTotalEntropy(Y):
    totalLabels = len(Y)
    yesLabels = np.count_nonzero(Y == 1)
    noLabels = np.count_nonzero(Y == 0)

    return entropyFormula(yesLabels, noLabels, totalLabels)

def CalculateFeatureEntropy(X, Y):
    featEntropy = {}

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
    
    return featEntropy

def CalculateIG(entropy, Y):
    print(2)

def DT_train_binary(X,Y,max_depth):
    if (len(X) != len(Y)):
        raise ValueError('Param 1 and 2 require same length arrays..')
    totalEntropy = CalculateTotalEntropy(Y)
    print(totalEntropy)


def DT_test_binary(X,Y,DT):
    print("fart1")

def DT_make_prediction(x,DT):
    print("fart2")

def DT_train_real(X,Y,max_depth):
    print("fart3")

def DT_test_real(X,Y,DT):
    print("fart4")