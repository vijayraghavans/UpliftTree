import numpy as np
import pandas as pd

class Node:
    def __init__(self, nodeType=None, depth=None, n_items=None, ate=None, split_feat=None, split_threshold=None):        
        self.nodeType=nodeType
        self.depth=depth
        self.n_items=n_items
        self.ate=ate
        self.split_feat=split_feat
        self.split_threshold=split_threshold
        self.leftNode=None
        self.rightNode=None

class NodeData:
    def __init__(self, features=None,treatment=None, target=None):
        self.features=features
        self.treatment=treatment
        self.length_treatment=len([x for x in treatment if x == 1])
        self.target=target        
        self.items=len(target)        
        self.control=(treatment-1)*(-1) 
        self.length_control=len([x for x in self.control if x == 1])
        avgTreatment = sum(np.multiply(treatment, target))/self.length_treatment
        avgControl = sum(np.multiply(self.control, target))/self.length_control
        self.M=(avgTreatment-avgControl)

class Split:
    def __init__(self, left:NodeData=None, right:NodeData=None, deltaDeltaP=None, split_threshold=None, split_feat=None):
        self.left=left
        self.right=right
        self.deltaDeltaP=deltaDeltaP
        self.split_threshold=split_threshold
        self.split_feat=split_feat

def getThresholdValues(column_values):
    unique_values = np.unique(column_values)
    if len(unique_values) >10:
        percentiles = np.percentile(column_values, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
    else:
        percentiles = np.percentile(unique_values, [10, 50, 90])
    threshold_options = np.unique(percentiles)
    return threshold_options

def makesplit(features,treatment, perValue,target,treat,index):
    leftData = []
    rightData  = []
    leftTreatment = []
    rightTreatment = []
    leftTarget = []
    rightTarget = []

    feat=features[:,index]
    for i in range(len(feat)):
        if feat[i] <=  perValue:
            leftData.append(features[i])
            leftTreatment.append(treatment[i])
            leftTarget.append(target[i])
        else:
            rightData.append(features[i])
            rightTreatment.append(treatment[i])
            rightTarget.append(target[i])

    leftData = np.array(leftData)
    rightData  = np.array(rightData)
    leftTreatment = np.array(leftTreatment)
    rightTreatment = np.array(rightTreatment)
    leftTarget = np.array(leftTarget)
    rightTarget = np.array(rightTarget)
    
    leftNodeData=NodeData(leftData,leftTreatment,leftTarget)
    rightNodeData=NodeData(rightData,rightTreatment,rightTarget)
    return leftNodeData,rightNodeData

def numtostr(num):
    converted_num = "% s" % num
    return converted_num
    
def retrunTabs(dd):
    tabs=""
    for r in range(dd):
        tabs+="\t"
    return tabs

def writeNode(n:Node):
    if n.nodeType == 'root':
        ioType='w'
    else:
        ioType='a'
    with open('NodeModel.txt', ioType) as f:
        f.write(retrunTabs(n.depth))
        f.write("Level " + numtostr(n.depth) + "\n")
        f.write(retrunTabs(n.depth))
        f.write(n.nodeType + "\n")
        f.write(retrunTabs(n.depth))
        f.write("Number of items : " + numtostr(n.n_items)+"\n")
        f.write(retrunTabs(n.depth))
        f.write("ATE : " + numtostr(n.ate)+"\n")
        f.write(retrunTabs(n.depth))
        f.write("Split_Feature : " + numtostr(n.split_feat)+"\n")
        f.write(retrunTabs(n.depth))
        f.write("Split_Threshold : " + numtostr(n.split_threshold)+"\n\n")
    f.close()    
    if(n.leftNode):
        writeNode(n.leftNode)
    if(n.rightNode):
        writeNode(n.rightNode)

def checkModel(value, model:Node):
    prediction=model.ate
    if model.leftNode is not None or model.rightNode is not None:
        prediction=0
        featureCheck=model.split_feat
        # print(model.split_threshold)
        if value[featureCheck] <= model.split_threshold:
            prediction=checkModel(value,model.leftNode)
        elif value[featureCheck] > model.split_threshold:
            prediction=checkModel(value,model.rightNode)
    return prediction

def predictValues(valueArray:np.ndarray, nodeModel:Node):
    predictions=[]
    for value in valueArray:
        p=checkModel(value,nodeModel)
        predictions.append(p)
    return predictions

class UpliftTreeRegressor:
    def __init__(self, Max_depth: int =3, Min_samples_leaf: int = 1000, Min_samples_leaf_treated: int = 300, Min_samples_leaf_control: int = 300):
        self.Max_depth=Max_depth
        self.Min_samples_leaf=Min_samples_leaf
        self.Min_samples_leaf_treated=Min_samples_leaf_treated
        self.Min_samples_leaf_control=Min_samples_leaf_control

    def buildNode(self,node:Node, nodeData:NodeData):
        if(node.depth<self.Max_depth):
            splits= []
            for colIndex in range(len(nodeData.features[0])):
                # print("Feature:",colIndex)
                feature = nodeData.features[:,colIndex]
                thresholds = getThresholdValues(feature)

                for threshold in thresholds:
                    # print("threshold:",threshold)
                    data_left, data_right=makesplit(nodeData.features,nodeData.treatment, threshold,nodeData.target,nodeData.treatment,colIndex)
                    if data_left.items>=self.Min_samples_leaf and data_left.length_treatment>=self.Min_samples_leaf_treated and data_left.length_control>=self.Min_samples_leaf_control and data_right.items>=self.Min_samples_leaf and data_right.length_treatment>=self.Min_samples_leaf_treated and data_right.length_control>=self.Min_samples_leaf_control:
                        deltaDeltaP=abs(data_left.M-data_right.M)
                        splits.append(Split(data_left,data_right,deltaDeltaP,threshold,colIndex))

            if len(splits) > 0:
                bestSplit = max(splits,key=lambda x:x.deltaDeltaP)
                node.nodeType = node.nodeType.replace(' <leaf>', '')
                node.split_feat = bestSplit.split_feat
                node.split_threshold = bestSplit.split_threshold
                leftNode = Node("left <leaf>",node.depth+1,bestSplit.left.items,bestSplit.left.M)
                rightNode = Node("right <leaf>",node.depth+1,bestSplit.right.items,bestSplit.right.M)
                node.leftNode = leftNode
                node.rightNode = rightNode   
                
                # print(nodeData.M)
                bestSplit.left

                self.buildNode(leftNode,bestSplit.left)    
                self.buildNode(rightNode,bestSplit.right) 

    def fit(self, X: np.ndarray, Treatment: np.ndarray, Y: np.ndarray):
        items=len(Y)
        length_treatment=len([x for x in Treatment if x == 1])
        Control=(Treatment-1)*(-1) 
        length_control=len([x for x in Control if x == 1])
        rootTreatment = sum(np.multiply(Treatment, Y))/length_treatment
        rootControl = sum(np.multiply(Control, Y))/length_control
        ate=(rootTreatment-rootControl)


        # ate=sum(np.multiply(Y,Treatment))/items
        self.rootNode=Node("root",0,items,ate)
        self.rootNodedata=NodeData(X,Treatment,Y)
        
        self.buildNode(self.rootNode,self.rootNodedata)        
        writeNode(self.rootNode)

    def predict(self, X: np.ndarray):
        self.predictions=predictValues(X,self.rootNode)
        np.save('predictions.npy', self.predictions)
        return self.predictions

givenArray=np.load("example_X.npy",allow_pickle=True)
givenTarget=np.load("example_y.npy")
givenTreatment=np.load("example_treatment.npy")
param_dict = {
    "max_depth":3,
    "min_samples_leaf" : 6000,
    "min_samples_leaf_trated":2500,
    "min_samples_leaf_control":2500
}

upliftTree=UpliftTreeRegressor(param_dict["max_depth"],param_dict["min_samples_leaf"],param_dict["min_samples_leaf_trated"],param_dict["min_samples_leaf_control"])

upliftTree.fit(givenArray,givenTreatment,givenTarget)

predictions=upliftTree.predict(givenArray)

