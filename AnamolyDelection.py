import pandas as pd
import time
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from sklearn import preprocessing

trainDataFile = "data/train.csv"
testDataFile  = "data/test.csv"
outputFile    = "output.csv"

def loadDataset(fileName):
    data = pd.read_csv(fileName)
    data['protocol']=data['protocol'].map({"TCP":0,"UDP":1,"HOPOPT":2})
    return data
    
def preprocessTrainData(trainData):
    mm_scaler = preprocessing.MinMaxScaler()
    trainData.iloc[:,1:] = mm_scaler.fit_transform(trainData.iloc[:,1:].values)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(trainData.iloc[:,1:])

    trainData['pca-one'] = pca_result[:,0]
    trainData['pca-two'] = pca_result[:,1] 

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    return mm_scaler,pca,trainData

def preprocessTestData(normalizer,pca, testData):
    testData.iloc[:,1:] = normalizer.transform(testData.iloc[:,1:].values)
    pca_result = pca.transform(testData.iloc[:,1:])
    testData['pca-one'] = pca_result[:,0]
    testData['pca-two'] = pca_result[:,1]
    return testData

if __name__ == "__main__":
    trainData = loadDataset(trainDataFile)
    testData = loadDataset(testDataFile)

    # Normalizing the data and using PCA to find top 2 dimensions which explain 
    # the variance in the data  
    normalizer,pca,trainData = preprocessTrainData(trainData)
    testData = preprocessTestData(normalizer,pca,testData)

    # Using unsupervised nearest neighbour
    neigh = NearestNeighbors(2, 0.4)
    neigh.fit(trainData[['pca-one','pca-two']].values) 

    # Find two nearest neighbours for each sample in training data.
    # 1st nearest neighbour obtained would the sample itself.
    # Generate nearest neighbour distance matrix
    (nnDist,_)= neigh.kneighbors(trainData[['pca-one','pca-two']].values, 2, return_distance=True)

    # ASSUMPTION: We assume that training dataset has 1% of outliers and 99% inliers. 
    # (TODO: Find a smarter way to get the percentange of outliers in train dataset).
    #
    # Using Nearest neighbour matrix we find the threshold value within which 99% of datapoints exists.
    # Any datapoint beyond this threshold would be considered as an outlier.
    # This threshold value will be used to identify outliers from the test dataset.
    maxVal = pd.DataFrame(nnDist)[1].quantile([.99]).values 

    (nnDist,_)= neigh.kneighbors(testData[['pca-one','pca-two']].values, 1, return_distance=True)
    nnDistDataFrame = pd.DataFrame(nnDist)

    # Finding outliers in test dataset using the threshold (maxVal)
    testData['score']=((nnDistDataFrame[0] - maxVal[0])/maxVal[0])
    testData['label']=testData['score']>0
    print(f"Found {testData[testData['label']==1].shape[0]} malicious samples out of {testData.shape[0]} samples")
    testData[['event_id','score','label']].to_csv(outputFile,index=False)
    print("Saved the results of test data to output.csv")
