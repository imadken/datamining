import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from models.Knn_custom import KNN 
from models.Kmeans import K_Means
from models.Dbscan import DBSCAN_custom
from models.dt import DecisionTree
from models.RF import RandomForest
from sklearn.decomposition import PCA
from models.Metrics import *

def preprocess_p2(data,duplicates=True):
    
    data_copy = data.copy(deep=True)
    
    data_copy["P"]=pd.to_numeric(data_copy["P"],errors="coerce")
    data_copy.dropna(inplace=True)
 
    if duplicates:
        data_copy.drop_duplicates(inplace=True)
        # data_copy.drop(columns=["OM","N"],inplace=True)
        data_copy.drop(columns=["OM"],inplace=True)
        
    data_copy.reset_index(inplace=True,drop=True)    
    
    return data_copy

def load_preprocess():
    data = pd.read_csv("data/Dataset1.csv")
    data = preprocess_p2(data)
    return data , data.drop(columns=['Fertility']).columns


def split_scale(data):
    X = np.array(data.drop(columns=['Fertility']))
    y = np.array(data['Fertility']).astype(int)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
    return X,y,X_train,X_test,y_train,y_test , scaler


def init_knn(k):
    knn = KNN(k)
    knn.fit(X_train, y_train)
    return knn


def init_dt(max_depth):
    dt = DecisionTree(max_depth)
    dt.fit(X_train, y_train)
    return dt
def init_rf(max_depth,n_trees,max_feautures):
    rf = RandomForest(n_trees, max_depth,max_feautures)
    rf.fit(X_train, y_train)
    return rf

def init_dbscan(eps,min_samples):
    dbscan = DBSCAN_custom(eps, min_samples)
    dbscan.fit(X)
    return dbscan

def init_kmeans(k):
    kmeans = K_Means(k)
    kmeans.fit(X)
    return kmeans    
        

def classification_report_custom(y_test,predictions):
    knn_eval=[]
    for c in range(3):
        knn_eval.append(["-",precision(y_test,predictions,c),recall(y_test,predictions,c),f1_score(y_test,predictions,c),specificity(y_test,predictions,c),(y_test==c).sum()])
    
    p,r,f,a,s = calculate_metrics(y_test,predictions)
    
    knn_eval.append([a,p,r,f,s,len(y_test)])

    return pd.DataFrame(knn_eval,columns=["Accuracy","Precision","Recall","F1-score","Specificity","Support"],index=["0","1","2","global"])

def get_dbscan_centroids(X, labels):
    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        if label == -1:
            # Skip noise points
            continue

        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)

    return np.array(centroids)

def clustering_report(data, labels, centroids):
    if len(np.unique(labels)) < 2:
        raise ValueError("Clustering did not form meaningful clusters.")
    metrics = [
        silhouette(data, labels),
        # calculate_silhouette_score(data, labels),
        calculate_inter_cluster_distance(data, labels),
        calculate_intra_cluster_distance(data, labels),
        inertia(data, labels, centroids),
        davies_bouldin_index(data, labels, centroids)[0][0]
    ]
    
    # Create a DataFrame with a single row and named columns
    report_df = pd.DataFrame([metrics], columns=["silhouette", "inter_cluster", "intra_cluster", "inertia", "davies"])
    
    return report_df

def scale_input(instance):
    
    instance = np.array([[float(x) for x in instance.split(",")]])
    print(instance)
    scaled_input = scaler.transform(instance)
    return scaled_input

def plot2(X,y,s):
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    
    for label in np.unique(y):
        indices = y == label
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f'Cluster {label}', edgecolors='k')
    
    plt.title(s)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    # plt.show()




data,columns = load_preprocess()
# print(columns)
X,y,X_train,X_test,y_train,y_test,scaler = split_scale(data)

# print(scale_input("136 , 64 ,0.36 ,6,2,2,2,2,2,2,2,2"))

