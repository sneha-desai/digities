from sklearn.decomposition import PCA
import pandas as pd
from extract_data import preprocess_data

X = pd.read_csv('data/devices.csv')

X = preprocess_data(X)

pca = PCA(n_components=2)
pca.fit(X.values)



