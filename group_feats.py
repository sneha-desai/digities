import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.neighbors import NearestNeighbors

feats = pd.read_csv('data/features.csv', index_col=None)

f_names = list(feats['feature'].values) + ['memory']

all_text = []
for _, row in feats.iterrows():
    all_text.append(row['description'] + row['feature'])

all_text.append('memory')

tokenizer = RegexpTokenizer('[\w]+')
cleaned_text = [" ".join(tokenizer.tokenize(note))
                 for note in all_text]

vectorizer = TfidfVectorizer(stop_words='english')
x = vectorizer.fit_transform(cleaned_text)

nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(x)
# distances, indices = nbrs.kneighbors(x)

# Testing

test_data = vectorizer.transform(['Camera','Dark display', 'Maps', 'Security'])
distances, indices = nbrs.kneighbors(test_data)

import numpy as np
test_features = list(np.asarray(f_names)[indices.flatten()])


# here for now
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

X = pd.read_csv('data/devices_preprocessed.csv')
X = shuffle(X)
target = X['displayName']
X.drop('displayName', axis=1, inplace=True)
noramlized_x = StandardScaler().fit_transform(X)

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(noramlized_x)
test_df = pd.DataFrame(columns=X.columns)
test_df = test_df.append({tf: 1 for tf in test_features}, ignore_index=True)
test_df = test_df.fillna(0)

distances_2, indices_2 = nbrs.kneighbors(test_df.values)




#
# from sklearn.cluster import KMeans
# num_clusters = 10
# kmeans = KMeans(n_clusters=num_clusters).fit(x)
#
# categorization = kmeans.labels_
#
# l = {}
# for i in range(num_clusters):
#     cat = categorization == i
#     l[i] = []
#     for j in range(len(categorization)):
#         if cat[j] == True:
#             l[i].append(f_names[j])
#
print()

#
