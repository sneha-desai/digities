from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.neighbors import NearestNeighbors

X = pd.read_csv('data/devices_preprocessed.csv')
X = shuffle(X)
target = X['displayName']
X.drop('displayName', axis=1, inplace=True)
X = StandardScaler().fit_transform(X)

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X)



# pca = PCA(n_components=2)
# principal_components = pca.fit_transform(X)
#
# principal_df = pd.DataFrame(data=principal_components,
#                             columns=['principal component 1',
#                                      'principal component 2'])
# principal_df.reset_index(inplace=True)
# target = pd.DataFrame(target)
# target.reset_index(inplace=True)
#
# final_df = pd.concat([principal_df, target], axis=1)
# final_df = shuffle(final_df)
# # final_df = final_df[0:5]
# final_df.drop('index', axis=1, inplace=True)
#
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# ax.scatter(final_df['principal component 1'],
#            final_df['principal component 2'], s=50)
# for i, t in enumerate(final_df['displayName']):
#     ax.annotate(t, (final_df['principal component 1'].iloc[i],
#                     final_df['principal component 2'].iloc[i]))
#
# plt.show()

