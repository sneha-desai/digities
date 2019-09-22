import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.neighbors import NearestNeighbors
import numpy as np

def learn():

    feats = pd.read_csv('data/features.csv', index_col=None)

    all_mem = ['16 GB',	'256 GB', '64 GB', '4 GB', '32 GB',	'128 GB', '512 GB']

    f_names = list(feats['feature'].values) + all_mem

    all_text = []
    for _, row in feats.iterrows():
        all_text.append(row['description'] + row['feature'])

    all_text = all_text + all_mem

    tokenizer = RegexpTokenizer('[\w]+')
    cleaned_text = [" ".join(tokenizer.tokenize(note))
                     for note in all_text]

    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(cleaned_text)

    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(x)
    # distances, indices = nbrs.kneighbors(x)

    return vectorizer, nbrs, f_names


def find_all_phones_of_type(type):
    X = pd.read_csv('data/devices_preprocessed.csv')
    sub_df = pd.DataFrame(columns=X.columns)
    for index, row in X.iterrows():
        if type.lower() in row['displayName'].lower():
            sub_df = sub_df.append(row)

    sub_df.drop('displayName', inplace=True, axis=1)
    sub_df.drop('Unnamed: 0', inplace=True, axis=1)

    sum = sub_df.sum()

    sum = np.array(sum).reshape(1, -1)
    for i in range(sum.shape[1]):
        if sum[0, i] > 0:
            sum[0, i] = 1

    sub_df = pd.DataFrame(sum, columns=sub_df.columns)

    return sub_df



def test(test_features, current_phone):
    vectorizer, nbrs, f_names = learn()

    curr_phone_feats = find_all_phones_of_type(current_phone)


    # Testing

    test_data = vectorizer.transform(test_features)
    distances, indices = nbrs.kneighbors(test_data)

    import numpy as np

    top_2_sim = np.asarray(f_names)[indices][:, 0:2]

    test_features = list(np.asarray(f_names)[indices.flatten()])

    # here for now
    from sklearn.utils import shuffle
    from sklearn.preprocessing import StandardScaler

    X = pd.read_csv('data/devices_preprocessed.csv')
    X = shuffle(X)
    pre_data = X.copy()
    target = X['displayName']
    X.drop('displayName', inplace=True, axis=1)
    X.drop('Unnamed: 0', inplace=True, axis=1)
    # noramlized_x = StandardScaler().fit_transform(X)
    #
    # nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(
    #     noramlized_x)
    test_df = pd.DataFrame(columns=X.columns)
    test_df = test_df.append({tf: 1 for tf in test_features}, ignore_index=True)
    test_df = test_df.fillna(0)


    test_feats = test_df.values + curr_phone_feats.values

    test_feats = np.array(test_feats).reshape(1, -1)
    for i in range(test_feats.shape[1]):
        if test_feats[0, i] > 0:
            test_feats[0, i] = 1

    # distances_2, indices_2 = nbrs.kneighbors(test_feats)

    from sklearn.metrics.pairwise import cosine_similarity

    cs = []

    for index, row in X.iterrows():
        cs.append(cosine_similarity(X.values[index].reshape(1, -1), test_feats))



    return list(np.asarray(target)[indices_2.flatten()]), top_2_sim


khadija = ['streaming platforms', 'Bluetooth', 'phone calls']
chad = ['social media', 'camera', 'online shopping']

recommended_phones_khadija = test(khadija, 'iphone')
recommended_phones_chad = test(chad, 'Samsung')

print()

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

#
