import json
import pandas as pd
from sklearn import preprocessing

def json_to_csv(path_to_json):
    devices = json.load(open(path_to_json, 'rb'))

    products = {}
    features_list = []
    specs = []

    for k in devices['devicesMap']:
        model_map = devices['devicesMap'][k]['modelMap']
        for prod in model_map.keys():
            # products[prod] = model_map[prod]
            #
            # # drop unnecessary keys
            # products[prod].pop('manufacturerMap')
            # products[prod].pop('boosterPoints')
            # products[prod].pop('limitedOfferPresent')
            #
            # products[prod]['size'] = products[prod]['size']['EN']
            # products[prod]['displayName'] = products[prod]['displayName']['EN']

            try:
                for i in range(len(model_map[prod]['skusByMemory'])):
                    for k_skus in model_map[prod]['skusByMemory'][i]:
                        for j in range(len(
                                model_map[prod]['skusByMemory'][i][k_skus])):

                            id = model_map[prod]['skusByMemory'][i][k_skus][j][
                                'id']
                            products[id] = \
                            model_map[prod]['skusByMemory'][i][k_skus][j]

                            to_pop = ['id', 'externalId']

                            for k in to_pop:
                                products[id].pop(k)

                            en_only = ['displayName', 'memoryMap', 'color',
                                       'size',
                                       'featuresList']

                            for k in range(len(en_only)):
                                try:
                                    products[id][en_only[k]] = \
                                    model_map[prod]['skusByMemory'][i][k_skus][
                                        j][en_only[k]]['EN']
                                except:
                                    print(
                                        'No EN attribute for feature {}'.format(
                                            en_only[k]))

                            features_list.append(products[id]['featuresList'])

                            for p in products[id]:
                                if p not in specs:
                                    specs.append(p)
            except:
                print('skusByMemory not in prod {}'.format(prod))

    specs.remove('featureGroupMap')
    specs.remove('promotionsList')
    specs.remove('featuresList')

    feats = []
    for features in features_list:
        for i in range(len(features)):
            feats.append(features[i]['featuretitle'])

    for prod in products:
        all_features = products[prod]['featuresList']
        for feat in all_features:
            products[prod][feat['featuretitle']] = 1

        products[prod].pop('featuresList')
        products[prod].pop('featureGroupMap')

    promo_list = {}
    for prod in products:
        promo_list[prod] = products[prod]['promotionsList']
        products[prod].pop('promotionsList')
        products[prod].pop('marketingContent')

    columns = specs + feats

    df = pd.DataFrame(columns=columns)

    for prod in products:
        df = df.append(products[prod], ignore_index=True)

    df.to_csv('data/devices.csv')


def preprocess_data(df):
    to_drop = ['displayName', 'Unnamed: 0', 'memoryMap', 'marketingContent']
    for drop in to_drop:
        df.drop(drop, axis=1, inplace=True)

    df['memory'] = df['memory'].apply(lambda x: x.strip('GB'))
    df['memory'] = df['memory'].apply(lambda x: x.strip())

    for index, row in df.iterrows():
        if 'mm' in row['memory']:
            df.drop(index, inplace=True)

    feats_to_encode = ['color', 'defaultPricePlanCategory']
    le_dict = {}

    for feat in feats_to_encode:
        le = preprocessing.LabelEncoder()
        xformed_data = le.fit_transform(df[feat])
        df[feat] = xformed_data
        le_dict[feat] = le



    return df
