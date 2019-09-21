import json

devices = json.load(open('/Users/snehadesai/Documents/techjam/data/devices.json', 'rb'))

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
                    for j in range(len(model_map[prod]['skusByMemory'][i][k_skus])):


                        id = model_map[prod]['skusByMemory'][i][k_skus][j]['id']
                        products[id] = model_map[prod]['skusByMemory'][i][k_skus][j]

                        to_pop = ['id', 'externalId']

                        for k in to_pop:
                            products[id].pop(k)

                        en_only = ['displayName', 'memoryMap', 'color', 'size',
                                   'featuresList']

                        for k in range(len(en_only)):
                            try:
                                products[id][en_only[k]] = model_map[prod]['skusByMemory'][i][k_skus][j][en_only[k]]['EN']
                            except:
                                print('No EN attribute for feature {}'.format(en_only[k]))

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

import pandas as pd

df = pd.DataFrame(columns=columns)

for prod in products:
    df = df.append(products[prod], ignore_index=True)

df.to_csv('devices.csv')






