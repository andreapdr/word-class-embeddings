import os.path
from sklearn.datasets import get_data_home
import pickle
import numpy as np
from data.labeled import LabelledDocuments
import pandas as pd
import json
import tqdm


def fetch_amazon_pet(data_path=None, subset='train'):
    if data_path is None:
        data_path = os.path.join(get_data_home(), 'amazon_dataset')
    amazon_pet_pickle_path = os.path.join(data_path, 'amazon_pet_supplies_5.' + subset + '.pickle')
    if not os.path.exists(amazon_pet_pickle_path):
        # TODO: should build an helper function to manage downlaod (?)
        with open(data_path + '/Pet_Supplies_5.json', 'rb') as infile:
            json_data = list()
            for i in tqdm.tqdm(range(150000), desc='Loading Dataset'):      # for i in tqdm.tqdm(range(2098324), desc='Loading Dataset'):
                row = infile.readline()
                row = json.loads(row)
                if 'overall' in row.keys() and 'reviewText' in row.keys():
                    _row = {'overall': row['overall'], 'reviewText': row['reviewText']}
                    json_data.append(_row)
                # else:
                    # print(f'Index {i} missing key')

        df = pd.DataFrame(json_data)
        documents = df['reviewText'].values
        labels = df['overall'].values
        del df
        tr_categories = np.unique(labels).astype(int)
        labels = labels - 1     # reindexing target class (score) --> from index 0 to max(score) (5) {0:1, 1:2 ... 4:5}

        def pickle_documents(docs, subset):
            pickle_docs = {'categories': tr_categories, 'documents': docs}
            pickle.dump(pickle_docs, open(os.path.join(data_path, "amazon_pet_supplies_5." + subset + ".pickle"), 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            return pickle_docs
        # TODO: splitting training - validation (?)
        pickle_tr = pickle_documents(list(zip(documents[:100000], labels[:100000])), 'train')
        pickle_te = pickle_documents(list(zip(documents[100000:], labels[100000:])), 'test')
        requested_subset = pickle_tr if subset == 'train' else pickle_te
    else:
        requested_subset = pickle.load(open(amazon_pet_pickle_path, 'rb'))

    text_data, topics = zip(*requested_subset['documents'])
    text_data = list(text_data)
    topics = np.array(topics)
    return LabelledDocuments(data=text_data, target=topics, target_names=requested_subset['categories'])
