import os.path
from sklearn.datasets import get_data_home
import pickle
import numpy as np
from data.labeled import LabelledDocuments
import pandas as pd
import json
import tqdm


def fetch_yelp_review(data_path=None, subset='train'):
    if data_path is None:
        data_path = os.path.join(get_data_home(), 'yelp_dataset')
    yelp_review_pickle_path = os.path.join(data_path, 'yelp_review.' + subset + '.pickle')
    if not os.path.exists(yelp_review_pickle_path):
        # TODO: should build an helper function to manage downlaod (?)
        with open(data_path + '/review.json', 'rb') as infile:
            json_data = list()
            # while len(json_data) < 1000000:
            for i in tqdm.tqdm(range(150000), desc='Loading Dataset'):      # for i in tqdm.tqdm(range(2098324), desc='Loading Dataset'):
                row = infile.readline()
                row = json.loads(row)
                if 'stars' in row.keys() and 'text' in row.keys():
                    _row = {'stars': row['stars'], 'text': row['text']}
                    json_data.append(_row)
                # else:
                    # print(f'Index {i} missing key')

        df = pd.DataFrame(json_data)
        documents = df['text'].values
        labels = df['stars'].values
        del df
        del json_data
        tr_categories = np.unique(labels).astype(int)
        labels = labels - 1     # reindexing target class (score) --> index 0 to max(score) (5) {0:1, 1:2 ... 4:5}

        def pickle_documents(docs, subset):
            pickle_docs = {'categories': tr_categories, 'documents': docs}
            pickle.dump(pickle_docs, open(os.path.join(data_path, "yelp_review." + subset + ".pickle"), 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            return pickle_docs
        # TODO: splitting training - validation (?)
        pickle_tr = pickle_documents(list(zip(documents[:100000], labels[:100000])), 'train')
        pickle_te = pickle_documents(list(zip(documents[100000:], labels[100000:])), 'test')
        requested_subset = pickle_tr if subset == 'train' else pickle_te
    else:
        requested_subset = pickle.load(open(yelp_review_pickle_path, 'rb'))

    text_data, topics = zip(*requested_subset['documents'])
    text_data = list(text_data)
    topics = np.array(topics)
    return LabelledDocuments(data=text_data, target=topics, target_names=requested_subset['categories'])
