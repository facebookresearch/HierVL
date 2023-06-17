# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


'''
For EPIC-KITCHENS MIR, loading the caption relevancy causes issue with DDP training
So, save every row of that matrix as a separate pickle file
'''

import pickle
from tqdm import tqdm
import os

path_relevancy = '/path/to/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_train.pkl'
save_folder = '/path/to/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_train/'

pkl_file = open(path_relevancy, 'rb')
relevancy_mat = pickle.load(pkl_file)

for i in tqdm(range(relevancy_mat.shape[0])):
    with open(os.path.join(save_folder, '{}.pkl'.format(i)),'wb') as f:
        pickle.dump(relevancy_mat[i], f)
