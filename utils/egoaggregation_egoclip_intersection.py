# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import pandas as pd
from tqdm import tqdm
import json

egosummary = pd.read_csv('../dataset/egosummary_full.csv', sep='\t', error_bad_lines=False)
egoclip = pd.read_csv('../dataset/egoclip.csv', sep='\t',error_bad_lines=False)

collection_dict = {}
for i in tqdm(range(len(egosummary))):
    continue
    sample = egosummary.iloc[i]
    video_uid = sample['video_uid']
    summary_source = sample['narration_source']
    summary_start = sample['clip_start']
    summary_end = sample['clip_end']
    # print(video_uid)
    clip_df = egoclip[(egoclip['video_uid'] == video_uid) & (egoclip['clip_start'] >= summary_start) & (egoclip['clip_end'] <= summary_end) & (egoclip['narration_source'] == summary_source)]
    clip_df = clip_df.sort_values('clip_start')
    narrations_dict = {'summary_start': sample['clip_start'], 'summary_end': sample['clip_end'], 'clip_start': [], 'clip_end': [], 'clip_text': [], 'tag_verb': [], 'tag_noun': []}
    for j in range(len(clip_df)):
        narrations_dict['clip_start'].append(clip_df.iloc[j]['clip_start'])
        narrations_dict['clip_end'].append(clip_df.iloc[j]['clip_end'])
        narrations_dict['clip_text'].append(clip_df.iloc[j]['clip_text'])
        narrations_dict['tag_verb'].append(clip_df.iloc[j]['tag_verb'])
        narrations_dict['tag_noun'].append(clip_df.iloc[j]['tag_noun'])
    collection_dict[sample['clip_text']] = narrations_dict

def get_result_dict(i):
    #print('Doing for idx: {}'.format(i))
    sample = egosummary.iloc[i]
    video_uid = sample['video_uid']
    summary_start = sample['clip_start']
    summary_end = sample['clip_end']
    # print(video_uid)
    clip_df = egoclip[(egoclip['video_uid'] == video_uid) & (egoclip['clip_start'] >= summary_start) & (egoclip['clip_end'] <= summary_end)]
    clip_df = clip_df.sort_values('clip_start')
    summary_start = int(1000*sample['clip_start'])/1000.0
    summary_end = int(1000*sample['clip_end'])/1000.0
    narrations_dict = {'summary_start': summary_start, 'summary_end': summary_end, 'clip_start': [], 'clip_end': [], 'clip_text': [], 'tag_verb': [], 'tag_noun': []}
    for j in range(len(clip_df)):
        clip_sample = clip_df.iloc[j]
        clip_start = int(1000*clip_sample['clip_start'])/1000.0
        clip_end = int(1000*clip_sample['clip_end'])/1000.0
        narrations_dict['clip_start'].append(clip_start)
        narrations_dict['clip_end'].append(clip_end)
        narrations_dict['clip_text'].append(clip_sample['clip_text'])
        narrations_dict['tag_verb'].append(clip_sample['tag_verb'])
        narrations_dict['tag_noun'].append(clip_sample['tag_noun'])
    #collection_dict[sample['clip_text']] = narrations_dict
    fin_dict = {'idx': i, '{}'.format(sample['clip_text']): narrations_dict}
    with open('fast_results_2/{}.json'.format(i), 'w') as outfile:
        json.dump(fin_dict, outfile)
    return (i, sample['clip_text'], narrations_dict)
'''
from concurrent.futures import ThreadPoolExecutor
inputs = [i for i in range(len(egosummary))]
with ThreadPoolExecutor() as executor:
    future_results = executor.map(get_result_dict, inputs)
    results = [result for result in future_results]
'''
import time
import multiprocessing as mp


results_pool = []
pool = mp.Pool()
for i in tqdm(range(len(egosummary))):
    results_p = pool.apply_async(get_result_dict, args= (i,))#pool.apply_async(get_result_dict, args=(i)) for i in range(100)
    #results_pool.append(results_p.get())
# results = [p.get() for p in results_p]
pool.close()
pool.join()
exit()
ordered_results = [None] * len(egosummary)
for result in results_pool:
    ordered_results[result[0]] = (result[1], result[2])
for ordered_result in ordered_results:
    collection_dict[ordered_result[0]] = ordered_result[1]

import json
with open('summary_clips_hierarchy_full.json', 'w') as fp:
    json.dump(collection_dict, fp)

exit()
# The above process is quite slow, to parallelize, save it as different files using the following
import os
collection_dict = {}
with os.scandir('fast_results_2/') as f:
    for entry in tqdm(f):
        with open('fast_results_2/{}'.format(entry.name)) as f:
            data = json.load(f)
            collection_dict[list(data.keys())[1]] = data[list(data.keys())[1]]

with open('summary_clips_hierarchy_full.json', 'w') as fp:
    json.dump(collection_dict, fp)
