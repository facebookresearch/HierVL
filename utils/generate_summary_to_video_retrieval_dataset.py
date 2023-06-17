# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import pandas as pd


data_dir = 'absolute/path/to/ego4d_chunked/'

# Find training videos to remove
train_data = pd.read_csv('../dataset/egoclip.csv', sep='\t',error_bad_lines=False)
train_vids = list(train_data.video_uid.unique())
summary_train_data = pd.read_csv('../dataset/egosummary_full.csv', sep='\t',error_bad_lines=False)
summary_train_vids = list(summary_train_data.video_uid.unique())

# print(len(train_vids))
# print(len(summary_train_vids))
overall_training_samples = list(set(train_vids).union(set(summary_train_vids)))
# print(len(overall_training_samples))

overall_samples = os.listdir(data_dir)
val_videos = list(set(overall_samples).difference(set(overall_training_samples)))
print(len(val_videos))

# Now once we get val_videos we can create the val dataset
with open('data/Ego4D/all_narrations_220802.json') as f:
    ego4d_full_narrations = json.load(f)
from tqdm import tqdm
overall_summaries = []
final_dataset = {}
for vid in tqdm(val_videos):
    #narration_passes = [key for key in ego4d_narrations[vid] if key != 'status']
    if vid in ego4d_full_narrations['videos']:
        narration = ego4d_full_narrations['videos'][vid]
        summaries = narration.get('summaries', None)
        if summaries is not None:
            for summary in summaries:
                summary_dict = {
                    "video_uid": vid,
                    "video_dur": None, # Not needed
                    "narration_source": 'raw',
                    "narration_ind": None, # Not needed
                    "narration_time": None, # Not needed
                    "clip_start": summary['start_time'],
                    "clip_end": summary['end_time'],
                    "clip_text": summary['text'],
                    "tag_verb": "[]",
                    "tag_noun": "[]"
                }
                overall_summaries.append(summary_dict)
        else:
            'Empty summaries...'
    else:
        print('Video not found: {}'.format(vid))
print('Dataset length: {}'.format(len(overall_summaries)))
from random import choice
import random
num_incorrects = 4
for idxy, datum in enumerate(overall_summaries):
    all_choices = [None] * (num_incorrects + 1)
    answer = random.randint(0, 4)
    all_choices[answer] = datum
    already_chosen_indices = [idxy]
    curr_iter = 0
    while len(already_chosen_indices) < num_incorrects + 1:
        idx_choice = choice([i for i in range(len(overall_summaries)) if i not in already_chosen_indices])
        if overall_summaries[idx_choice]['video_uid'] != datum['video_uid']:
            already_chosen_indices.append(idx_choice)
            while all_choices[curr_iter] is not None:
                curr_iter += 1 
            all_choices[curr_iter] = overall_summaries[idx_choice]
    all_choices_dict = {}
    for option_idx in range(len(all_choices)):
        if all_choices[option_idx] is None:
            raise ValueError
        all_choices_dict[str(option_idx)] = all_choices[option_idx]
    final_dataset[str(idxy)] = {
        'query': datum,
        'answer': answer,
        'choices': all_choices_dict,
        'types': 3
    }

print('Final data length: {}'.format(len(final_dataset)))

with open('summarymcq_full.json', 'w') as fp:
    json.dump(final_dataset, fp,  indent=4)
