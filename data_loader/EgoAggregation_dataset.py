# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import math
import random
import pandas as pd

from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict

import torch
from PIL import Image
from torchvision import transforms

class EgoAggregation(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'egosummary_full.csv',
            'val': 'summarymcq_full.json',
            'test': 'egomcq.json'
        }
        target_split_fp = split_files[self.split]

        self.chunk_sec = 600  # Each segment is up to 600s
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary

        if self.split == 'train':
            self.metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), sep='\t',error_bad_lines=False)
            self.frame_sample = 'rand'

            load_all_once = True # load all summary hierarchy at once, maybe high on CPU usage, set to False for dynamic loading
            if load_all_once:
                with open(os.path.join(self.meta_dir, 'summary_clips_hierarchy_full.json')) as f:
                    self.summarry_narration_hierarchy = json.load(f)
            else:
                self.summarry_narration_hierarchy = None

            if self.neg_param:
                self.metadata['chunk_id'] = self.metadata['narration_time'] // self.neg_param
                self.metadata['chunk_id'] = self.metadata['chunk_id'].astype(str)
                self.metadata['segment_id'] = self.metadata['video_uid'] + '_' + self.metadata['chunk_id']

        elif self.split in ['val', 'test']:
            self.frame_sample = 'uniform'
            with open(os.path.join(self.meta_dir, target_split_fp), 'r') as load_f:
                self.metadata = json.load(load_f)

    def _get_video_path(self, sample):
        video_uid = sample['video_uid']
        video_start_sec = max(float(sample['clip_start']), 0)
        video_end_sec   = max(float(sample['clip_end']), 0)

        chunk_start_id = int(video_start_sec // self.chunk_sec)
        chunk_end_id = int(video_end_sec // self.chunk_sec)

        full_video_start_fp = os.path.join(self.data_dir, video_uid, str(chunk_start_id) + ".mp4")
        full_video_end_fp = os.path.join(self.data_dir, video_uid, str(chunk_end_id) + ".mp4")

        video_fp = [full_video_start_fp, full_video_end_fp]
        video_sec = [video_start_sec, video_end_sec]
        bound_sec = (chunk_start_id + 1) * self.chunk_sec
        return video_fp, video_sec, bound_sec

    def _get_video_frames(self, video_fp, video_sec, bound_sec):
        video_loading = self.video_params.get('loading', 'strict')
        try:
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                imgs, idxs = self.video_reader(video_fp[0], video_fp[1], self.video_params['num_frames'], self.frame_sample,
                                               start_sec=video_sec[0], end_sec=video_sec[1], bound_sec=bound_sec)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        return final

    def _get_caption(self, sample):
        noun_vec = torch.zeros(self.noun_dim)
        verb_vec = torch.zeros(self.verb_dim)
        noun_idx = eval(sample['tag_noun'])
        verb_idx = eval(sample['tag_verb'])
        for i in noun_idx:
            noun_vec[i] = 1
        for i in verb_idx:
            verb_vec[i] = 1

        return sample['clip_text'], noun_vec, verb_vec

    def _get_stacked_caption(self, sample, index=-1, num_text_samples=16):
        noun_vec = torch.zeros(num_text_samples, self.noun_dim)
        verb_vec = torch.zeros(num_text_samples, self.verb_dim)
        clip_texts = []

        if self.summarry_narration_hierarchy is not None:
            datum = self.summarry_narration_hierarchy[sample['clip_text']]
        else:
            with open('utils/fast_results_2/{}.json'.format(index)) as f:
                idx_dict = json.load(f)
            datum = idx_dict[list(idx_dict.keys())[1]]
        # print(datum.keys()) #['summary_start', 'summary_end', 'clip_start', 'clip_end', 'clip_text', 'tag_verb', 'tag_noun'] 

        # If there are no narrations in the summary duration
        if len(datum['clip_text']) == 0:
            # If there are no text in the summary duration, simply take the summary and duplicate it
            for idx in range(num_text_samples):
                caption_summary, noun_vec_summary, verb_vec_summary = self._get_caption(sample)
                clip_texts.append(caption_summary)
                noun_vec[idx, :] = noun_vec_summary
                verb_vec[idx, :] = verb_vec_summary
            return clip_texts, noun_vec, verb_vec

        # RANDOM SAMPLING WHILE PRESERVING ORDER
        try:
            #TODO: Check if we need to send all the samples rather than sampling a few from the list
            indices = sorted(random.sample(range(0, len(datum['clip_text'])), num_text_samples))
        except Exception as e:
            # If too few samples, just iterate over them
            indices = [x%len(datum['clip_text']) for x in range(num_text_samples)]
        for iter_, idx in enumerate(indices):
            try:
                noun_idx = eval(datum['tag_noun'][idx])
                verb_idx = eval(datum['tag_verb'][idx])
            except:
                print('Various lengths: {} {} and {}'.format(len(datum['clip_text']), len(datum['tag_noun']), len(datum['tag_verb'])))
                assert False
            for i in noun_idx:
                noun_vec[iter_][i] = 1
            for i in verb_idx:
                verb_vec[iter_][i] = 1
            clip_texts.append(datum['clip_text'][idx])

        #noun_vec = (torch.mean(noun_vec, dim=0) > 0.5).float()
        #verb_vec = (torch.mean(verb_vec, dim=0) > 0.5).float()

        return clip_texts, noun_vec, verb_vec

    def _get_stacked_frames(self, sample, num_clips=16):
        # Return stack of frames for each of the video clips. Later we will use this to get stacked features
        #clip_quantum = 16.0
        clip_duration = min(8.0, sample['clip_end']-sample['clip_start']) #TODO: Move to config
        start_timestamps = [sample['clip_start'] + (i/((num_clips-1)*1.0))*(sample['clip_end'] - sample['clip_start'] - clip_duration - 1.0)  for i in range(num_clips)]
        final_frames = []
        #print('start and end times: {} and {}. Final nums = {}'.format(start_time_index, end_time_index, int((end_time_index-start_time_index)/clip_quantum)))
        for start_time in start_timestamps:
            clip_sample = sample.copy()
            clip_sample['clip_start'] = start_time
            clip_sample['clip_end'] = start_time + clip_duration
            clip_fp, clip_sec, clip_bound_sec = self._get_video_path(clip_sample)
            frames_clip = self._get_video_frames(clip_fp, clip_sec, clip_bound_sec)
            #print('Frame level for {}, start_time {}: {}'.format(idx, i, frames_clip.shape))
            final_frames.append(frames_clip)
        return torch.stack(final_frames)
        #print('Overall : {}'.format(final.shape))

    def _get_train_item(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        final = self._get_stacked_frames(sample)
        #video_fp, video_sec, bound_sec = self._get_video_path(sample)
        #final = self._get_video_frames(video_fp, video_sec, bound_sec)

        # Summary Text
        caption, noun_vec, verb_vec = self._get_caption(sample)
        # Text aggregation
        try:
            aggregated_caption, aggregated_noun_vec, aggregated_verb_vec = self._get_stacked_caption(sample, index=item)
        except Exception as e:
            #print('Error in text aggregation: {}. Text length: {}'.format(e, len(self.summarry_narration_hierarchy[sample['clip_text']]["clip_text"])))
            print('Error in text aggregation: {}.'.format(e))
            aggregated_caption, aggregated_noun_vec, aggregated_verb_vec = self._get_caption(sample)

        #final = final.unsqueeze(0)

        # Scene-aware negative sampling
        if self.neg_param:
            print('Not implemented. Summaries should sample negatives from the batch itself.')
            raise ValueError

        meta_arr = {'raw_captions': caption, 'paths': None, 'dataset': self.dataset_name}
        return {
            'video': final,
            'text': caption, #Always returns the summary text features
            'aggregated_text': aggregated_caption, #Always returns the stacked clip text features
            'meta': meta_arr,
            'noun_vec': noun_vec,
            'verb_vec': verb_vec,
            'aggregated_noun_vec': aggregated_noun_vec,
            'aggregated_verb_vec': aggregated_verb_vec
        }

    def _get_val_item(self, item):
        item = item % len(self.metadata)
        itemMCQ = self.metadata[str(item)]
        num_video_clips = 16 # TODO: Move this to config

        answerIndex = itemMCQ['answer']
        sampleQuery = itemMCQ['query']
        textQuery, _, _ = self._get_caption(sampleQuery)

        sampleOptions = itemMCQ['choices']
        num_options = len(sampleOptions)
        textOptions = []
        videoOptions = torch.zeros([num_options, num_video_clips, self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])

        for id, option in enumerate(sampleOptions):
            sampleOptioni = sampleOptions[option]
            #video_fp, video_sec, bound_sec = self._get_video_path(sampleOptioni)
            caption, _, _ = self._get_caption(sampleOptioni)
            textOptions.append(caption)

            #imgs = self._get_video_frames(video_fp, video_sec, bound_sec)
            imgs = self._get_stacked_frames(sampleOptioni)
            videoOptions[id] = imgs

        type =  itemMCQ['types']    # 1 for inter; 2 for intra
        data = {'video': videoOptions, 'text': textQuery, 'text_ops':textOptions, 'correct': answerIndex, 'type': type}
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        if self.split == 'train':
            return self._get_train_item(item)
        elif self.split in ['val', 'test']:
            return self._get_val_item(item)
        else:
            raise ValueError

if __name__ == "__main__":
    kwargs = dict(
        dataset_name="EgoClip_dataset",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="video_dataset/ego4d_256/data_chunked",
        meta_dir="video_dataset/ego4d_toolbox/0_metadata/egovlp",
        tsfms=init_video_transform_dict()['test'],
        reader='cv2_egoclip',
        split='val',
        neg_param=60
    )
    dataset = EgoAggregation(**kwargs)
    for i in range(100):
        item = dataset[i]
