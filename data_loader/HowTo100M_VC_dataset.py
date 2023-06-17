# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import json
import pandas as pd

from base.base_dataset import TextVideoDataset
try:
    from data_loader.transforms import init_transform_dict, init_video_transform_dict
except:
    from transforms import init_transform_dict, init_video_transform_dict

import torch
from PIL import Image
from torchvision import transforms

class HowTo100MVideoClassification(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'howto100m_classification_train_sorted.csv',
            'val': 'howto100m_classification_val_sorted.csv',
            'test': 'howto100m_classification_val_sorted.csv'
        }
        target_split_fp = split_files[self.split]

        self.metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), sep='\t',error_bad_lines=False)
        self.metadata = self.metadata[self.metadata['label'] < 100]
        if self.split == 'train':
            self.frame_sample = 'rand'
        elif self.split in ['val', 'test']:
            self.frame_sample = 'uniform'

    def _get_video_path(self, sample):
        video_uid = sample['video_uid']
        video_start_sec = max(float(sample['clip_start']), 0)
        video_end_sec   = max(float(sample['clip_end']), 0)

        video_fp = os.path.join(self.data_dir, video_uid)

        video_sec = [video_start_sec, video_end_sec]
        return video_fp, video_sec

    def _get_video_frames(self, video_fp, video_sec, fps):
        video_loading = self.video_params.get('loading', 'strict')
        try:
            if os.path.isfile(video_fp):
                imgs, idxs = self.video_reader(video_fp, self.video_params['num_frames'], self.frame_sample,
                                               start_sec=video_sec[0], end_sec=video_sec[1], fps=fps)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                print('Video loading failed with error: {}... continuing in lax mode...'.format(e))
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

    def _get_stacked_frames(self, sample, num_clips=16):
        # Return stack of frames for each of the video clips. Later we will use this to get stacked features
        clip_duration = min(8.0, sample['clip_end']-sample['clip_start']) #TODO: Move to config
        start_timestamps = [sample['clip_start'] + (i/((num_clips-1)*1.0))*(sample['clip_end'] - sample['clip_start'] - clip_duration - 1.0)  for i in range(num_clips)]
        final_frames = []
        #print('start and end times: {} and {}. Final nums = {}'.format(start_time_index, end_time_index, int((end_time_index-start_time_index)/clip_quantum)))
        for start_time in start_timestamps:
            clip_sample = sample.copy()
            clip_sample['clip_start'] = start_time
            clip_sample['clip_end'] = start_time + clip_duration
            clip_fp, clip_sec = self._get_video_path(clip_sample)
            frames_clip = self._get_video_frames(clip_fp, clip_sec, clip_sample['fps'])
            #print('Frame level for {}, start_time {}: {}'.format(idx, i, frames_clip.shape))
            final_frames.append(frames_clip)
        return torch.stack(final_frames)
        #print('Overall : {}'.format(final.shape))

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        final = self._get_stacked_frames(sample, num_clips=16)

        meta_arr = {'dataset': self.dataset_name}
        return {
            'video': final,
            'meta': meta_arr,
            'label': torch.tensor(int(sample['label']))
        }

    def __len__(self):
        return len(self.metadata)

if __name__ == "__main__":
    kwargs = dict(
        dataset_name="HowTo100M_VC_dataset",
        text_params=None,
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "strict"
        },
        data_dir="/datasets01/HowTo100M/022520/videos/",
        meta_dir="absolute/path/to/dataset/",
        tsfms=init_video_transform_dict()['test'],
        reader='cv2_howto100m',
        split='val',
        neg_param=60
    )
    dataset = HowTo100MVideoClassification(**kwargs)
    for i in range(10):
        item = dataset[i]
