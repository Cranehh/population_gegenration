from torch.utils.data import Dataset
import torch
import bisect
from mmengine import fileio
import io
import os
import json
import numpy as np
def opendata(path):
    
    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    npz_data = np.load(buff, allow_pickle=True)

    return npz_data

class VAETrainingData(Dataset):
    def __init__(self, family_data_dir, person_data_dir, data_len_list):
        self.family_data_dir = family_data_dir
        self.individual_data_dir = person_data_dir
        self.data_len_list = data_len_list
        self.prefix = self.data_len_list.cumsum()
        

    def __len__(self):
        return self.data_len_list.shape[0]

    def __getitem__(self, idx):
        # idx += 1
        # file_index = bisect.bisect_left(self.prefix, idx)
        # pos_in_file = idx - (self.prefix[file_index] - self.data_len_list[file_index]) - 1
        # family_data = opendata(self.family_data_dir[file_index])[pos_in_file][:10]
        # individual_data = opendata(self.individual_data_dir[file_index])[pos_in_file]


        family_data = opendata(self.family_data_dir[idx])[:, :10]
        individual_data = opendata(self.individual_data_dir[idx])


        ## 原代码
        # data = {
        #     "ego_current_state": ego_current_state,
        #     "ego_future_gt": ego_agent_future,
        #     "neighbor_agents_past": neighbor_agents_past,
        #     "neighbors_future_gt": neighbor_agents_future,
        #     "lanes": lanes,
        #     "lanes_speed_limit": lanes_speed_limit,
        #     "lanes_has_speed_limit": lanes_has_speed_limit,
        #     "route_lanes": route_lanes,
        #     "route_lanes_speed_limit": route_lanes_speed_limit,
        #     "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
        #     "static_objects": static_objects,
        # }

        return {
            'family': family_data,
            'member': individual_data
        }