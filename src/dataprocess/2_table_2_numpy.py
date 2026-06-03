import os
import pandas as pd
import numpy as np
from tqdm import tqdm

input_data_dir = '../../datastore'
output_data_dir = '../../datastore'

sta_day = 35
video_action = ['seek_video', 'play_video', 'pause_video', 'stop_video', 'load_video']
problem_action = ['problem_get', 'problem_check', 'problem_save', 'reset_problem', 'problem_check_correct',
                  'problem_check_incorrect']
forum_action = ['create_thread', 'create_comment', 'delete_thread', 'delete_comment', 'close_forum']
click_action = ['click_info', 'click_courseware', 'click_about', 'click_forum', 'click_progress']
close_action = ['close_courseware']
all_action = video_action + problem_action + forum_action + click_action + close_action

all_action_feat = []
for action in all_action:
    for i in range(sta_day):
        all_action_feat.append(str(action) + '_' + str(i) + '#num')

context_feat = ['age', 'gender', 'education', 'user_enroll_num', 'course_enroll_num', 'course_category']
train_df = pd.read_csv(os.path.join(input_data_dir, 'train_feat_35_final.csv'))[all_action_feat]
train_context = pd.read_csv(os.path.join(input_data_dir, 'train_feat_35_final.csv'))[context_feat].to_numpy()
train_truth = pd.read_csv(os.path.join(input_data_dir, 'train_feat_35_final.csv'))['truth'].to_numpy()
test_df = pd.read_csv(os.path.join(input_data_dir, 'test_feat_35_final.csv'))[all_action_feat]
test_context = pd.read_csv(os.path.join(input_data_dir, 'test_feat_35_final.csv'))[context_feat].to_numpy()
test_truth = pd.read_csv(os.path.join(input_data_dir, 'test_feat_35_final.csv'))['truth'].to_numpy()

train_data = None
test_data = None
count = 0
for i in tqdm(range(sta_day)):
    for action in tqdm(all_action):
        cur_name = str(action) + '_' + str(i) + '#num'
        if count == 0:
            train_data = train_df[cur_name].to_numpy().reshape(len(train_df), -1)
            test_data = test_df[cur_name].to_numpy().reshape(len(test_df), -1)
            count = count + 1
        else:
            train_data = np.concatenate((train_data, train_df[cur_name].to_numpy().reshape(len(train_df), -1)), axis=1)
            test_data = np.concatenate((test_data, test_df[cur_name].to_numpy().reshape(len(test_df), -1)), axis=1)

train_data = np.reshape(train_data, newshape=(len(train_data), 5, 7, 22))
test_data = np.reshape(test_data, newshape=(len(test_data), 5, 7, 22))
np.savez(os.path.join(output_data_dir, 'all_data_std'), t_data=train_data, t_label=train_truth, t_context=train_context,
         v_data=test_data,
         v_label=test_truth, v_context=test_context)

