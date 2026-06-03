import os
import pandas as pd
from tqdm import tqdm
import gc

input_data_dir = '../../datastore'
output_data_dir = '../../datastore'
sta_day = 35

train = pd.read_csv(os.path.join(input_data_dir, 'prediction_log/train_log.csv'))[
    ['enroll_id', 'username', 'course_id', 'action', 'time']]
train_truth = pd.read_csv(os.path.join(input_data_dir, 'prediction_log/train_truth.csv'), index_col='enroll_id')
test = pd.read_csv(os.path.join(input_data_dir, 'prediction_log/test_log.csv'))[
    ['enroll_id', 'username', 'course_id', 'action', 'time']]
test_truth = pd.read_csv(os.path.join(input_data_dir, 'prediction_log/test_truth.csv'), index_col='enroll_id')
all_truth = pd.concat([train_truth, test_truth])
all_log = pd.concat([train, test])

course_info = pd.read_csv(os.path.join(input_data_dir, 'course_info.csv'))
course_info['start'] = pd.to_datetime(course_info['start'])
all_log['time'] = pd.to_datetime(all_log['time'])
all_log = pd.merge(all_log, course_info[['course_id', 'start']], how='left', on='course_id')
all_log['time_diff'] = (all_log['time'] - all_log['start']).dt.days

video_action = ['seek_video', 'play_video', 'pause_video', 'stop_video', 'load_video']
problem_action = ['problem_get', 'problem_check', 'problem_save', 'reset_problem', 'problem_check_correct',
                  'problem_check_incorrect']
forum_action = ['create_thread', 'create_comment', 'delete_thread', 'delete_comment', 'close_forum']
click_action = ['click_info', 'click_courseware', 'click_about', 'click_forum', 'click_progress']
close_action = ['close_courseware']
all_action = video_action + problem_action + forum_action + click_action + close_action

cal_count = 0
all_num = None
all_action_feat = []
for action in tqdm(all_action):
    for i in tqdm(range(sta_day)):
        action_day_ = ((all_log['action'] == action) & (all_log['time_diff'] == i)).astype(int)
        all_log[str(action) + '_' + str(i) + '#num'] = action_day_
        action_day_num = all_log.groupby('enroll_id')[[str(action) + '_' + str(i) + '#num']].sum()
        if cal_count == 0:
            cal_count += 1
            all_num = action_day_num
        else:
            cal_count += 1
            all_num = pd.merge(all_num, action_day_num, left_index=True, right_index=True)
        del action_day_num
        del action_day_
        del all_log[str(action) + '_' + str(i) + '#num']
        gc.collect()
        all_action_feat.append(str(action) + '_' + str(i) + '#num')

all_num = pd.merge(all_num, all_truth, left_index=True, right_index=True)
enroll_info = all_log[['username', 'course_id', 'enroll_id']].drop_duplicates()
enroll_info.index = enroll_info['enroll_id']
del enroll_info['enroll_id']
all_num = pd.merge(all_num, enroll_info, left_index=True, right_index=True)
train_enroll = list(set(list(train['enroll_id'])))
test_enroll = list(set(list(test['enroll_id'])))
all_num.loc[test_enroll].to_csv(os.path.join(output_data_dir, 'test_features_35.csv'))
all_num.loc[train_enroll].to_csv(os.path.join(output_data_dir, 'train_features_35.csv'))

