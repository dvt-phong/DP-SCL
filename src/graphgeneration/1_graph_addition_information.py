import os
import numpy as np
from tqdm import tqdm
import pickle as pkl
import warnings
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dir = '../../datastore'
graph_dir = '../../datastore'

with open(os.path.join(graph_dir, 'StrongClassmatesGraph.pkl'), 'rb') as f:
    graph = pkl.load(f)
train = pd.read_csv(os.path.join(input_dir, 'train_feat_35_final.csv'))
test = pd.read_csv(os.path.join(input_dir, 'test_feat_35_final.csv'))
all_log = pd.concat([train, test])

mapping_graph_log = pd.read_csv(os.path.join(input_dir, 'graph_mapping.csv'))

all_log = pd.merge(all_log, mapping_graph_log, how='left', on='enroll_id')
all_log = all_log.sort_values('record_graph_id')
del all_log['username_y']
del all_log['course_id_y']

user_feat = all_log[
    ['user_graph_id', 'age', 'gender', 'education', 'cluster_label', 'user_enroll_num']].drop_duplicates(
    subset=['user_graph_id'])
del user_feat['user_graph_id']
user_std = StandardScaler()
user_feat_std = user_std.fit_transform(user_feat)
enroll_user_feat = torch.tensor([user_feat_std[int(user_graph_id)] for user_graph_id in all_log['user_graph_id']])
course_feat = all_log[['course_graph_id', 'course_enroll_num', 'course_category']].drop_duplicates(
    subset=['course_graph_id'])
del course_feat['course_graph_id']

course_std = StandardScaler()
course_feat = course_std.fit_transform(course_feat.values)

user_select_courses_mean_df = all_log.groupby('user_graph_id')
user_course_feat_dict = dict()
for data in tqdm(user_select_courses_mean_df):
    cur_username = data[0]
    course_select_list = list(data[1]['course_graph_id'])
    course_feat_cur_user = np.array([course_feat[int(course_graph_id)] for course_graph_id in course_select_list])
    course_feat_cur_user = np.mean(course_feat_cur_user, axis=0)
    user_course_feat_dict[int(cur_username)] = course_feat_cur_user
user_course_feat_matrix = torch.tensor([user_course_feat_dict[int(username)] for username in all_log['user_graph_id']])
user_course_org_feat = torch.cat((enroll_user_feat, user_course_feat_matrix), dim=1).to(torch.float)
graph.org_context = user_course_org_feat
sta_day = 35
video_action = ['seek_video', 'play_video', 'pause_video', 'stop_video', 'load_video']
problem_action = ['problem_get', 'problem_check', 'problem_save', 'reset_problem', 'problem_check_correct',
                  'problem_check_incorrect']
forum_action = ['create_thread', 'create_comment', 'delete_thread', 'delete_comment', 'close_forum']
click_action = ['click_info', 'click_courseware', 'click_about', 'click_forum', 'click_progress']
close_action = ['close_courseware']
all_action = video_action + problem_action + forum_action + click_action + close_action
all_action_feat = []
for i in range(sta_day):
    for action in all_action:
        all_action_feat.append(str(action) + '_' + str(i) + '#num')
seq_feat = torch.from_numpy(all_log[all_action_feat].values).to(torch.float)
seq_feat = seq_feat.view(len(all_log), sta_day, -1)
graph.seq_feat = seq_feat

with open(os.path.join(input_dir, 'StrongClassmatesGraph.pkl'), 'wb') as f:
    pkl.dump(graph, f)
