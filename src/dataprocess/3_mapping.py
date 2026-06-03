import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

input_data_dir = '../../datastore'
output_data_dir = '../../datastore'
train = pd.read_csv(os.path.join(input_data_dir, 'train_feat_35_final.csv'))[
    ['enroll_id', 'username', 'age', 'gender', 'education', 'cluster_label', 'user_enroll_num', 'course_id',
     'course_enroll_num',
     'course_category', 'truth']]
train['tt_label'] = ['train'] * len(train)
test = pd.read_csv(os.path.join(input_data_dir, 'test_feat_35_final.csv'))[
    ['enroll_id', 'username', 'age', 'gender', 'education', 'cluster_label', 'user_enroll_num', 'course_id',
     'course_enroll_num',
     'course_category', 'truth']]
test['tt_label'] = ['test'] * len(test)

all_log = pd.concat([train, test])
user_feat = all_log[['username', 'age', 'gender', 'education', 'cluster_label', 'user_enroll_num']].drop_duplicates(
    subset=['username'])
unique_user_id = user_feat['username'].unique()
del user_feat['username']
user_std = StandardScaler()
user_feat_std = user_std.fit_transform(user_feat)
user_feat = torch.from_numpy(user_feat_std).to(torch.float)
unique_user_id_df = pd.DataFrame(data={'username': unique_user_id, 'user_graph_id': pd.RangeIndex(len(unique_user_id))})
username_graph_dict = dict()
for i in range(len(unique_user_id_df)):
    username_graph_dict[i] = unique_user_id_df.loc[i, 'username']
all_log = pd.merge(all_log, unique_user_id_df, how='left', on='username')
course_feat = all_log[['course_id', 'course_enroll_num', 'course_category']].drop_duplicates(subset=['course_id'])
unique_course_id = course_feat['course_id'].unique()
del course_feat['course_id']
course_std = StandardScaler()
course_feat_std = course_std.fit_transform(course_feat)
course_feat = torch.from_numpy(course_feat_std).to(torch.float)
unique_course_id_df = pd.DataFrame(
    data={'course_id': unique_course_id, 'course_graph_id': pd.RangeIndex(len(unique_course_id))})
course_graph_dict = dict()
for i in range(len(unique_course_id_df)):
    course_graph_dict[i] = unique_course_id_df.loc[i, 'course_id']
all_log = pd.merge(all_log, unique_course_id_df, how='left', on='course_id')
all_log['record_graph_id'] = list(range(len(all_log)))
all_log = all_log[
    ['record_graph_id', 'enroll_id', 'username', 'user_graph_id', 'course_id', 'course_graph_id', 'truth', 'tt_label']]
all_log = all_log.drop_duplicates(subset=['enroll_id'])
all_log.to_csv(os.path.join(output_data_dir, 'graph_mapping.csv'), index=False)
