"""
Origin and attribution:
  Project: DP-SCL.
  Purpose: Add XuetangX/KDD Cup contextual features to the 35-day activity
  tables before converting them to NumPy tensors.

Reference source:
  Adapted from CA-TFHN by codeds27:
  https://github.com/codeds27/CA-TFHN
  Original file:
  https://github.com/codeds27/CA-TFHN/blob/main/src/dataprocess/1_process_user_contextual_features.py

Adaptation notes:
  The demographic/course-count/category feature pipeline follows CA-TFHN.
  Local DP-SCL changes include adding cluster_label loading from
  datastore/cluster and importing NumPy for that step.
"""

import math
import os
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

input_data_dir = '../../datastore'
output_data_dir = '../../datastore'
train_feat = pd.read_csv(os.path.join(input_data_dir, 'train_features_35.csv'), index_col=0)
test_feat = pd.read_csv(os.path.join(input_data_dir, 'test_features_35.csv'), index_col=0)
all_feat = pd.concat([train_feat, test_feat])

user_profile = pd.read_csv(os.path.join(input_data_dir, 'user_info.csv'), index_col='user_id')
birth_year = user_profile['birth'].to_dict()


def age_convert(y):
    if y == None or math.isnan(y):
        return 0
    a = 2023 - int(y)
    if a > 70 or a < 10:
        a = 0
    return a


all_feat['age'] = [age_convert(birth_year.get(int(u), None)) for u in all_feat['username']]
user_gender = user_profile['gender'].to_dict()


def gender_convert(g):
    if g == 'm':
        return 1
    elif g == 'f':
        return 2
    else:
        return 0


all_feat['gender'] = [gender_convert(user_gender.get(int(u), None)) for u in all_feat['username']]
user_edu = user_profile['education'].to_dict()


def edu_convert(x):
    edus = ["Bachelor's", "High", "Master's", "Primary", "Middle", "Associate", "Doctorate"]
    if not isinstance(x, str):
        return 0
    ii = edus.index(x)
    return ii + 1


all_feat['education'] = [edu_convert(user_edu.get(int(u), None)) for u in all_feat['username']]
# Load cluster labels
cluster_label = np.load(os.path.join(input_data_dir, 'cluster/label_5_10time.npy'), allow_pickle=True)
with open(os.path.join(input_data_dir, 'cluster/user_dict'), 'rb') as f:
    user_dict = pkl.load(f, encoding='latin1')
# Map username to cluster label
def get_cluster_label(username):
    idx = user_dict.get(int(username), None)
    if idx is not None and idx < len(cluster_label):
        return int(cluster_label[idx])
    return 0
all_feat['cluster_label'] = [get_cluster_label(u) for u in all_feat['username']]

user_enroll_num = all_feat.groupby('username').count()[['course_id']]
course_enroll_num = all_feat.groupby('course_id').count()[['username']]
user_enroll_num.columns = ['user_enroll_num']
course_enroll_num.columns = ['course_enroll_num']
all_feat = pd.merge(all_feat, user_enroll_num, left_on='username', right_index=True)
all_feat = pd.merge(all_feat, course_enroll_num, left_on='course_id', right_index=True)

courseinfo = pd.read_csv(os.path.join(input_data_dir, 'course_info.csv'), index_col='id')
en_categorys = ['math', 'physics', 'electrical', 'computer', 'foreign language', 'business', 'economics', 'biology',
                'medicine', 'literature', 'philosophy', 'history', 'social science', 'art', 'engineering', 'education',
                'environment', 'chemistry']


def category_convert(cc):
    if isinstance(cc, str):
        print(cc)
        for i, c in zip(range(len(en_categorys)), en_categorys):
            if cc == c:
                return i + 1
    else:
        return 0


category_dict = courseinfo['category'].to_dict()
all_feat['course_category'] = [category_convert(category_dict.get(str(x), None)) for x in all_feat['course_id']]
act_feats = [c for c in train_feat.columns if 'count' in c or 'time' in c or 'num' in c]
pkl.dump(act_feats, open('act_feats.pkl', 'wb'))
num_feats = ['age', 'course_enroll_num', 'user_enroll_num']
scaler = StandardScaler()
newX = scaler.fit_transform(all_feat[num_feats])
for i, n_f in tqdm(enumerate(num_feats)):
    all_feat[n_f] = newX[:, i]
all_feat.loc[train_feat.index].to_csv(os.path.join(output_data_dir, 'train_feat_35_final.csv'))
all_feat.loc[test_feat.index].to_csv(os.path.join(output_data_dir, 'test_feat_35_final.csv'))
