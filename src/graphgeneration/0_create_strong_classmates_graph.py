import os
import torch
import pandas as pd
import numpy as np
import pickle as pkl

from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

sim_threshold = 0.95

# Chọn chiến lược ngưỡng η₂: 'fixed' hoặc 'adaptive'
# fixed: dùng sim_threshold cố định (0.95) — hành vi gốc
# adaptive: per-course percentile — mỗi khóa học có ngưỡng riêng
SIM_MODE = os.environ.get('SIM_MODE', 'fixed')
SIM_PERCENTILE = int(os.environ.get('SIM_PERCENTILE', '90'))
print(f'  [config] η₂: mode={SIM_MODE}, sim_threshold={sim_threshold}, '
      f'sim_percentile={SIM_PERCENTILE}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_data_dir = '../../datastore'
dict_dir = '../../datastore'

all_log = pd.read_csv(os.path.join(input_data_dir, 'graph_mapping.csv'))
all_log_copy = all_log.sort_values('record_graph_id')

user_feat_dict = np.load(os.path.join(dict_dir, 'test_user.npy'), allow_pickle=True).item(0)
course_feat_dict = np.load(os.path.join(dict_dir, 'test_course.npy'), allow_pickle=True).item(0)
link_predict_matrix = np.load(os.path.join(dict_dir, 'link_predict_matrix.npy'))

data_all = all_log.groupby('course_id')
user_feat_matrix = torch.tensor([user_feat_dict[int(username)] for username in all_log_copy['username']],
                                dtype=torch.float)
user_select_courses_mean_df = all_log.groupby('username')
user_course_feat_dict = dict()

for cur_username, group_df in tqdm(user_select_courses_mean_df):
    course_select_list = list(group_df['course_id'])
    course_feat_cur_user = np.array([course_feat_dict[str(course_id)] for course_id in course_select_list])
    course_feat_cur_user = np.mean(course_feat_cur_user, axis=0)
    user_course_feat_dict[int(cur_username)] = course_feat_cur_user

user_course_feat_matrix = torch.tensor([user_course_feat_dict[int(username)] for username in all_log_copy['username']])
node_features = torch.cat((user_feat_matrix, user_course_feat_matrix), dim=1)
node_index = torch.tensor(list(all_log_copy['record_graph_id']), dtype=torch.long)
node_y = torch.tensor(list(all_log_copy['truth']), dtype=torch.long)

train_df = all_log_copy[all_log_copy['tt_label'] == 'train']
train_mask = torch.tensor(list(train_df['record_graph_id']), dtype=torch.long)
train_truth = torch.tensor(list(train_df['truth']), dtype=torch.long)
test_df = all_log_copy[all_log_copy['tt_label'] == 'test']
test_mask = torch.tensor(list(test_df['record_graph_id']), dtype=torch.long)
test_truth = torch.tensor(list(test_df['truth']), dtype=torch.long)

source_nodes = []
target_nodes = []
user_enroll_course_matrix = np.zeros((len(user_feat_dict), len(course_feat_dict)))
for index, item in all_log_copy.iterrows():
    row = int(item['user_graph_id'])
    col = int(item['course_graph_id'])
    user_enroll_course_matrix[row][col] = 1

for course_name, group_df in tqdm(data_all):
    user_graph_id_list = np.array([user_graph_id for user_graph_id in group_df['user_graph_id']])
    course_graph_id_list = np.array([course_graph_id for course_graph_id in group_df['course_graph_id']])
    record_graph_id_list = [record_graph_id for record_graph_id in group_df['record_graph_id']]
    Left_Matrix = np.array([link_predict_matrix[one_hot_u] for one_hot_u in user_graph_id_list])
    Right_Matrix = np.array([link_predict_matrix[one_hot_u] for one_hot_u in user_graph_id_list])
    cosine_matrix = cosine_similarity(Left_Matrix, Right_Matrix)
    np.fill_diagonal(cosine_matrix, val=0)

    if SIM_MODE == 'adaptive':
        # η₂ adaptive: per-course percentile
        # Lấy tất cả giá trị cosine > 0 (bỏ diagonal = 0)
        nonzero_vals = cosine_matrix[cosine_matrix > 0]
        if len(nonzero_vals) > 0:
            course_threshold = np.percentile(nonzero_vals, SIM_PERCENTILE)
        else:
            course_threshold = sim_threshold  # fallback
        row, col = np.where(cosine_matrix >= course_threshold)
    else:
        # η₂ fixed: ngưỡng cố định 0.95
        row, col = np.where(cosine_matrix >= sim_threshold)

    for i in range(len(row)):
        source_nodes.append(record_graph_id_list[row[i]])
        target_nodes.append(record_graph_id_list[col[i]])

source_nodes = torch.tensor(source_nodes).view(-1, len(source_nodes))
target_nodes = torch.tensor(target_nodes).view(-1, len(target_nodes))
edge_index = torch.cat((source_nodes, target_nodes), dim=0)

Sim_Graph = Data(edge_index=edge_index)
Sim_Graph.labels = node_y
Sim_Graph.enhanced_context = node_features
Sim_Graph.n_id = node_index
Sim_Graph.train_mask = train_mask
Sim_Graph.train_truth = train_truth
Sim_Graph.test_mask = test_mask
Sim_Graph.test_truth = test_truth

with open(os.path.join(input_data_dir, 'StrongClassmatesGraph.pkl'), 'wb') as f:
    pkl.dump(Sim_Graph, f)
    pkl.dump(Sim_Graph, f)