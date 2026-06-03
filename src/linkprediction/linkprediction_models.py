import os
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import pandas as pd

from torch import nn as nn
from tqdm import tqdm
from torch_geometric.data import HeteroData
from torch_geometric.nn import to_hetero, SAGEConv
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threshold_strategies import apply_threshold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_data_dir = '../../datastore'
output_data_dir = '../../datastore'
model_save_dir = '../../datastore'
threshold = 0.6

# Chọn chiến lược ngưỡng η₁: 'fixed' hoặc 'adaptive'
THRESHOLD_MODE = os.environ.get('THRESHOLD_MODE', 'fixed')
THRESHOLD_PERCENTILE = int(os.environ.get('THRESHOLD_PERCENTILE', '70'))
print(f'  [config] η₁: mode={THRESHOLD_MODE}, percentile={THRESHOLD_PERCENTILE}')

train = pd.read_csv(os.path.join(input_data_dir, 'train_feat_35_final.csv'))[
    ['username', 'age', 'gender', 'education', 'cluster_label', 'user_enroll_num', 'course_id', 'course_enroll_num',
     'course_category']]
test = pd.read_csv(os.path.join(input_data_dir, 'test_feat_35_final.csv'))[
    ['username', 'age', 'gender', 'education', 'cluster_label', 'user_enroll_num', 'course_id', 'course_enroll_num',
     'course_category']]
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
user_enroll_course = all_log[['user_graph_id', 'course_graph_id']].drop_duplicates()
user_enroll_id = torch.from_numpy(user_enroll_course['user_graph_id'].values).to(torch.long)
course_enroll_id = torch.from_numpy(user_enroll_course['course_graph_id'].values).to(torch.long)
user_enroll_course_edge_index = torch.stack([user_enroll_id, course_enroll_id], dim=0)

h_graph = HeteroData()
h_graph['user'].x = user_feat
h_graph['user'].node_index = torch.arange(len(unique_user_id))
h_graph['user'].n_id = torch.arange(len(unique_user_id))
h_graph['course'].x = course_feat
h_graph['course'].node_index = torch.arange(len(unique_course_id))
h_graph['course'].n_id = torch.arange(len(unique_course_id))
h_graph['user', 'enroll', 'course'].edge_index = user_enroll_course_edge_index
h_graph = T.ToUndirected()(h_graph)

transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=('user', 'enroll', 'course'),
    rev_edge_types=('course', 'rev_enroll', 'user'),
)

pre_train_compose = T.Compose([
    transform
])

train_data, val_data, test_data = pre_train_compose(h_graph)
edge_label_index = train_data['user', 'enroll', 'course'].edge_label_index
edge_label = train_data['user', 'enroll', 'course'].edge_label
edge_label_index_val = val_data['user', 'enroll', 'course'].edge_label_index
edge_label_val = val_data['user', 'enroll', 'course'].edge_label

train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[3, 90],
    neg_sampling_ratio=3,
    edge_label_index=(('user', 'enroll', 'course'), edge_label_index),
    edge_label=edge_label,
    batch_size=256,
    shuffle=True,
)

val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[3, 90],
    edge_label_index=(('user', 'enroll', 'course'), edge_label_index_val),
    edge_label=edge_label_val,
    batch_size=128,
    shuffle=False
)


class HeteroGNNModel(nn.Module):
    def __init__(self, param_dict):
        super(HeteroGNNModel, self).__init__()
        self.input_features1 = param_dict['graph_hidden_features'] * 2
        self.hidden_features1 = param_dict['hidden_features1']
        self.output_features1 = param_dict['output_features1']
        self.heads = param_dict['heads']
        self.conv1 = SAGEConv(self.input_features1, self.hidden_features1)
        self.conv2 = SAGEConv(self.input_features1, self.output_features1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(self, x_user, x_course, edge_label_index):
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_course = x_course[edge_label_index[1]]
        return (edge_feat_user * edge_feat_course).sum(dim=-1)


class EncoderModel(nn.Module):
    def __init__(self, param_dict):
        super(EncoderModel, self).__init__()
        self.graph_hidden_features = param_dict['graph_hidden_features']
        self.user_features = param_dict['user_features_len']
        self.course_features = param_dict['course_features_len']

        self.user_lin = torch.nn.Linear(self.user_features, self.graph_hidden_features)
        self.course_lin = torch.nn.Linear(self.course_features, self.graph_hidden_features)

        self.user_embed = torch.nn.Embedding(h_graph['user'].num_nodes, self.graph_hidden_features)
        self.course_embed = torch.nn.Embedding(h_graph['course'].num_nodes, self.graph_hidden_features)

        self.gnn = HeteroGNNModel(param_dict)
        self.gnn = to_hetero(self.gnn, metadata=h_graph.metadata(), aggr='mean')

        self.classifier = Classifier()

    def forward(self, input_data):
        x_dict = {
            "user": torch.concat((self.user_embed(input_data['user'].node_index), self.user_lin(input_data['user'].x)),
                                 dim=1),
            "course": torch.concat(
                (self.course_embed(input_data['course'].node_index), self.course_lin(input_data['course'].x)), dim=1)
        }

        x_dict = self.gnn(x_dict, input_data.edge_index_dict)
        pred = self.classifier(
            x_dict['user'],
            x_dict['course'],
            input_data['user', 'enroll', 'course'].edge_label_index,
        )
        return pred


param_dict = dict({
    'hidden_features1': 32,
    'output_features1': 16,
    'heads': 1,
    'user_features_len': 5,
    'course_features_len': 2,
    'graph_hidden_features': 16
})

encoder = EncoderModel(param_dict)
encoder = encoder.to(device)
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
epoch_num = 10
best_model_epoch = None
best_model_auc = -np.inf
for epoch in range(epoch_num):
    total_loss = total_examples = 0
    for sampled_data in tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = encoder(sampled_data)
        ground_truth = sampled_data["user", "enroll", "course"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

    preds = []
    ground_truths = []
    for sampled_data in tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(encoder(sampled_data))
            ground_truths.append(sampled_data["user", "enroll", "course"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    if auc > best_model_auc:
        best_model_auc = auc
        model_save_path = os.path.join(model_save_dir, '{}.pth'.format(epoch))
        best_model_epoch = epoch
        torch.save(encoder, model_save_path)

best_model = torch.load(os.path.join(model_save_dir, '{}.pth'.format(best_model_epoch)), weights_only=False)
best_model = best_model.to(device)
user_enroll_course_matrix = torch.zeros((len(unique_user_id_df), len(unique_course_id_df)))
for index, item in all_log.iterrows():
    row = int(item['user_graph_id'])
    col = int(item['course_graph_id'])
    user_enroll_course_matrix[row][col] = 1

dataset_loader = NeighborLoader(
    data=h_graph,
    num_neighbors=[3, 90],
    input_nodes=('user', h_graph['user'].node_index),
    batch_size=256,
    shuffle=False
)
user_encoder_features_dict = dict()
user_encoder_features_list = []
all_count = 0
with torch.no_grad():
    for cur_data in tqdm(dataset_loader):
        cur_data = cur_data.to(device)
        batch_size = cur_data['user'].batch_size
        cur_data_dict = {
            'user': torch.concat(
                (best_model.user_embed(cur_data['user'].n_id[:]), best_model.user_lin(cur_data['user'].x[:])), dim=1),
            'course': torch.concat((best_model.course_embed(cur_data['course'].n_id[:]), best_model.course_lin(
                cur_data['course'].x[:])), dim=1)
        }
        cur_data_dict = best_model.gnn(cur_data_dict, cur_data.edge_index_dict)
        batch_user_features_cp = cur_data_dict['user'].cpu()
        batch_user_features = cur_data_dict['user'].cpu().numpy()

        for i in range(batch_size):
            cur_data_user = batch_user_features[i]
            cur_username = username_graph_dict[all_count]
            all_count += 1
            user_encoder_features_dict[cur_username] = cur_data_user
            user_encoder_features_list.append(batch_user_features_cp[i].view(1, -1))

dataset_loader = NeighborLoader(
    data=h_graph,
    num_neighbors=[90, 3],
    input_nodes=('course', h_graph['course'].node_index),
    batch_size=64,
    shuffle=False
)

course_encoder_features_dict = dict()
course_encoder_features_list = []
all_count = 0
with torch.no_grad():
    for cur_data in tqdm(dataset_loader):
        cur_data = cur_data.to(device)
        batch_size = cur_data['course'].batch_size
        cur_data_dict = {
            'user': torch.concat(
                (best_model.user_embed(cur_data['user'].n_id[:]), best_model.user_lin(cur_data['user'].x[:])), dim=1),
            'course': torch.concat((best_model.course_embed(cur_data['course'].n_id[:]), best_model.course_lin(
                cur_data['course'].x[:])), dim=1)
        }
        cur_data_dict = best_model.gnn(cur_data_dict, cur_data.edge_index_dict)
        batch_course_features_cp = cur_data_dict['course'].cpu()
        batch_course_features = cur_data_dict['course'].cpu().numpy()

        for i in range(batch_size):
            cur_data_course = batch_course_features[i]
            cur_course_id = course_graph_dict[all_count]
            all_count += 1
            course_encoder_features_dict[cur_course_id] = cur_data_course
            course_encoder_features_list.append(batch_course_features_cp[i].view(1, -1))

enhanced_user_features = torch.cat(user_encoder_features_list, dim=0)
enhanced_course_features = torch.cat(course_encoder_features_list, dim=0)
link_predict_matrix = enhanced_user_features @ enhanced_course_features.T
link_predict_matrix = F.sigmoid(link_predict_matrix)
link_predict_matrix = link_predict_matrix.cpu().numpy()
user_enroll_course_matrix = user_enroll_course_matrix.cpu().numpy()

# Binarize bằng strategy đã chọn (η₁)
link_predict_matrix = apply_threshold(
    link_predict_matrix,
    mode=THRESHOLD_MODE,
    threshold=threshold,                # dùng cho mode 'fixed'
    percentile=THRESHOLD_PERCENTILE,    # dùng cho mode 'adaptive'
)

# Đảm bảo các enrollment thực tế luôn có mặt
for index, item in all_log.iterrows():
    row = int(item['user_graph_id'])
    col = int(item['course_graph_id'])
    link_predict_matrix[row][col] = 1

np.save(os.path.join(output_data_dir, 'link_predict_matrix.npy'), link_predict_matrix)
np.save(os.path.join(output_data_dir, 'test_user.npy'), user_encoder_features_dict)
np.save(os.path.join(output_data_dir, 'test_course.npy'), course_encoder_features_dict)
