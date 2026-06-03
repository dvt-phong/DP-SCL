"""
LGB Trainer — Framework 0: Training cho legacy LGB/graph baseline (18 modes).

Nhánh GRAPH (9 modes):  default, cnn, cnn2d, gat, cnn_gat, cross_attn, mba_cnn, mba_cnn_gat, bilstm_graph
Nhánh NO-GRAPH (9 modes): no_graph, bilstm_cnn, bilstm_mha, bilstm_cross, mba_bilstm, cnn_only, mba_only, cnn_day, bilstm_day
"""
import os
import torch

from .base_trainer import BaseTrainer
from .utils import load_temporal_data, load_graph_data, get_loss_fn
from src.models import LGB, SupConLoss
from src.mode_registry import is_no_graph_mode


class LGBTrainer(BaseTrainer):
    """Trainer cho LGB (Framework 0)."""

    def build_param_dict(self):
        mode = self.mode
        ds = self.ds_config

        self.param_dict = {
            'activity_num': ds['activity_num'], 'sta_day': ds['sta_day'],
            'week_count': ds['week_count'], 'select_count': ds['week_count'],
            'org_context_feat_len': 7, 'enhanced_context_feat_len': 32 if self.args.dataset == 'xuetangx' else 7,
            'context_each_embed': 16, 'context_all_len': 16,
            'input_features': 16, 'hidden_features': 32, 'output_features': 16,
            'lstm_input_features': 8 * (ds['activity_num'] + 1),
            'lstm_hidden_features': 128, 'lstm_hidden_num_layers': 1,
            'num_attention_heads': 1, 'attention_features': 64,
            'l2_input_features': 64, 'l2_hidden_features': 32, 'l2_hidden_num_layers': 1,
            's2_num_attention_heads': 1, 's2_attention_features': 16,
            'ws_num_attention_heads': 1, 'ws_input_features': 32, 'ws_attention_features': 16,
            'dnn_input_f1': 16, 'dnn_hidden_f1': 16, 'dnn_hidden_f2': 8,
            'dnn_hidden_f3': 4, 'dnn_output': 1,
        }

        # Mode-specific params
        if mode in ('cnn', 'cnn_gat'):
            self.param_dict.update({
                'cnn_in_channels': 7, 'cnn_out_channels_1': 32, 'cnn_out_channels_2': 64,
                'cnn_kernel_size': 3, 'cnn_fc_output': 128,
            })
        if mode in ('gat', 'cnn_gat', 'mba_cnn_gat'):
            self.param_dict.update({'gat_heads': 4, 'gat_dropout': 0.3})
        if mode == 'cnn2d':
            self.param_dict.update({
                'cnn2d_out_channels_1': 32, 'cnn2d_out_channels_2': 64,
                'cnn2d_kernel_size': 3, 'cnn2d_fc_output': 128,
            })
        if mode == 'cross_attn':
            self.param_dict.update({'ca_num_heads': 4, 'ca_output_dim': 16, 'ca_ffn_dim': 32})
        if mode in ('mba_cnn', 'mba_cnn_gat'):
            self.param_dict.update({
                'mba_cnn_temporal_channels_1': 32, 'mba_cnn_temporal_channels_2': 64,
                'mba_cnn_daily_channels': 32, 'mba_cnn_weekly_channels': 32,
                'mba_cnn_fc_hidden': 256, 'mba_cnn_output': 128, 'mba_cnn_dropout': 0.3,
            })
        if mode == 'bilstm_cnn':
            self.param_dict.update({
                'cnn_in_channels': 22, 'cnn_out_channels_1': 64, 'cnn_out_channels_2': 128,
                'cnn_kernel_size': 3, 'cnn_fc_output': 128,
                'lstm_input_features': 256, 'lstm_hidden_features': 64,
                'l2_input_features': 64,   # SA1 output dim → BiLSTM2 input
                'l2_hidden_features': 16,
            })
        if mode == 'bilstm_mha':
            self.param_dict.update({
                'cnn_in_channels': 22, 'cnn_out_channels_1': 64, 'cnn_out_channels_2': 128,
                'cnn_kernel_size': 3, 'cnn_fc_output': 128,
                'lstm_input_features': 128, 'lstm_hidden_features': 64, 'lstm_hidden_num_layers': 1,
                'l2_input_features': 128, 'l2_hidden_features': 32, 'l2_hidden_num_layers': 1,
                'mha_num_heads': 4,
            })
        if mode == 'bilstm_cross':
            self.param_dict.update({
                'cnn_in_channels': 22, 'cnn_out_channels_1': 64, 'cnn_out_channels_2': 128,
                'cnn_kernel_size': 3, 'cnn_fc_output': 128,
                'lstm_input_features': 128, 'lstm_hidden_features': 64, 'lstm_hidden_num_layers': 1,
                'l2_input_features': 128, 'l2_hidden_features': 32, 'l2_hidden_num_layers': 1,
                'mha_num_heads': 4,
            })
        if mode == 'bilstm_graph':
            self.param_dict.update({
                'cnn_in_channels': 22, 'cnn_out_channels_1': 64, 'cnn_out_channels_2': 128,
                'cnn_kernel_size': 3, 'cnn_fc_output': 128,
                'lstm_input_features': 128, 'lstm_hidden_features': 64, 'lstm_hidden_num_layers': 1,
                'attention_features': 64,
                'l2_input_features': 64, 'l2_hidden_features': 16, 'l2_hidden_num_layers': 1,
                's2_attention_features': 16,
                'ws_input_features': 32, 'ws_attention_features': 16,
            })
        if mode == 'mba_bilstm':
            self.param_dict.update({
                'cnn_in_channels': 7,
                'mba_cnn_temporal_channels_1': 32, 'mba_cnn_temporal_channels_2': 64,
                'mba_cnn_daily_channels': 32, 'mba_cnn_weekly_channels': 32,
                'mba_cnn_fc_hidden': 256, 'mba_cnn_output': 128, 'mba_cnn_dropout': 0.3,
                'lstm_input_features': 128, 'lstm_hidden_features': 64, 'lstm_hidden_num_layers': 1,
                'attention_features': 64,
                'l2_input_features': 64, 'l2_hidden_features': 16, 'l2_hidden_num_layers': 1,
                's2_attention_features': 16,
            })
        if mode == 'cnn_only':
            self.param_dict.update({
                'cnn_in_channels': 22, 'cnn_out_channels_1': 64, 'cnn_out_channels_2': 128,
                'cnn_kernel_size': 3, 'cnn_fc_output': 128,
            })
        if mode == 'mba_only':
            self.param_dict.update({
                'cnn_in_channels': 7,
                'mba_cnn_temporal_channels_1': 32, 'mba_cnn_temporal_channels_2': 64,
                'mba_cnn_daily_channels': 32, 'mba_cnn_weekly_channels': 32,
                'mba_cnn_fc_hidden': 256, 'mba_cnn_output': 128, 'mba_cnn_dropout': 0.3,
            })
        if mode == 'cnn_day':
            self.param_dict.update({
                'cnn_in_channels': 7, 'cnn_out_channels_1': 64, 'cnn_out_channels_2': 128,
                'cnn_kernel_size': 3, 'cnn_fc_output': 128,
            })
        if mode == 'bilstm_day':
            self.param_dict.update({
                'cnn_in_channels': 7, 'cnn_out_channels_1': 64, 'cnn_out_channels_2': 128,
                'cnn_kernel_size': 3, 'cnn_fc_output': 128,
                'lstm_input_features': 128, 'lstm_hidden_features': 64, 'lstm_hidden_num_layers': 1,
                'attention_features': 64,
                'l2_input_features': 64, 'l2_hidden_features': 16, 'l2_hidden_num_layers': 1,
                's2_attention_features': 16,
            })

    def build_dataloaders(self):
        input_dir = os.path.abspath(os.path.expanduser(str(self.args.indir)))
        if is_no_graph_mode(self.mode):
            self.train_loader, self.test_loader, self._train_labels, self._n_pos, self._n_neg = \
                load_temporal_data(input_dir, self.ds_config, self.args.batch_size, self.args.sampling)
            self._is_temporal = True
        else:
            self.train_loader, self.test_loader, self._graph = \
                load_graph_data(input_dir, self.ds_config, self.args.batch_size, self.device)
            self._is_temporal = False
            # Compute n_pos, n_neg for loss
            _train_labels = self._graph.labels[self._graph.train_mask]
            self._n_pos = int(_train_labels.sum().item())
            self._n_neg = len(_train_labels) - self._n_pos

    def build_model(self):
        self.model = LGB(self.param_dict, mode=self.mode, contrastive=self.args.contrastive)
        self.model = self.model.to(self.device)
        if self.args.contrastive:
            self.supcon_criterion = SupConLoss(temperature=0.07).to(self.device)

    def build_loss_fn(self):
        self.bce_loss_fn = get_loss_fn(self.args.sampling, self._n_pos, self._n_neg, self.device)

    def train_step(self, batch):
        if self._is_temporal:
            seq_feat_batch, labels_batch = batch
            seq_feat_batch = seq_feat_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)
            batch_size = seq_feat_batch.shape[0]
            sub_graph = {'batch_size': batch_size, 'seq_feat': seq_feat_batch}
            ground_truth = labels_batch.view(-1, 1).to(torch.float)
        else:
            sub_graph = batch
            batch_size = sub_graph['batch_size']
            ground_truth = sub_graph['labels'][:batch_size].view(-1, 1).to(torch.float)

        pred = self.model(sub_graph)

        if self.args.contrastive:
            pred, proj_embed = pred
            bce_loss = self.bce_loss_fn(pred, ground_truth)
            con_loss = self.supcon_criterion(proj_embed, ground_truth)
            loss = bce_loss + self.args.lambda_con * con_loss
        else:
            loss = self.bce_loss_fn(pred, ground_truth)

        return loss, batch_size

    def eval_step(self, batch):
        if self._is_temporal:
            seq_feat_batch, labels_batch = batch
            seq_feat_batch = seq_feat_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)
            batch_size = seq_feat_batch.shape[0]
            sub_graph = {'batch_size': batch_size, 'seq_feat': seq_feat_batch}
            truth = labels_batch.view(-1, 1)
        else:
            sub_graph = batch
            batch_size = sub_graph['batch_size']
            truth = sub_graph['labels'][:batch_size].view(-1, 1)

        pred = self.model(sub_graph)
        if self.args.contrastive:
            pred, _ = pred
        return pred, truth
