import os.path as osp
import os
import random
import json
import numpy

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SplineConv, GCNConv, BatchNorm, PNAConv
from torch.nn import Linear
from torch_geometric.utils import degree
#from torch_geometric.utils import dropout_node
from torch_geometric.utils.dropout import dropout_node
#import torch_geometric.utils
from torch_geometric.nn import global_max_pool

from torch_geometric.data import InMemoryDataset  # , download_url
from torch_geometric.data import Dataset  # , download_url
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Dropout

epoch_list = []
best_val_acc_list = []
test_acc_list = []
loss_list = []
print("Enter Familiy input file, like RF01493.txt")
#predata = 'RF00034.txt'
predata = input()

class MyOwnDataset(InMemoryDataset):
    root = "/home/mgarcia/si_RNA_Network/"

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        input_data = predata
        return [input_data]

    @property
    def processed_file_names(self):
        processed_folder = "/home/mgarcia/si_RNA_Network/processed/"
        file_name = predata.split('.')
        processed_file = file_name[0] + '.pt'
        return processed_file

    def download(self):
        pass

    def process(self):
        for raw_path in self.raw_paths:
            print("raw_path", raw_path)
            # Read data from `raw_path`.
            raw_list_atrr = []
            raw_list_edge = []
            raw_list_feature = []
            raw_list_label = []
            raw_train_mask = []
            raw_test_mask = []
            raw_val_mask = []

            with open(raw_path, 'r') as f:
                for i in f:
                    if "feature" in i:
                        feature = i.strip().split('feature')
                        feature = feature[1:]
                        for k in feature:
                            d = k.replace("[", "").replace("]", "").split(",")
                            for k in d:
                                k = float(k)
                                raw_list_feature.append(k)
                    elif "edge_index" in i:
                        edge_index_raw = i.strip().split('edge_index')
                        edge_index_raw = edge_index_raw[1:]
                        for k in edge_index_raw:
                            q = k.split("], ")
                            for j in q:
                                u = j.replace("[", "").replace("]", "").split(",")
                                u = [int(u[0]), int(u[1])]
                                raw_list_edge.append(u)
                    elif "edge_attribute" in i:
                        edge_attribute = i.strip().split('edge_attribute')
                        edge_attribute = edge_attribute[1:]
                        for k in edge_attribute:
                            q = k.split("], ")
                            for j in q:
                                u = j.replace("[", "").replace("]", "").split(",")
                                u = [float(u[0]), float(u[1])]
                                raw_list_atrr.append(u)
                    elif "label" in i:
                        label = i.strip().split('label')
                        label = label[1:]
                        for k in label:
                            d = k.replace("[", "").replace("]", "").split(",")
                            for k in d:
                                k = int(k)
                                raw_list_label.append(k)
                # train_, test_ and val_mask:
                count_target = 0
                cluster_count = 0
                print("count_target_start: ", count_target)
                print("cluster_count_start: ", cluster_count)
                print("num_labels", len(raw_list_label))
                cluster_index_list = []
                target_index_list = []

                for index, reader in enumerate(raw_list_label):
                    if reader == 0:
                        cluster_count = cluster_count + 1
                        cluster_index_list.append(index)
                    elif reader == 1:
                        count_target = count_target + 1
                        target_index_list.append(index)
                    elif reader == 2:
                        count_target = count_target + 1
                        target_index_list.append(index)

                print("cluster_count", cluster_count, "count_target", count_target)
                print("80% target: ", round(count_target * 0.8), "20% target: ", round(count_target * 0.2))
                train_targets = round(count_target * 0.8)
                train_mask_num = [1 for i in range(len(raw_list_label))]
                val_mask_num = [0 for i in range(len(raw_list_label))]
                test_mask_num = [0 for i in range(len(raw_list_label))]
                check = 0
                target_train = random.sample(target_index_list, k=round(count_target * 0.8))
                test_and_val = []
                for target in target_index_list:
                    if target not in target_train:
                        test_and_val.append(target)

                test_list = random.sample(test_and_val, k=round(len(test_and_val) / 2))
                val_list = []
                for val in test_and_val:
                    if val not in test_list:
                        val_list.append(val)

                for index, i in enumerate(train_mask_num):
                    if index in test_and_val:
                        train_mask_num[index] = 0
                for index, i in enumerate(test_mask_num):
                    if index in test_list:
                        test_mask_num[index] = 1

                for index, i in enumerate(val_mask_num):
                    if index in val_list:
                        val_mask_num[index] = 1

                train_mask = torch.tensor(train_mask_num, dtype=torch.bool)
                test_mask = torch.tensor(test_mask_num, dtype=torch.bool)
                val_mask = torch.tensor(val_mask_num, dtype=torch.bool)

                attributes_tensor = torch.tensor(raw_list_atrr)
                edge_index_tensor = torch.tensor(raw_list_edge, dtype=torch.long).t().contiguous()
                node_feature_tensor = torch.tensor(raw_list_feature, dtype=torch.float)
                label_tensor = torch.tensor(raw_list_label)

        data = Data(x=node_feature_tensor, edge_index=edge_index_tensor, edge_attr=attributes_tensor,
                    y=label_tensor)
        data.test_mask = test_mask
        data.train_mask = train_mask
        data.val_mask = val_mask

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # data, slices = self.collate(data_list)
        torch.save((data), self.processed_paths[0])
        # create log file
        out_log = "/home/mgarcia/si_RNA_Network/label_logs/"+predata+"lbael_log_prob.txt"
        with open(out_log, 'w') as f:
            f.write("label_tensor")
            f.write(str(data.y))
            f.write('\n')
            f.write("train_mask")
            f.write(str(data.train_mask))
            f.write('\n')
            f.write("test_mask")
            f.write(str(data.test_mask))
            f.write('\n')
            f.write("val_mask")
            f.write(str(data.val_mask))
            f.write('\n')
            f.write("target_train")
            f.write(str(target_train))
            f.write('\n')
            f.write("test_and_val")
            f.write(str(test_and_val))
            f.write('\n')


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_channels = 32

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.conv1 = PNAConv(in_channels=data.num_node_features, out_channels=hidden_channels,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=2, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        self.batch_norm1 = BatchNorm(hidden_channels)
        self.do1 = dropout_node(data.edge_index)
        self.do1 = Dropout(p=0.2)

        self.conv2 = PNAConv(in_channels=hidden_channels, out_channels=16,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=2, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        self.batch_norm2 = BatchNorm(16)
        #self.do2 = dropout_node(p=0.1)

        self.conv3 = PNAConv(in_channels=16, out_channels=8,
                             aggregators=aggregators, scalers=scalers, deg=deg,
                             edge_dim=2, towers=1, pre_layers=1, post_layers=1,
                             divide_input=False)
        self.batch_norm3 = BatchNorm(8)
        #self.do3 = dropout_node(0.01)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        #print(type(x))
        x = self.batch_norm1(x)
        x = self.do1(x)
        #x = self.do1(0.2)
        x = F.elu(x)
        #print(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.batch_norm2(x)
        #x = self.do2(x)
        #x = self.do1(x)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.batch_norm3(x)
        #x = self.do3(x)
        x = self.do1(x)
        x = F.elu(x)
        return F.log_softmax(x, dim=1)


dataset = MyOwnDataset("/home/mgarcia/si_RNA_Network")  # , transform=transform

data = dataset[0]


max_degree = -1
d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
#d = degree(data.edge_index[1], dtype=torch.long)
max_degree = max(max_degree, int(d.max()))
deg = torch.zeros(max_degree + 1, dtype=torch.long)
deg += torch.bincount(d, minlength=deg.numel())
device = torch.device('cuda')
model = Net()
model.to(device)
data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    optimizer.zero_grad()
    out = model()
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

pred_list = []
test_accs_list = []
diff_list = []
real_list = []
pred_array_list = []
@torch.no_grad()
def test():
    model.eval()
    log_probs, accs = model(), []
    for _, mask in data('train_mask','test_mask'): #'train_mask'
        #print('_',_)
        #print('mask',mask)
        pred = log_probs[mask].max(1)[1]
        print('pred',pred)
        pred_list.append(pred)
        print('data.y[mask])',data.y[mask])
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
        test_accs_list.append(acc)
        print('acc',acc)
        pred_array = pred.to('cpu').numpy()
        pred_array_list.append(pred_array)
        real_array = data.y[mask].to('cpu').numpy()
        real_list.append(real_array)
        #print(real_array)
        #print(pred_array)
        diff = real_array - pred_array
        diff_list.append(diff)
        #print('diff', diff)
        #for i in diff:
          #for k in real_array:
            #if i == 0:
              #print('diff',i)
              #print('real',k)
    return accs


best_val_acc = test_acc = 0
val_acc = 0
epoch_train = 300
epoch_log = []
best_val_acc_log = []
test_acc_log = []
loss_log = []
for epoch in range(1, epoch_train + 1):
    loss = train()
    val_acc, tmp_test_acc = test()
    print('val_acc', val_acc)
    print('tmp_test_acc',tmp_test_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    epoch_list.append(float(epoch))
    loss_list.append(float(loss))
    best_val_acc_list.append(float(best_val_acc))
    test_acc_list.append(float(test_acc))
    log = 'Epoch: {:03d}, Val: {:.4f}, Test:{:.4f} loss: {:.4f}'
    epoch_log.append(epoch)
    best_val_acc_log.append(best_val_acc)
    test_acc_log.append(test_acc)
    loss_log.append(float(loss))
    if epoch % 10 == 0:
        print(log.format(epoch, best_val_acc, test_acc, loss))
    if epoch == epoch_train:
        out_file_name = predata.split('.')
        outfile = out_file_name[0]
        print(real_list[-1])
        print(pred_array_list[-1])
        print(diff_list[-1])
        real_calc = real_list[-1]
        diff_calc = diff_list[-1]
        print('real_calc',real_calc)
        print('diff_calc',diff_calc)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for index ,i in enumerate(diff_calc):
           if i ==0:
             if real_calc[index] == 2:
               TP = TP + 1
             elif real_calc[index] == 1:
               TN = TN + 1
           elif i == 1:
             FN = FN + 1
           elif i == -1:
             FP = FP + 1
        print('TP',TP)
        print('TN',TN)
        print('FP',FP)
        print('FN',FN)
        sensitivity = (TP/(TP+FN))*100
        specificity = (TN/(FP+TN))*100
        FPR = FP/(FP+TN)
        print('sensitivity in %',sensitivity)
        print('specificity in %',specificity)
        print('FPR',FPR)

        checkpointpath_state = "/home/mgarcia/si_RNA_Network/checkpoints/" + str(
            epoch_train) + "/state_dict/checkpoint_dropout_CH_32_16_8_prob_" + str(epoch_train) + "epochs_" + outfile + ".pth"
        """
        checkpointpath_state = "/home/mgarcia/si_RNA_Network/checkpoints/" + str(
            epoch_train) + "/state_dict/checkpoint_modified_degree_" + str(epoch_train) + "epochs_" + outfile + ".pth"
        """
        checkpointpath_info = "/home/mgarcia/si_RNA_Network/checkpoints/" + str(
            epoch_train) + "/info_dict/checkpoint_dropout_CH_32_16_8_prob_" + str(epoch_train) + "epochs_" + outfile + ".pth"
        """
        checkpointpath_info = "/home/mgarcia/si_RNA_Network/checkpoints/" + str(
            epoch_train) + "/info_dict/checkpoint_modified_degree_" + str(epoch_train) + "epochs_" + outfile + ".pth"
        """
        checkpoint_state = {
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),

        }
        checkpoint_info = {
            'epoch': epoch_train,
            'deg': deg,
            'dataset': outfile
        }
        torch.save(checkpoint_state, checkpointpath_state)
        torch.save(checkpoint_info, checkpointpath_info)
        fig = plt.figure(figsize=(12, 15), dpi=80)
        fig.suptitle('loss, test- and validation accuracy over trained epochs', fontsize=20, fontweight='bold')

        plt.subplot(221)
        plt.plot(epoch_list, loss_list)
        plt.grid()
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('loss', fontsize=20)

        plt.subplot(222)
        plt.plot(epoch_list, test_acc_list)
        plt.grid()
        plt.ylim(0,1)
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('test_acc', fontsize=20)

        plt.subplot(223)
        plt.plot(epoch_list, best_val_acc_list)
        plt.grid()
        plt.ylim(0,1)
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('val_acc', fontsize=20)

        name_plot = "/home/mgarcia/si_RNA_Network/figures/" + str(outfile) + "_CH_32_16_8_dropout_prob_" + str(epoch_train) + ".png"
        #name_plot = "/home/mgarcia/si_RNA_Network/figures/" + str(outfile) + "_modified_degree_" + str(epoch_train) + ".png"

        fig.savefig(name_plot)
        plot_data = "/home/mgarcia/si_RNA_Network/figures/plot_data/"+str(outfile)+"_CH_32_16_8_dropout_prob_"+str(epoch_train)+".json"
        plot_data_dict = {"epoch_list":epoch_list,
                          "loss_list":loss_list,
                          "test_acc_list":test_acc_list,
                          "best_val_acc_list":best_val_acc_list,
                          "TP":TP,
                          "TN":TN,
                          "FP":FP,
                          "FN":FN,
                          "sensitivity in %":sensitivity,
                          "specificity in %":specificity,
                          "FPR":FPR
                          }
        with open(plot_data, 'w') as f:
            json.dump(plot_data_dict, f)

#print(pred_list)
#print(test_accs_list)
