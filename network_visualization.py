import networkx as nx
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm

def get_key(val):
    for key, value in d_dict.items():
        if val == value:
            return key

    return "key doesn't exist"

raw_list_atrr = []
raw_list_edge = []
raw_list_feature = []
raw_list_label = []
raw_train_mask = []
raw_test_mask = []
raw_val_mask = []
d_dict = {}
graph_node_list = []

#root = "/home/mgarcia/si_RNA_Network/raw_strings/"
#dummy_root = "/home/mgarcia/si_RNA_Network/raw_dummy/"

#txt_path = root + "/RF00034.txt"
#txt_path =dummy_root + "RF00034.txt" #RF01476 / RF00034
txt_path = "/home/mgarcia/si_RNA_Network/raw/RF01395.txt"
#path_idx = root + "RF01476.idx"
#path_idx = dummy_root + "RF00034.idx"

with open(txt_path, 'r') as f:
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
                    u = j.replace("[", "").replace("'","").replace("]", "").split(",")
                    #u = [int(u[0]), int(u[1])]
                    u = [u[0], u[1]]
                    #print(u)
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
"""
with open(path_idx, 'r') as f:
    for i in f:
        if 'pos_idx' in i:
            pos_idx = i.strip().split('pos_idx')
            pos_idx = pos_idx[1]
            pos_idx = pos_idx.replace('[', '').replace(']', '').split(',')
        elif 'targets' in i:
            targets = i.strip().split('targets')
            targets = targets[1]
            targets = targets.replace('[', '').replace('[', '').split(',')
        elif 'graph_nodes' in i:
            graph_nodes = i.strip().split('graph_nodes')
            graph_nodes = graph_nodes[1]
            graph_nodes = graph_nodes.replace('[', '').replace('[', '').split(',')
            for j in graph_nodes:
                j.strip()
                d = j.split("'")
                graph_node_list.append(d[1])
        elif "d_dict" in i:
            feature = i.strip().split('d_dict')
            feature = feature[1:]
            # print("feature", feature)
            for k in feature:
                d = k.replace("{", "").replace("}", "").split(",")  # .strip()
                # print("d", d)
                for j in d:
                    j.strip()
                    #print("j", j)
                    item = j.split(":")
                    item[0] = item[0].replace("'", "")
                    item[0] = item[0].strip()
                    d_dict[item[0]] = int(item[1])
"""

#Listprints

#print("raw_list_atrr", raw_list_atrr[0:5])
# print("raw_list_atrr", raw_list_atrr)
#print("raw_list_edge", raw_list_edge[0:5])
# print("raw_list_edge", raw_list_edge)
#print("raw_list_feature", raw_list_feature[0:5])
# print("raw_list_feature", raw_list_feature)
#print("raw_list_label", raw_list_label[0:5])
# print("raw_list_label", raw_list_label)
# print("pos_idx", pos_idx[0:5])
# print("pos_idx", pos_idx)
# print("targets", targets[0:5])
# print("targets", targets)
#print("d_dict",d_dict)
#print("graph_node_list", graph_node_list)

#Auf bau des netrwerkgraphen

list_b = []
for node in graph_node_list:
    if node in d_dict:
        #print("here I'am")
        if 'cluster'  in node:
            #print("I'm a cluster")
            enty_cluster = (node, {'color': 'blue'})
            list_b.append(enty_cluster)
        else:
            #print("I'm a target")
            enty_target = (node, {'color': 'green'})
            list_b.append(enty_target)
            #print(node)

    else:
        #print('nich da', node)
        entry_fals_target = (node, {'color': 'red'})
        list_b.append(entry_fals_target)




G = nx.Graph()

G.add_nodes_from(list_b)
color_map = nx.get_node_attributes(G, 'color')
#print(color_map)
node_color = [color_map.get(node) for node in G.nodes()]
#print(node_color)
"""
for edge in raw_list_edge:
    for edge_attribute_graph in raw_list_atrr:
        G.add_edge(edge[0], edge[1])#, [{'color': 'black'}])#, edge_attribute_graph)
"""
for edge in tqdm(raw_list_edge, desc='edge adding to graph'):
    sleep(0.000001)
    G.add_edge(edge[0], edge[1])#, [{'color': 'black'}])#, edge_attribute_graph)

#print(list_b[0:5])
#edge_attr = nx.get_edge_attributes(G, 'attr')
#print("edge_attr", edge_attr)


nx.draw_networkx(G, with_labels=True)#, node_color=node_color)

plt.show()
