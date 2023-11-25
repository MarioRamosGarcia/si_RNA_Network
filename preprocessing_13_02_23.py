import re
import os
from time import sleep
from tqdm import tqdm

"""
Root paths for source and output
"""
root = "/home/mgarcia/si_RNA_Network/source/Results_15_11_21_GLASSgo_vs_GLASSgo_CMsearch-default/"
#dummy_root = "/home/mgarcia/si_RNA_Network/source/dummy_source/"
tsv_root = root + "Bench_GLASSgo-syntney_cm_results_default/"
#tsv_root = dummy_root + "Bench_GLASSgo-syntney_cm_results_default/"
Syntney_root = root + "Bench_GLASSgo-out_RFAM_GLASSgo_Syntney/"
#Syntney_root = dummy_root + "Bench_GLASSgo-out_RFAM_GLASSgo_Syntney/"

preprocessed_data = "/home/mgarcia/si_RNA_Network/raw/"
#preprocessed_dummy = "/home/mgarcia/si_RNA_Network/raw_dummy/"

"""
Start Preprocessing
"""

print("read folders")
print()
print("start preproceccing")
print()
arr = os.listdir(Syntney_root)
print("first 5 folders:", arr[:5])
pbar = tqdm(total=len(arr), desc= "preprocessing " + str(len(arr))+ " folders")

for folder in arr:
    pbar.update(1)
    sleep(0.000001)
    d_dict = {}
    counter = 0
    id = ""
    Flag = True
    edge_index = []
    edge_attribute = []
    node_feature_raw = []
    label_raw = []
    threshold_e_value = 0.026
    accession = []
    value_list = []
    edge_list = []
    PID_dict = {}
    target_nodes = []
    cluster_nodes = []
    pos_idx = []
    targets = []
    graph_nodes = []
    _Network_Cluster = folder +"_Network_Cluster.txt"
    Network_Cluster_path = Syntney_root + folder + "/" + _Network_Cluster
    with open(Network_Cluster_path, "r") as a_file:
        for i in a_file:
            if not 'identifier' in i:
                q = re.split(r'[\t|,]+', i.strip(",\n"))
                for index, k in enumerate(q):
                    if k == 'sRNA':
                        continue
                    elif not k in d_dict:
                        d_dict[k] = counter
                        counter = counter + 1
                for l in q:
                    if id == "":
                        id = l
                    elif l == 'sRNA':
                        sRNA = id
                        Flag = True
                    else:
                        if Flag == True:
                            previous = l
                            edge_index.append([d_dict[sRNA], d_dict[l]])
                            edge_attribute.append([1,0])
                            edge_index.append([d_dict[l], d_dict[sRNA]])
                            edge_attribute.append([1, 0])
                            Flag = False
                        else:
                            edge_index.append([d_dict[previous], d_dict[l]])
                            edge_attribute.append([1, 0])
                            edge_index.append([d_dict[l], d_dict[previous]])
                            edge_attribute.append([1, 0])
                            previous = l
    #print("done Network_cluster.txt")
    _Network = folder + "_Network.txt"
    network_path = Syntney_root + folder + "/" + _Network
    with open(network_path, "r") as a_file:
        for i in a_file:
            if not 'cluster,connected cluster,PageRank, connection weight' in i:
                edge_info = []
                q = re.split(r'[,]+', i.strip(",\n"))
                cluster_feature = [q[0], float(q[2])]
                cluster_label = 0
                if not cluster_feature in node_feature_raw:
                    node_feature_raw.append(cluster_feature[1])
                    label_raw.append(cluster_label)
                    cluster_nodes.append(cluster_feature[0])
                    pos_idx.append(cluster_feature[0])
                    graph_nodes.append(cluster_feature[0])
    #print("done Network.txt")
    tsv_file = folder + ".tsv"
    tsv_path = tsv_root + tsv_file
    with open(tsv_path, "r") as a_file:
        for i in a_file:
            d = re.split(r'[:\t]+', i.strip(",\n"))
            id_num = d[1]
            id_num = id_num.split('-')
            id_num = id_num[0]
            id_num = id_num.strip('c')
            label_id = d[0] + "_" + id_num
            if float(d[3]) < threshold_e_value:
                label_value = 2# = Member
            elif  float(d[3]) > threshold_e_value:
                label_value = 1# = no Member
            if label_id in d_dict:
                label_taget = [label_id, label_value]
            else:
                label_taget = [label_id, label_value]
            if label_taget[0] in d_dict:
                target_node_feature = 0
                node_feature_raw.append(target_node_feature)
                label_raw.append(label_taget[1])
                target_nodes.append(label_taget[0])
                pos_idx.append(label_taget[0])
                targets.append(label_taget[0])
                graph_nodes.append(label_taget[0])
            elif not label_taget[0] in d_dict:
                continue

    PID_root = Syntney_root + folder + "/PID.txt"
    with open(PID_root, "r") as a_file:
        for index, x in enumerate(a_file):
            if index != 0:
                acc_ids = x.strip().split(" ")[0]
                acc_ids = acc_ids.strip().split("-")[0]
                acc_ids = acc_ids.replace(":", '_')
                if 'c' in acc_ids:
                    acc_ids = acc_ids.split("c")
                    acc_ids = acc_ids[0] + acc_ids[1]
                if acc_ids in d_dict:
                    acc_ids = d_dict[acc_ids]
                    accession.append(acc_ids)
                row = x.split()
                lenght = len(row)
                value_list.append(row)
    for j in range(len(accession)):
        for i in range(len(accession)):
            if i > j:
                edge_list.append([[accession[j], accession[i]], round(float(value_list[i][j + 1]) / 100, 7)])
                edge_list.append([[accession[i], accession[j]], round(float(value_list[i][j + 1]) / 100, 7)])
    for p in edge_list:
        key = tuple(p[0])
        PID_entry = [0, p[1]]
        PID_dict[key] = PID_entry
    for h in PID_dict:
        edge_index.append(list(h))
        edge_attribute.append(PID_dict[h])
    #print("done PID.txt")

    #print("saving files for  " + folder)
    outfile_input = preprocessed_data + folder +  ".txt"
    outfile_input2 = preprocessed_data + folder +  ".idx"
    #outfile_input = preprocessed_dummy + folder + ".txt"
    #outfile_input2 = preprocessed_dummy + folder + ".idx"

    with open(outfile_input, 'w') as f:
        #print(folder)
        f.write("feature")
        f.write(str(node_feature_raw))
        f.write("\n")
        f.write("edge_index")
        f.write(str(edge_index))
        f.write("\n")
        f.write("edge_attribute")
        f.write(str(edge_attribute))
        f.write("\n")
        f.write("label")
        f.write(str(label_raw))
        f.write("\n")
        #f.write("target_nodes")
        #f.write(str(target_nodes))
        #f.write("\n")
        #f.write("cluster_nodes")
        #f.write(str(cluster_nodes))
        #f.write("\n")
    """
    with open(outfile_input2, 'w') as f:
         f.write("pos_idx")
         f.write(str(pos_idx))
         f.write("\n")
         #f.write("targets")
         #f.write(str(targets))
         #f.write("\n")
         f.write("d_dict")
         f.write(str(d_dict))
         f.write("\n")
         #f.write("graph_nodes")
         #f.write(str(graph_nodes))
         #f.write("\n")
    """
pbar.close()
print()
print('saved files for:',"\n", arr)