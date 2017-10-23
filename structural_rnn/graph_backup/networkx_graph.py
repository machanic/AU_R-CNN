import argparse
import uuid

import networkx as nx

import config


class NetworkXGraph(object):

    def __init__(self, graph_data_set):
        self.graph_data_set = graph_data_set
        self.networkx_list = list()
        self.node_dict_list = list()  # each entry is dict which store node_id: node

    def __getitem__(self, index):
        return self.networkx_list[index]

    def construct_networkx(self, exclude_edge_type=None):
        for curt_sample in self.graph_data_set.samples:
            G = nx.Graph()
            nodeid_node_dict = dict()
            for node in curt_sample.node_list:
                node_dict = node.__dict__
                node_dict["color"] = str(uuid.uuid1())
                G.add_node(node.id, node_dict)  # also contains feature
                # 每个node和自身也有一个edge
                G.add_edge(node.id, node.id,{"color":str(uuid.uuid1()), "type":"same",
                                            "type_int":len(self.graph_data_set.edge_type_dict.mapping_dict)})
                                            #"feature":np.concatenate((node.feature, node.feature))})
                nodeid_node_dict[node.id] = node
            for edge in curt_sample.edge_list:
                edge_type_str = self.graph_data_set.edge_type_dict.mapping_dict.inv[edge.edge_type]
                if exclude_edge_type and edge_type_str not in exclude_edge_type:  # for example: exclude temporal
                    G.add_edge(edge.a, edge.b,
                               {"color": str(uuid.uuid1()),
                                "type": edge_type_str,
                                "type_int": edge.edge_type})
                elif exclude_edge_type is None:
                    G.add_edge(edge.a, edge.b,
                               {"color": str(uuid.uuid1()),
                                "type": edge_type_str,
                                "type_int": edge.edge_type})
            self.networkx_list.append(G)
            self.node_dict_list.append(nodeid_node_dict)

    def remove_edge_attrib(self, edge_attrib, remove_value):
        for G in self.networkx_list:
            for edge in G.edges_iter(data=True):
                u,v,edge_type_dict = edge
                if edge_type_dict[edge_attrib] == remove_value:
                    G.remove_edge(u, v)

    def partition_graph_by_region(self, remove_temporal_edge=False):

        for G in self.networkx_list:

            for edge in G.edges_iter(data=True):
                u, v,  edge_type_dict = edge
                if edge_type_dict["type"] == "symmetry":
                    color_twin_node = str(uuid.uuid1())
                    # 双胞胎的三种东西需要标成一个颜色
                    # 1. symmetry的双胞胎node需要标成一个颜色
                    # 2. 其自己指向自己的type是same的edge
                    # 3. 双胞胎node对应的同一个外部节点也要edge标成同一个颜色
                    # 4. 还有两个双胞胎之间的symmetry的edge也要标成同一个颜色
                    G.node[u]["color"] = color_twin_node
                    G.node[v]["color"] = color_twin_node
                    color_same_edge = str(uuid.uuid1())
                    color_twin_edge = str(uuid.uuid1())
                    edge_type_dict["color"] = color_twin_edge
                    color_out_edge_dict = {}
                    for twin_edge in G.edges([u, v],data=True):
                        u1, v1, edge_type_dict1 = twin_edge
                        if edge_type_dict1["type"] == "symmetry": continue
                        outter_node_id = u1 if u1 not in [u,v] else v1
                        if outter_node_id not in color_out_edge_dict:
                            color_out_edge_dict[outter_node_id] = str(uuid.uuid1())
                        if edge_type_dict1["type"] == "same":
                            edge_type_dict1["color"] = color_same_edge
                        elif edge_type_dict1["type"] == "spatio":
                            edge_type_dict1["color"] = color_out_edge_dict[outter_node_id]
                elif remove_temporal_edge and edge_type_dict["type"] == "temporal":
                    G.remove_edge(u,v)   # partition level 1: partition to cut temporal link

    def sort_by_frame(self):
        for idx, G in enumerate(self.networkx_list):

            for frame_idx, node_id_set in enumerate(nx.connected_components(G)):
                color_node = set()
                color_edge = set()
                for node_id, node_info in G.nodes(data=True):
                    if node_id in node_id_set:
                        color_node.add(node_info["color"])
                for node_a, node_b, edge_info in G.edges(list(node_id_set), data=True):
                    color_edge.add(edge_info["color"])
                print(frame_idx, len(color_node), len(color_edge))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate Graph desc file script')
    parser.add_argument('--mean', default=config.ROOT_PATH + "BP4D/idx/mean_enhance.npy", help='image mean .npy file')
    parser.add_argument("--graph_desc_dir", default="/home/machen/face_expr/result/graph_backup")
    parser.add_argument('--database', default='BP4D',
                        help='Output directory')
    args = parser.parse_args()
    graph_dir = args.graph_desc_dir
    from structural_rnn.dataset.graph_dataset_reader import GlobalDataSet
    dataset = GlobalDataSet()
    dataset.load_data("/home/machen/face_expr/result/graph_backup/F009_T6.txt")
    print('read done')
    networkx = NetworkXGraph(dataset)
    networkx.construct_networkx(exclude_edge_type=["temporal"])
    print("construct done")
    networkx.partition_graph_by_region()
    networkx.sort_by_frame()