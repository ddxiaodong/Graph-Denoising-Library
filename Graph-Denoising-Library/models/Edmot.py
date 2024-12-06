""" EdMot clustering class."""

import community
import networkx as nx
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
from tqdm import tqdm
import networkx as nx
import community  # For Louvain partitioning (requires python-louvain)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = EdMot(configs.data, configs.component_count, configs.cutoff)


    def forward(self, x, edge_index):
        return self.model.fit()



# 改造成pyg思路大致是输入的pyg的数据 通过to_networkx 转换后其他函数不用修改 最后return时再转换回去
class EdMotPyg:
    """
    Edge Motif Clustering Class (PyG version).
    """
    def __init__(self, data, component_count, cutoff):
        """
        :param data: PyG Data object.
        :param component_count: Number of extract motif hypergraph components.
        :param cutoff: Motif edge cut-off value.
        """
        self.data = data
        self.graph = to_networkx(data, to_undirected=True)
        self.component_count = component_count
        self.cutoff = cutoff

    def _overlap(self, node_1, node_2):
        """
        Calculating the neighbourhood overlap for a pair of nodes.
        :param node_1: Source node 1.
        :param node_2: Source node 2.
        :return neighbourhood overlap: Overlap score.
        """
        neighbors_1 = set(self.graph.neighbors(node_1))
        neighbors_2 = set(self.graph.neighbors(node_2))
        return len(neighbors_1.intersection(neighbors_2))

    def _calculate_motifs(self):
        """
        Enumerating pairwise motif counts.
        """
        print("\nCalculating overlaps.\n")
        edges = [
            e for e in tqdm(self.graph.edges())
            if self._overlap(e[0], e[1]) >= self.cutoff
        ]
        self.motif_graph = nx.Graph()
        self.motif_graph.add_edges_from(edges)

    def _extract_components(self):
        """
        Extracting connected components from motif graph.
        """
        print("\nExtracting components.\n")
        components = [c for c in nx.connected_components(self.motif_graph)]
        components = [[len(c), c] for c in components]
        components.sort(key=lambda x: x[0], reverse=True)
        self.blocks = [
            list(graph) for graph in
            [components[i][1] for i in range(min(len(components), self.component_count))]
        ]

    def _fill_blocks(self):
        """
        Filling the dense blocks of the adjacency matrix.
        """
        print("\nAdding edge blocks.\n")
        new_edges = [
            (n_1, n_2) for nodes in self.blocks for n_1 in nodes for n_2 in nodes if n_1 != n_2
        ]
        self.graph.add_edges_from(new_edges)

    def fit(self):
        """
        Clustering the target graph.
        """
        self._calculate_motifs()
        self._extract_components()
        self._fill_blocks()
        partition = community.best_partition(self.graph)

        # Convert modified NetworkX graph back to PyG Data
        self.data = from_networkx(self.graph)

        return partition, self.data


class EdMot(object):
    """
    Edge Motif Clustering Class.
    """
    def __init__(self, graph, component_count, cutoff):
        """
        :param graph: NetworkX object.
        :param component_count: Number of extract motif hypergraph components.
        :param cutoff: Motif edge cut-off value.
        """
        self.graph = graph
        self.component_count = component_count
        self.cutoff = cutoff

    def _overlap(self, node_1, node_2):
        """
        Calculating the neighbourhood overlap for a pair of nodes.
        :param node_1: Source node 1.
        :param node_2: Source node 2.
        :return neighbourhood overlap: Overlap score.
        """
        nodes_1 = self.graph.neighbors(node_1)
        nodes_2 = self.graph.neighbors(node_2)
        return len(set(nodes_1).intersection(set(nodes_2)))

    def _calculate_motifs(self):
        """
        Enumerating pairwise motif counts.
        """
        print("\nCalculating overlaps.\n")
        edges = [e for e in tqdm(self.graph.edges()) if self._overlap(e[0], e[1]) >= self.cutoff]
        self.motif_graph = nx.from_edgelist(edges)

    def _extract_components(self):
        """
        Extracting connected components from motif graph.
        """
        print("\nExtracting components.\n")
        components = [c for c in nx.connected_components(self.motif_graph)]
        components = [[len(c), c] for c in components]
        components.sort(key=lambda x: x[0], reverse=True)
        important_components = [components[comp][1] for comp
                                in range(self.component_count if len(components)>=self.component_count else len(components))]
        self.blocks = [list(graph) for graph in important_components]

    def _fill_blocks(self):
        """
        Filling the dense blocks of the adjacency matrix.
        """
        print("Adding edge blocks.\n")
        new_edges = [(n_1, n_2) for nodes in self.blocks for n_1 in nodes for n_2 in nodes if n_1!= n_2]
        self.graph.add_edges_from(new_edges)

    def fit(self):
        """
        Clustering the target graph.
        """
        self._calculate_motifs()
        self._extract_components()
        self._fill_blocks()
        partition = community.best_partition(self.graph)
        return partition
