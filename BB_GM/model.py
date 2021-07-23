import torch
from torch import nn

import utils.backbone
from BB_GM.affinity_layer import InnerProductWithWeightsAffinity
from BB_GM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
from lpmp_py import GraphMatchingModule
from lpmp_py import MultiGraphMatchingModule
from utils.config import cfg
from utils.feature_align import feature_align
from utils.utils import lexico_iter
from utils.visualization import easy_visualize


def normalize_over_channels(x):
    # 这 N 个数据求 p 范数
    channel_norms = torch.norm(x, dim=1, keepdim=True)
    return x / channel_norms

def concat_features(embeddings, num_vertices):
    res = torch.cat([embedding[:, :num_v] for embedding, num_v in zip(embeddings, num_vertices)], dim=-1)
    return res.transpose(0, 1) # 转置为列向量


# class Net(utils.backbone.VGG16_bn):
class Net(utils.backbone.VGG16_bn):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=1024)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = 1024
        self.vertex_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim, self.message_pass_node_features.num_node_features)
        self.edge_affinity = InnerProductWithWeightsAffinity(
            self.global_state_dim,
            self.build_edge_features_from_node_features.num_edge_features)

        self.fusion_global = nn.Linear(in_features=24, out_features=512)
        self.local_topo_expand = nn.Sequential(nn.Linear(8, 1024),nn.Sigmoid())


    def forward(
        self,
        images,
        points,
        graphs,
        n_points,
        perm_mats,
        loc_topo_feats,
        glob_topo_feats,
        graph_feats,
        visualize_flag=False,
        visualization_params=None,
    ):
        global_list = []
        global_graph_topo_list = []
        orig_graph_list = []

        # 分别对 source 和 target 的 batch_size 个图片
        for image, p, n_p, graph, loc_topo_feat, glob_topo_feat, graph_feat in zip(images, points, n_points, graphs, loc_topo_feats,
                                                                       glob_topo_feats, graph_feats):
            # extract feature
            nodes = self.node_layers(image) # [batch_size, feature_dim, featuremap_size, featuremap_size]
            edges = self.edge_layers(nodes) # [batch_size, feature_dim, featuremap_size, featuremap_size]

            # print('nodes=', nodes.shape)
            # print('edges=', edges.shape)
            # print('final_layers(edges)=', self.final_layers(edges)[0].shape)
            # print('self.final_layers(edges)=', self.final_layers(edges)[0].reshape((nodes.shape[0], -1)).shape)
            
            # self.final_layers(edges)[0].shape: [batch_size, feature_dim, 1, 1]
            # self.final_layers(edges)[0].reshape((nodes.shape[0], -1)): [batch_size, feature_dim]
            global_list.append(self.final_layers(edges)[0].reshape((nodes.shape[0], -1)))
            
            # 特征归一化，不改变维度
            nodes = normalize_over_channels(nodes)
            edges = normalize_over_channels(edges)

            # arrange features
            # feature_align 将关键点的特征向量提取出来
            # U: [n_p, feature_dim]
            # F: [n_p, feature_dim]
            U = concat_features(feature_align(nodes, p, n_p, (256, 256)), n_p)
            F = concat_features(feature_align(edges, p, n_p, (256, 256)), n_p)
            # 此时节点特征维度为1024,值范围[0, 1]
            node_features = torch.cat((U, F), dim=-1)

            # 特征转置
            loc_topo_feat = loc_topo_feat.permute(0, 2, 1)
            glob_topo_feat = glob_topo_feat.permute(0, 2, 1)

            # 找出所有点的特征
            loc_topo_feat = concat_features(loc_topo_feat, n_p)
            glob_topo_feat = concat_features(glob_topo_feat, n_p)
            graph_feat = self.fusion_global(graph_feat)
            global_graph_topo_list.append(graph_feat)

            # 特征拼接
            # topo_feat = torch.cat((loc_topo_feat, glob_topo_feat), 1)
            topo_feat = self.local_topo_expand(loc_topo_feat)

            graph.x = node_features+topo_feat

            graph = self.message_pass_node_features(graph)
            orig_graph = self.build_edge_features_from_node_features(graph)
            orig_graph_list.append(orig_graph)

        # global_weights_list = [
        #     torch.cat([global_src, global_tgt], dim=-1)+torch.cat([graph_topo_src, graph_topo_tgt], dim=-1) for (global_src, global_tgt),(graph_topo_src,graph_topo_tgt) in zip(lexico_iter(global_list), lexico_iter(global_graph_topo_list))
        # ]
        global_weights_list = [
            torch.cat([global_src, global_tgt], axis=-1) for global_src, global_tgt in lexico_iter(global_list)
        ]

        global_weights_list = [normalize_over_channels(g) for g in global_weights_list]

        # print(g_1)
        unary_costs_list = [
            self.vertex_affinity([item.x for item in g_1], [item.x for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Similarities to costs
        unary_costs_list = [[-x for x in unary_costs] for unary_costs in unary_costs_list]

        if self.training:
            unary_costs_list = [
                [
                    x + 1.0*gt[:dim_src, :dim_tgt]  # Add margin with alpha = 1.0
                    for x, gt, dim_src, dim_tgt in zip(unary_costs, perm_mat, ns_src, ns_tgt)
                ]
                for unary_costs, perm_mat, (ns_src, ns_tgt) in zip(unary_costs_list, perm_mats, lexico_iter(n_points))
            ]

        quadratic_costs_list = [
            self.edge_affinity([item.edge_attr for item in g_1], [item.edge_attr for item in g_2], global_weights)
            for (g_1, g_2), global_weights in zip(lexico_iter(orig_graph_list), global_weights_list)
        ]

        # Aimilarities to costs
        quadratic_costs_list = [[-0.5 * x for x in quadratic_costs] for quadratic_costs in quadratic_costs_list]

        if cfg.BB_GM.solver_name == "lpmp":
            all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
            gm_solvers = [
                GraphMatchingModule(
                    all_left_edges,
                    all_right_edges,
                    ns_src,
                    ns_tgt,
                    cfg.BB_GM.lambda_val,
                    cfg.BB_GM.solver_params,
                )
                for (all_left_edges, all_right_edges), (ns_src, ns_tgt) in zip(
                    lexico_iter(all_edges), lexico_iter(n_points)
                )
            ]
            matchings = [
                gm_solver(unary_costs, quadratic_costs)
                for gm_solver, unary_costs, quadratic_costs in zip(gm_solvers, unary_costs_list, quadratic_costs_list)
            ]
        elif cfg.BB_GM.solver_name == "multigraph":
            all_edges = [[item.edge_index for item in graph] for graph in orig_graph_list]
            gm_solver = MultiGraphMatchingModule(
                all_edges, n_points, cfg.BB_GM.lambda_val, cfg.BB_GM.solver_params)
            matchings = gm_solver(unary_costs_list, quadratic_costs_list)
        else:
            raise ValueError(f"Unknown solver {cfg.BB_GM.solver_name}")

        if visualize_flag:
            easy_visualize(
                orig_graph_list,
                points,
                n_points,
                images,
                unary_costs_list,
                quadratic_costs_list,
                matchings,
                **visualization_params,
            )

        return matchings
