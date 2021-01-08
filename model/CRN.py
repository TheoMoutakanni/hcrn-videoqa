import numpy as np
import itertools

import torch
import torch.nn as nn
from torch.nn.modules.module import Module


def _moveaxis(tensor: torch.Tensor, source: int, destination: int) -> torch.Tensor:
    # From https://github.com/pytorch/pytorch/issues/36048
    # Implemented in 1.7
    dim = tensor.dim()
    perm = list(range(dim))
    if destination < 0:
        destination += dim
    perm.pop(source)
    perm.insert(destination, source)
    return tensor.permute(*perm)


class LambdaModule(nn.Module):
    def __init__(self, fun):
        super().__init__()
        self.fun = fun

    def forward(self, x):
        return self.fun(x)



class NetVlad(nn.Module):
    def __init__(self, input_size, cluster_size):
        super().__init__()
        self.assignement_fc = nn.Linear(input_size, cluster_size)
        self.mu = nn.Parameter(torch.rand(cluster_size, input_size),
                               requires_grad=True)
        self.cluster_size = cluster_size

    def forward(self, x):
        batch, feat, num = x.shape
        x = x.permute(0, 2, 1)
        a = torch.softmax(self.assignement_fc(x), 2).unsqueeze(2)

        mu = self.mu.expand(batch, num, -1, -1)
        vlad = (a * (x.unsqueeze(3) - mu)).sum(1)

        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(batch, -1)
        vlad = F.normalize(vlad, p=2, dim=1)

        return vlad


class FVNet(nn.Module):
    def __init__(self, input_size, cluster_size):
        super().__init__()
        self.assignement_fc = nn.Linear(input_size, cluster_size)
        self.mu = nn.Parameter(torch.rand(input_size, cluster_size))
        self.sigma = nn.Parameter(torch.rand(input_size, cluster_size))

    def forward(self, x):
        batch, feat, num = x.shape
        x = x.permute(0, 2, 1)
        a = torch.softmax(self.assignement_fc(x), 2).unsqueeze(2)

        mu = self.mu.expand(batch, num, -1, -1)
        sigma = self.sigma.expand(batch, num, -1, -1)

        f = (x.unsqueeze(3) - mu) / sigma

        fv1 = (a * f).sum(1)
        fv2 = (a * (f**2 - 1)).sum(1)

        fv1 = F.normalize(fv1, p=2, dim=2)
        fv1 = fv1.view(batch, -1)
        fv1 = F.normalize(fv1, p=2, dim=1)

        fv2 = F.normalize(fv2, p=2, dim=2)
        fv2 = fv2.view(batch, -1)
        fv2 = F.normalize(fv2, p=2, dim=1)

        fv = torch.cat([fv1, fv2], 1)
        return fv


class NetRVlad(nn.Module):
    def __init__(self, input_size, nb_cluster, dim=1):
        super().__init__()
        self.nb_cluster = nb_cluster
        self.assignement_fc = nn.Linear(input_size, nb_cluster)
        self.dim = dim

    def forward(self, x):
        feat = x.shape[-1]
        # batch, ... , num, feat
        x = _moveaxis(x, self.dim, -2)
        # batch, ... , num, clust
        a = torch.softmax(self.assignement_fc(x), dim=-1)
        # batch, ..., num, clust, feat
        a_x = torch.einsum('...j,...k->...jk', a, x)
        # batch, ... , clust, feat
        x = a_x.sum(-3) / a.sum(-2).unsqueeze(-1)
        x = x.view(*x.shape[:-2], self.nb_cluster*feat)

        return x


class CRN(Module):
    def __init__(self, module_dim, num_objects, max_subset_size, gating=False, spl_resolution=1, num_cluster_g=1, num_cluster_p=1):
        super(CRN, self).__init__()
        self.module_dim = module_dim
        self.gating = gating

        self.k_objects_fusion = nn.ModuleList()
        self.g_agg = nn.ModuleList()
        if num_cluster_g > 1:
            pooling_g = NetRVlad(module_dim, num_cluster_g)
        self.p_agg = nn.ModuleList()
        if num_cluster_p > 1:
            pooling_p = NetRVlad(module_dim, num_cluster_p)
        if self.gating:
            self.gate_k_objects_fusion = nn.ModuleList()
        for i in range(min(num_objects, max_subset_size + 1), 1, -1):
            self.k_objects_fusion.append(nn.Linear((num_cluster_g + 1) * module_dim, module_dim))
            if num_cluster_g == 1:
                self.g_agg.append(LambdaModule(lambda x: torch.mean(x, 1)))
            else:
                self.g_agg.append(pooling_g)
            if num_cluster_p == 1:
                self.p_agg.append(LambdaModule(lambda x: torch.mean(x, 0)))
            else:
                self.p_agg.append(pooling_p)
            if self.gating:
                self.gate_k_objects_fusion.append(nn.Linear((num_cluster_g + 1) * module_dim, module_dim))
        self.spl_resolution = spl_resolution
        self.activation = nn.ELU()
        self.max_subset_size = max_subset_size

    def forward(self, object_list, cond_feat):
        """
        :param object_list: list of tensors or vectors
        :param cond_feat: conditioning feature
        :return: list of output objects
        """
        scales = [i for i in range(len(object_list), 1, -1)]

        relations_scales = []
        subsample_scales = []
        for scale in scales:
            relations_scale = self.relationset(len(object_list), scale)
            relations_scales.append(relations_scale)
            subsample_scales.append(min(self.spl_resolution, len(relations_scale)))

        crn_feats = []
        if len(scales) > 1 and self.max_subset_size == len(object_list):
            start_scale = 1
        else:
            start_scale = 0
        for scaleID in range(start_scale, min(len(scales), self.max_subset_size)):
            idx_relations_randomsample = np.random.choice(len(relations_scales[scaleID]),
                                                          subsample_scales[scaleID], replace=False)
            mono_scale_features = []
            for id_choice, idx in enumerate(idx_relations_randomsample):
                clipFeatList = [object_list[obj].unsqueeze(1) for obj in relations_scales[scaleID][idx]]
                clipFeatList = torch.cat(clipFeatList, dim=1)
                g_feat = self.g_agg[scaleID](clipFeatList)
                if len(g_feat.size()) == 2:
                    h_feat = torch.cat((g_feat, cond_feat), dim=-1)
                elif len(g_feat.size()) == 3:
                    if len(cond_feat.size()) == 3 and cond_feat.size(1) == g_feat.size(1):
                        h_feat = torch.cat((g_feat, cond_feat), dim=-1)
                    elif len(cond_feat.size()) == 2:
                        cond_feat_repeat = cond_feat.unsqueeze(1).repeat(1, g_feat.size(1), 1)
                        h_feat = torch.cat((g_feat, cond_feat_repeat), dim=-1)
                    else:
                        cond_feat_repeat = cond_feat.repeat(1, g_feat.size(1), 1)
                        h_feat = torch.cat((g_feat, cond_feat_repeat), dim=-1)
                if self.gating:
                    h_feat = self.activation(self.k_objects_fusion[scaleID](h_feat)) * torch.sigmoid(
                        self.gate_k_objects_fusion[scaleID](h_feat))
                else:
                    h_feat = self.activation(self.k_objects_fusion[scaleID](h_feat))
                mono_scale_features.append(h_feat)
            mono_scale_features = torch.stack(mono_scale_features)
            mono_scale_features_agg = self.p_agg[scaleID](mono_scale_features)
            crn_feats.append(mono_scale_features_agg)
        return crn_feats

    def relationset(self, num_objects, num_object_relation):
        return list(itertools.combinations([i for i in range(num_objects)], num_object_relation))
