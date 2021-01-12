import numpy as np
import itertools

import einops
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


def repeat_as(x, y):
    if x.shape[:-1] == y.shape[:-1]:
        return x
    if len(x.shape) == len(y.shape):
        dim = np.where(np.array(x.shape[:-1])!=np.array(y.shape[:-1]))[0][0]
        return x.repeat(*(1 if i!= dim else y.size(dim) for i in range(len(y.shape))))
    i = len(x.shape) - 1
    while len(x.shape) < len(y.shape):
        x = einops.repeat(x, '... i -> ... k i', k=y.size(i))
        i += 1

    if x.shape[:-1] != y.shape[:-1]:
        x = repeat_as(x, y)
    return x


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
    def __init__(self, input_size, nb_cluster, dim=1, flatten=True):
        super().__init__()
        self.nb_cluster = nb_cluster
        self.assignement_fc = nn.Linear(input_size, nb_cluster)
        self.mu = nn.Parameter(torch.rand(input_size, nb_cluster))
        self.sigma = nn.Parameter(torch.rand(input_size, nb_cluster))

    def forward(self, x):
        feat = x.shape[-1]
        # batch, ... , num, feat
        x = _moveaxis(x, self.dim, -2)
        a = torch.softmax(self.assignement_fc(x), dim=-1)

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
    def __init__(self, input_size, nb_cluster, dim=1, flatten=True):
        super().__init__()
        self.nb_cluster = nb_cluster
        self.assignement_fc = nn.Linear(input_size, nb_cluster)
        self.dim = dim
        self.flatten = flatten

    def forward(self, x):
        feat = x.shape[-1]
        # batch, ... , num, feat
        x = _moveaxis(x, self.dim, -2)
        # batch, ... , num, clust
        a = torch.softmax(self.assignement_fc(x), dim=-1)
        # batch, ..., num, clust, feat
        a_x = torch.einsum('...ij,...ik->...jk', a, x)
        # batch, ... , clust, feat
        x = a_x / a.sum(-2).unsqueeze(-1)
        #x = a_x.sum(-3) / a.sum(-2).unsqueeze(-1)
        if self.flatten:
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

        if len(range(num_objects, 1, -1)) > 1 and max_subset_size == num_objects:
            start_scale = 1
        else:
            start_scale = 0
        start_scale = 0 #FORCE

        for i in range(min(num_objects, max_subset_size + 1), start_scale, -1):
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
                h_feat = self.cat_cond_feat(g_feat, cond_feat)
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

    def cat_cond_feat(self, g_feat, cond_feat):
        cond_feat_repeat = repeat_as(cond_feat, g_feat)
        return torch.cat((g_feat, cond_feat_repeat), dim=-1)


class FasterCRN(Module):
    def __init__(self, module_dim, num_objects, max_subset_size, gating=False, spl_resolution=1, dim=-2):
        super(FasterCRN, self).__init__()
        self.module_dim = module_dim
        self.gating = gating

        self.k_objects_fusion = nn.ModuleList()
        self.g_agg = nn.ModuleList()
        if self.gating:
            self.gate_k_objects_fusion = nn.ModuleList()
        for t in range(spl_resolution):
            self.k_objects_fusion.append(nn.Linear(2*module_dim, module_dim))
            if max_subset_size == 1:
                self.g_agg.append(LambdaModule(lambda x: torch.mean(x, dim)))
            else:
                self.g_agg.append(NetRVlad(module_dim, max_subset_size-2, dim=dim, flatten=False))
            if self.gating:
                self.gate_k_objects_fusion.append(nn.Linear(2*module_dim, module_dim))
        self.p_agg = LambdaModule(lambda x: torch.mean(x, 1))
        self.spl_resolution = spl_resolution
        self.activation = nn.ELU()
        self.max_subset_size = max_subset_size

    def forward(self, object_list, cond_feat):
        """
        :param object_list: tensors (batch, ..., num_objects, feat)
        :param cond_feat: conditioning feature
        :return: list of output objects
        """
        features = []
        for t in range(self.spl_resolution):
            g_feat = self.g_agg[t](object_list) # (batch, ..., max_subset_size, feat)
            h_feat = self.cat_cond_feat(g_feat, cond_feat)
            if self.gating:
                h_feat = self.activation(self.k_objects_fusion[t](h_feat)) * torch.sigmoid(
                    self.gate_k_objects_fusion[t](h_feat))
            else:
                h_feat = self.activation(self.k_objects_fusion[t](h_feat))
            features.append(h_feat)

        features = torch.stack(features, 1) # (batch, spl_resolution, ..., max_subset_size, feat)
        features_agg = self.p_agg(features) # (batch, ..., max_subset_size, feat)
        return features_agg

    def cat_cond_feat(self, g_feat, cond_feat):
        cond_feat_repeat = repeat_as(cond_feat, g_feat)
        return torch.cat((g_feat, cond_feat_repeat), dim=-1)