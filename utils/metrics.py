import torch
import numpy as np
import os
import ot  
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


# =========================================================================
# [新增] OT Re-ranking 模块 (基于论文 Posture-Aware ... via OT Calibration)
# =========================================================================
def ot_reranking(qf, gf, distmat, top_k=5, lambda_ot=0.5):
    """
    使用 OT 距离对原始距离矩阵进行重排序
    qf: [num_query, feat_dim] 查询特征
    gf: [num_gallery, feat_dim] 库特征
    distmat: 原始欧氏距离矩阵 [num_query, num_gallery]
    top_k: 可信集大小 (Credible Set Size)
    lambda_ot: OT 距离的权重
    """
    print(f"=> Starting OT Re-ranking (Top-{top_k} Credible Set)...")
    num_q, num_g = distmat.shape
    new_distmat = distmat.copy()

    # 将 Tensor 转为 numpy 用于计算 
    if isinstance(qf, torch.Tensor): qf = qf.cpu().numpy()
    if isinstance(gf, torch.Tensor): gf = gf.cpu().numpy()


    rank_scope = 100

    for i in range(num_q):
        # 1. 构建可信集 (Credible Set)
        # 找到当前 Query 在 Gallery 里最像的 top_k 个样本
        # indices 是从小到大排序的索引 (距离越小越像)
        indices = np.argsort(distmat[i, :])[:top_k]
        credible_feats = gf[indices]  # [top_k, dim]

        # 构建分布 P (Source): Query + Credible Set
        # 形状: [1 + top_k, dim]
        query_feat = qf[i].reshape(1, -1)
        source_feats = np.concatenate([query_feat, credible_feats], axis=0)

        # 分布 P 的权重: Query 给 0.5, 剩下的 top_k 平分 0.5
        n_source = 1 + top_k
        a = np.zeros(n_source)
        a[0] = 0.5
        a[1:] = 0.5 / top_k

        # 2. 针对前 rank_scope 的 Gallery 样本计算 OT 距离
        re_rank_indices = np.argsort(distmat[i, :])[:rank_scope]

        for j in re_rank_indices:
            # 构建分布 Q (Target): 单个 Gallery 样本
            target_feat = gf[j].reshape(1, -1)
            b = np.array([1.0])  # 单点分布

            # 计算 Cost Matrix (使用余弦距离: 1 - cos)
            # source: [N, D], target: [1, D]
            # 点积
            dot_prod = np.dot(source_feats, target_feat.T)  # [N, 1]
            # 归一化 (假设特征已经是归一化的，如果没有，需要除以模长)
            # 这里 metrics 里 compute 函数已经做了 normalize，所以直接用
            cost_m = 1 - dot_prod

            # 计算 OT 距离
            # 因为 Target 只有一个点，Sinkhorn 退化为加权求和
            # ot_dist = sum(a_i * cost_i)
            # 如果想严谨一点用 ot 库 (适合 Target 也是集合的情况):
            # M = ot.dist(source_feats, target_feat) # 欧氏距离平方
            # T = ot.emd(a, b, M)
            # ot_dist = np.sum(T * M)

            # 简化版 OT (针对 Point-to-Set 的特殊情况，速度极快)
            ot_d = np.sum(a.reshape(-1, 1) * cost_m)

            # 3. 融合距离
            new_distmat[i, j] = (1 - lambda_ot) * distmat[i, j] + lambda_ot * ot_d

    return new_distmat


# =========================================================================


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        # 新增开关，默认不开启 OT
        self.use_ot_reranking = False

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):  # called after each epoch

        print(f"[DEBUG CHECK] self.use_ot_reranking is: {self.use_ot_reranking}")

        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])

        # 1. 先计算基础距离矩阵
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)
        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)

        # 2. 如果开启了 OT 重排序，就在这里进行
        if self.use_ot_reranking:
            print(f"[DEBUG] OT Re-ranking ACTIVATED! Processing {self.num_query} queries...")
            distmat = ot_reranking(qf, gf, distmat, top_k=5, lambda_ot=0.4)
        else:
            print(f"[DEBUG] OT Re-ranking is OFF (Standard Euclidean).")

        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf