import torch
from torch import Tensor

# feature_align(nodes, p, n_p, (256, 256))
# nodes: [batch_size, 512, 32, 32]
# edges: [batch_size, 512, 16, 16]
def feature_align(raw_feature: Tensor, P: Tensor, ns_t: Tensor, ori_size: tuple, device=None):
    """
    Perform feature align from the raw feature map.
    :param raw_feature: raw feature map
    :param P: point set containing point coordinates
    :param ns_t: number of exact points in the point set
    :param ori_size: size of the original image
    :param device: device. If not specified, it will be the same as the input
    :return: F
    """
    if device is None:
        device = raw_feature.device
    f_dim = raw_feature.shape[-1] # feature_map 维度
    
    ori_size_t = torch.tensor(ori_size, dtype=torch.float32, device=device)
    step = ori_size_t[0] / f_dim # image 与 feature map 大小比例

    channel_num = raw_feature.shape[1]
    n_max = P.shape[1]
    bs = raw_feature.shape[0]

    p_calc = (P - step / 2) / step
    
    p_floor = p_calc.floor()
    shifts = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=device)
    p_shifted = torch.stack([p_floor + shift for shift in shifts])
    
    # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。
    p_shifted_clamped = p_shifted.clamp(0, f_dim - 1)
    p_shifted_flat = p_shifted_clamped[..., 1] * f_dim + p_shifted_clamped[..., 0]

    w_feat = 1 - (p_calc - p_shifted).abs()
    w_feat_mul = w_feat[..., 0] * w_feat[..., 1]

    raw_features_flat = raw_feature.flatten(2, 3)

    # mask to disregard information in keypoints that don't matter (meaning that for the given image the number of keypoints is smaller than the maximum number in the batch)
    mask = torch.zeros(bs, n_max, device=device)
    for i in range(bs):
        mask[i][0 : ns_t[i]] = 1
    mask = mask.unsqueeze(1).expand(bs, channel_num, n_max)

    raw_f_exp = raw_features_flat.unsqueeze(0).expand(4, bs, channel_num, f_dim ** 2)
    p_flat_exp = p_shifted_flat.unsqueeze(2).expand(4, bs, channel_num, n_max).long()
    features = raw_f_exp.gather(3, p_flat_exp)
    w_exp = w_feat_mul.unsqueeze(2).expand(4, bs, channel_num, n_max)
    f = torch.sum(features * w_exp, dim=0) * mask
    return f
