import numpy as np
import torch


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def convert_to_onehot(sca_label, class_num):
    return np.eye(class_num)[sca_label]


def cal_weight(s_label, t_label, class_num):
    batch_size = int(s_label.size()[0])

    s_sca_label = s_label.cpu().data.numpy()
    s_vec_label = convert_to_onehot(s_sca_label, class_num=class_num)
    s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
    s_sum[s_sum == 0] = 100
    s_vec_label = s_vec_label / s_sum

    t_sca_label = t_label.cpu().data.numpy()
    t_vec_label = convert_to_onehot(t_sca_label, class_num=class_num)
    t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
    t_sum[t_sum == 0] = 100
    t_vec_label = t_vec_label / t_sum

    index = list(set(s_sca_label) & set(t_sca_label))
    mask_arr = np.zeros((batch_size, class_num))
    mask_arr[:, index] = 1
    t_vec_label = t_vec_label * mask_arr
    s_vec_label = s_vec_label * mask_arr

    weight_ss = np.matmul(s_vec_label, s_vec_label.T)
    weight_tt = np.matmul(t_vec_label, t_vec_label.T)
    weight_st = np.matmul(s_vec_label, t_vec_label.T)

    length = len(index)
    if length != 0:
        weight_ss = weight_ss / length
        weight_tt = weight_tt / length
        weight_st = weight_st / length
    else:
        weight_ss = np.array([0])
        weight_tt = np.array([0])
        weight_st = np.array([0])
    return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')


# Local Joint Maximum Mean Discrepancy
def JDA_W(source_list, target_list, s_label, t_label, class_num=13, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68]):
    batch_size = int(source_list[0].size()[0])

    weight_ss, weight_tt, weight_st = cal_weight(s_label, t_label, class_num=class_num)

    weight_ss = torch.from_numpy(weight_ss).cuda()
    weight_tt = torch.from_numpy(weight_tt).cuda()
    weight_st = torch.from_numpy(weight_st).cuda()

    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels

    SS = joint_kernels[:batch_size, :batch_size]
    TT = joint_kernels[batch_size:, batch_size:]
    ST = joint_kernels[:batch_size, batch_size:]

    CDA_loss = torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
    MDA_loss = torch.mean(SS + TT - 2 * ST)

    return CDA_loss + MDA_loss