import math
import re

import matplotlib

import matplotlib.pyplot as plt

import numpy as np
import torch
import string
import torch.nn as nn


for m in range(1):
    print(0)

tensor1 = torch.Tensor([3])
res = 3+tensor1.size(0)


tensor1 = torch.Tensor([1, 2, 3])
tensor2 = torch.Tensor([4, 5, 6])
tensor_list = list()
tensor_list.append(tensor1)
tensor_list.append(tensor2)
final_tensor = torch.stack(tensor_list)  ###
print('tensor_list:', tensor_list, '  type:', type(tensor_list))
print('final_tensor:', final_tensor, '  type', type(final_tensor))

cls_num_list = [118037, 118037, 69007, 48582, 32770, 24470, 20759, 13047, 12215, 11482, 8411, 5355, 4939, 4732,
                    4507, 3808, 2496, 2109, 1705, 1586, 1322, 1277, 1116, 1026, 907, 807, 779, 702, 679, 652, 641, 580,
                    512, 511, 493, 485, 435, 369, 343, 327, 289, 265, 263, 224, 198, 172, 153, 134, 128, 49, 5]


per_cls_weights = []


for i in range(len(cls_num_list)):
    if i<25:
        per_cls_weights.append(cls_num_list[25]/cls_num_list[i])
    else:
        per_cls_weights.append(1)

    # beta = 0.9999
    # effective_num = 1.0 - np.power(beta, cls_num_list)
    # per_cls_weights = (1.0 - beta) / np.array(effective_num)
    # per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    #
    # per_cls_weights = per_cls_weights.tolist()


print(per_cls_weights)

res = []
for i in range(len(per_cls_weights)):
    res.append(per_cls_weights[i]*cls_num_list[i])

print(res)
per_cls_weights = []





# input = torch.tensor([[ 0.0500,  0.0734,  0.5450, -0.8221, -0.1549],
#         [ 0.0045,  0.4474, -1.2117,  0.3814, -0.5574],
#         [-0.1080,  0.1423,  0.4503,  0.7007,  0.7402]], requires_grad=True)
# target = torch.tensor([0, 0, 2])
# # def CE_torch(input,target,weight):
# #
# #     first = 0  # 第一项
# #     second = 0  # 第二项
# #     s = input[i][target[i]] * weight[target[i]]
# #     for i in range(target.size(0)):
# #
# #         first += -input[i][target[i]]*weight[target[i]]
# #         tempSum = 0
# #         for j in range(input.size(1)):
# #             tempSum += torch.exp(input[i][j])
# #         second += torch.log(tempSum)*weight[target[i]]
# #     res =(first+second)/target.size(0)
# #     return res
# def CE(input,target,weight):
#
#     first = 0  # 第一项
#     second = 0  # 第二项
#     for i in range(target.size(0)):
#         first += -input[i][target[i]]*weight[target[i]]
#         tempSum = 0
#         for j in range(input.size(1)):
#             tempSum += torch.exp(input[i][j])
#         second += torch.log(tempSum)*weight[target[i]]
#     res =(first+second)/target.size(0)
#     return res
#
#
#
#
# print("手写："+str(CE(input,target,weight=torch.tensor([5,1,1,1,1]))))
#
#
#
#
# loss_mean = nn.CrossEntropyLoss(weight=torch.tensor([5.,1.,1.,1.,1.]))
#
# loss_sum = nn.CrossEntropyLoss(weight=torch.tensor([5.,1.,1.,1.,1.]),reduction='sum')
#
# loss_none = nn.CrossEntropyLoss(weight=torch.tensor([5.,1.,1.,1.,1.]),reduction='none')
# print("CB_MEAN："+str(loss_mean(input,target)))
# print("CB_SUM："+str(loss_sum(input,target)))
# print("CB_NONE:"+str(loss_none(input,target)))
# print(loss_sum(input,target)/input.size(0))






def filterSRC(src):
    srclist = re.findall(r'[(](.*?)[)]', src)
    predicatelist=[]
    problist=[]
    for item in srclist:
        item = item.replace('(','')
        item = item.replace(')','')
        predicatelist.append(item.split(':')[0])
        problist.append(item.split(':')[1])
    return predicatelist,problist

def dic_order_value_and_get_key(dicts, count):
    # by hellojesson
    # 字典根据value排序，并且获取value排名前几的key
    # 样例： dicts = {'王二狗':66,'李大东':55,'刘小明':99, '胡八一':88}
    final_result = []
    # 先对字典排序
    sorted_dic = sorted([(k, v) for k, v in dicts.items()], reverse=True)
    tmp_set = set()  # 定义集合 会去重元素 --此处存在一个问题，成绩相同的会忽略，有待改进
    for item in sorted_dic:
        tmp_set.add(item[1])
    for list_item in sorted(tmp_set, reverse=True)[:count]:
        for dic_item in sorted_dic:
            if dic_item[1] == list_item:
                final_result.append(dic_item[0])
    return final_result




mR_VSL = '(on:0.1680) (has:0.6331) (wearing:0.6354) (of:0.5805) (in:0.3219) (near:0.1537) (behind:0.5559) (with:0.2723) (holding:0.4704) (above:0.2411) (sitting on:0.4603) (wears:0.3546) (under:0.4916) (riding:0.8595) (in front of:0.3246) (standing on:0.3353) (at:0.6219) (carrying:0.6728) (attached to:0.2400) (walking on:0.8376) (over:0.3280) (for:0.4220) (looking at:0.2670) (watching:0.5693) (hanging from:0.5253) (laying on:0.4861) (eating:0.7068) (and:0.3240) (belonging to:0.2892) (parked on:0.9298) (using:0.4284) (covering:0.4788) (between:0.1750) (along:0.5359) (covered in:0.6586) (part of:0.0412) (lying on:0.0663) (on back of:0.3109) (to:0.4672) (walking in:0.0493) (mounted on:0.0840) (across:0.1944) (against:0.1532) (from:0.0873) (growing on:0.2205) (painted on:0.1914) (playing:0.0909) (made of:0.1094) (says:0.0833) (flying in:0.0000) '
predicatelist,problist_mR_VSL = filterSRC(mR_VSL)
# VSL_Dict = dict(zip(predicatelist,problist_mR_VSL))
# print(dic_order_value_and_get_key(VSL_Dict,10))
# for i in problist_mR_VSL:
#     print(i)
print("**************************")
mR_VS = '(on:0.1769) (has:0.6281) (wearing:0.5665) (of:0.5723) (in:0.3118) (near:0.1438) (behind:0.5633) (with:0.2656) (holding:0.4828) (above:0.2384) (sitting on:0.4331) (wears:0.4407) (under:0.4606) (riding:0.8299) (in front of:0.3129) (standing on:0.3465) (at:0.6904) (carrying:0.6470) (attached to:0.2450) (walking on:0.7172) (over:0.3281) (for:0.4048) (looking at:0.2641) (watching:0.5349) (hanging from:0.4576) (laying on:0.4899) (eating:0.7454) (and:0.3033) (belonging to:0.2719) (parked on:0.9242) (using:0.4137) (covering:0.4572) (between:0.1382) (along:0.3593) (covered in:0.5371) (part of:0.0442) (lying on:0.0531) (on back of:0.3109) (to:0.4795) (walking in:0.0070) (mounted on:0.0694) (across:0.2540) (against:0.1371) (from:0.0704) (growing on:0.1897) (painted on:0.1011) (playing:0.0000) (made of:0.1250) (says:0.0000) (flying in:0.0000) '
_,problist_mR_VS = filterSRC(mR_VS)
# VS_Dict = dict(zip(predicatelist,problist_mR_VS))
# print(dic_order_value_and_get_key(VS_Dict,10))
# for i in problist_mR_VS:
#     print(i)
print("**************************")
problist_mR_VS_num = []


# for ite in problist_mR_VS:
#     problist_mR_VS_num.append(float(ite))

# print(sum(problist_mR_VS_num)/50)

mR_SL = "(on:0.1536) (has:0.6730) (wearing:0.6308) (of:0.6590) (in:0.3280) (near:0.1540) (behind:0.5755) (with:0.2497) (holding:0.4740) (above:0.2769) (sitting on:0.4139) (wears:0.3794) (under:0.4916) (riding:0.8311) (in front of:0.3290) (standing on:0.2674) (at:0.6224) (carrying:0.6067) (attached to:0.2842) (walking on:0.7859) (over:0.3572) (for:0.3898) (looking at:0.1857) (watching:0.4669) (hanging from:0.4353) (laying on:0.4276) (eating:0.7075) (and:0.3383) (belonging to:0.2180) (parked on:0.8961) (using:0.3450) (covering:0.4221) (between:0.1458) (along:0.4373) (covered in:0.5068) (part of:0.0766) (lying on:0.0102) (on back of:0.1939) (to:0.4522) (walking in:0.0070) (mounted on:0.0208) (across:0.0317) (against:0.0403) (from:0.0563) (growing on:0.0388) (painted on:0.1236) (playing:0.0000) (made of:0.0625) (says:0.0000) (flying in:0.0000) "
_,problist_mR_SL = filterSRC(mR_SL)

# for i in problist_mR_SL:
#     print(i)

# for item in problist_mR_SL:
#     print(item)
res_VSL = []
res_VS = []
res_SL = []
res = []
for i in range(len(predicatelist)):
    if problist_mR_VSL[i]>problist_mR_VS[i] and problist_mR_VSL[i]>problist_mR_SL[i]:
        res_VSL.append(predicatelist[i])
    elif problist_mR_VS[i]>problist_mR_VSL[i] and problist_mR_VS[i]>problist_mR_SL[i]:
        res_VS.append(predicatelist[i])
    elif problist_mR_SL[i]>problist_mR_VSL[i] and problist_mR_SL[i]>problist_mR_VS[i]:
        res_SL.append(predicatelist[i])
    else:
        res.append(predicatelist[i])


# print("res_VSL"+str(res_VSL))
# print("res_VS"+str(res_VS))
# print("res_SL"+str(res_SL))
# print("equal"+str(res))
maxlist = []
for i in range(len(predicatelist)):
   maxlist.append(max(problist_mR_VSL[i],problist_mR_SL[i],problist_mR_VS[i]))
maxlist_num = []
for i in maxlist:
    maxlist_num.append(float(i))
print("-=-==--=-==-=-==-=")
print(sum(maxlist_num)/50)
#
# for i in maxlist_num:
#     print(i)
cls_num_list = [63023, 25140, 20148, 19501, 11383, 7394, 4558, 5358, 4657, 2388, 2300, 2324, 1705, 1729, 1276, 1409, 749, 687, 838, 648, 491, 354, 349, 392, 407, 334, 338, 216, 743, 292, 221, 243, 216, 155, 196, 142, 170, 178, 153, 113, 169, 82, 69, 98, 101, 102, 29, 37, 12, 25]
tmpSum = 0
for i in range(len(predicatelist)):
    tmpSum += cls_num_list[i]*maxlist_num[i]
print(tmpSum/sum(cls_num_list))
