import re

import matplotlib

import matplotlib.pyplot as plt

import numpy as np
import torch
import string

f = open("/home/share/zhanghao/data/image/datasets/output/CBLoss_VSL/LossWeight.txt", "w")

f.writelines("Loss of rel is [CBLoss]!")
f.writelines("per_cls_weights")
f.close()

cls_num_list = [118037, 69007, 48582, 32770, 24470, 20759, 13047, 12215, 11482, 8411, 5355, 4939, 4732, 4507, 3808, 2496, 2109, 1705, 1586, 1322, 1277, 1116, 1026, 907, 807, 779, 702, 679, 652, 641, 580, 512, 511, 493, 485, 435, 369, 343, 327, 289, 265, 263, 224, 198, 172, 153, 134, 128, 49, 5]


beta = 0.9999
effective_num = 1.0 - np.power(beta, cls_num_list)
per_cls_weights = (1.0 - beta) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
res = [0]

per_cls_weights=per_cls_weights.tolist()
per_cls_weights.insert(0,0)

per_cls_weights = torch.FloatTensor(per_cls_weights)
torch.backends.cudnn.benchmark = True

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
# label = ['G1', 'G2', 'G3', 'G4', 'G5']
#
# first = [20, 34, 30, 35, 27]
#
# second = [25, 32, 34, 20, 25]
#
# third = [21, 31, 37, 21, 28]
#
# fourth = [26, 31, 35, 27, 21]
#
# data = [first, second, third, fourth]

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



import json
with open('/home/share/zhanghao/data/image/datasets/output/Motifs_VG_PredCls_WithoutVisual2/inference_final/VG_stanford_filtered_with_attribute_test/visual_info.json', 'r') as f:
    visual_info = json.load(f)

with open('/home/zhanghao/code/SGG/Motif_Codebase/maskrcnn_benchmark/data/datasets/evaluation/vg/vg_test_caption_anno.json', 'r') as f:
    vg_test_caption_anno = json.load(f)

with open('/home/zhanghao/code/SGG/Motif_Codebase/maskrcnn_benchmark/data/datasets/evaluation/vg/vg_test_capgraph_anno.json', 'r') as f:
    vg_test_capgraph_anno = json.load(f)
num = vg_test_capgraph_anno['coco_ids'].count(None)
mR_VSL = '(on:0.7939) (has:0.7709) (wearing:0.9647) (of:0.6290) (in:0.4048) (near:0.4227) (behind:0.5627) (with:0.1836) (holding:0.6967) (above:0.1542) (sitting on:0.2655) (wears:0.0000) (under:0.4053) (riding:0.3124) (in front of:0.1866) (standing on:0.0713) (at:0.3414) (carrying:0.3473) (attached to:0.0148) (walking on:0.2315) (over:0.1650) (for:0.0622) (looking at:0.0890) (watching:0.3067) (hanging from:0.0996) (laying on:0.0300) (eating:0.3845) (and:0.0118) (belonging to:0.0000) (parked on:0.0598) (using:0.1240) (covering:0.0258) (between:0.0312) (along:0.0459) (covered in:0.2083) (part of:0.0000) (lying on:0.0000) (on back of:0.0000) (to:0.0328) (walking in:0.0070) (mounted on:0.0000) (across:0.0317) (against:0.0081) (from:0.0141) (growing on:0.0000) (painted on:0.0000) (playing:0.0000) (made of:0.0000) (says:0.0000) (flying in:0.0000)'
predicatelist,problist_mR_VSL = filterSRC(mR_VSL)
# VSL_Dict = dict(zip(predicatelist,problist_mR_VSL))
# print(dic_order_value_and_get_key(VSL_Dict,10))


mR_VS = '(on:0.7969) (has:0.7955) (wearing:0.9683) (of:0.6460) (in:0.3985) (near:0.4441) (behind:0.6023) (with:0.1396) (holding:0.6989) (above:0.1665) (sitting on:0.2985) (wears:0.0000) (under:0.3868) (riding:0.3430) (in front of:0.1548) (standing on:0.0525) (at:0.3263) (carrying:0.2859) (attached to:0.0096) (walking on:0.2031) (over:0.1517) (for:0.0672) (looking at:0.0928) (watching:0.3136) (hanging from:0.0866) (laying on:0.0717) (eating:0.4297) (and:0.0207) (belonging to:0.0000) (parked on:0.0575) (using:0.1929) (covering:0.0345) (between:0.0139) (along:0.0642) (covered in:0.1988) (part of:0.0000) (lying on:0.0102) (on back of:0.0000) (to:0.0205) (walking in:0.0070) (mounted on:0.0000) (across:0.0079) (against:0.0000) (from:0.0141) (growing on:0.0000) (painted on:0.0000) (playing:0.0000) (made of:0.0000) (says:0.0000) (flying in:0.0000)'
_,problist_mR_VS = filterSRC(mR_VS)
# VS_Dict = dict(zip(predicatelist,problist_mR_VS))
# print(dic_order_value_and_get_key(VS_Dict,10))
problist_mR_VS_num = []


for ite in problist_mR_VS:
    problist_mR_VS_num.append(float(ite))
    print(float(ite))
print(sum(problist_mR_VS_num)/50)

mR_SL = "(on:0.7778) (has:0.7910) (wearing:0.9639) (of:0.6437) (in:0.3966) (near:0.4527) (behind:0.5662) (with:0.1555) (holding:0.7047) (above:0.1776) (sitting on:0.3328) (wears:0.0000) (under:0.4058) (riding:0.4002) (in front of:0.1755) (standing on:0.0637) (at:0.3634) (carrying:0.3290) (attached to:0.0184) (walking on:0.2325) (over:0.1322) (for:0.0631) (looking at:0.0910) (watching:0.3494) (hanging from:0.0872) (laying on:0.0848) (eating:0.4069) (and:0.0118) (belonging to:0.0000) (parked on:0.0548) (using:0.2010) (covering:0.0611) (between:0.0278) (along:0.0887) (covered in:0.1929) (part of:0.0000) (lying on:0.0000) (on back of:0.0000) (to:0.0328) (walking in:0.0000) (mounted on:0.0000) (across:0.0317) (against:0.0081) (from:0.0188) (growing on:0.0000) (painted on:0.0000) (playing:0.0000) (made of:0.0000) (says:0.0000) (flying in:0.0000)"
_,problist_mR_SL = filterSRC(mR_SL)
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
#print("-=-==--=-==-=-==-=")
# print(sum(maxlist_num)/50)
#
# for i in maxlist_num:
#     print(i)


Cls_Nums = "118037;69007;48582;32770;24470;20759;13047;12215;11482;8411;5355;4939;4732;4507;3808;2496;2109;1705;1586;1322;1277;1116;1026;907;807;779;702;679;652;641;580;512;511;493;485;435;369;343;327;289;265;263;224;198;172;153;134;128;49;5"
Cls_Nums = Cls_Nums.split(";")
Sample_list = []
for i in Cls_Nums:
    Sample_list.append(int(i))

print(Sample_list)