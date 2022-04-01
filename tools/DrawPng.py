import re

import matplotlib

import matplotlib.pyplot as plt

import numpy as np



def create_multi_bars(labels, datas, tick_step=6, group_gap=0.8, bar_gap=0):

    '''

    labels : x轴坐标标签序列

    datas ：数据集，二维列表，要求列表每个元素的长度必须与labels的长度一致

    tick_step ：默认x轴刻度步长为1，通过tick_step可调整x轴刻度步长。

    group_gap : 柱子组与组之间的间隙，最好为正值，否则组与组之间重叠

    bar_gap ：每组柱子之间的空隙，默认为0，每组柱子紧挨，正值每组柱子之间有间隙，负值每组柱子之间重叠

    '''

    # ticks为x轴刻度

    ticks = np.arange(len(labels)) * tick_step

    # group_num为数据的组数，即每组柱子的柱子个数

    group_num = len(datas)

    # group_width为每组柱子的总宽度，group_gap 为柱子组与组之间的间隙。

    group_width = tick_step - group_gap

    # bar_span为每组柱子之间在x轴上的距离，即柱子宽度和间隙的总和

    bar_span = group_width / group_num

    # bar_width为每个柱子的实际宽度

    bar_width = bar_span - bar_gap

    # baseline_x为每组柱子第一个柱子的基准x轴位置，随后的柱子依次递增bar_span即可

    baseline_x = ticks - (group_width - bar_span) / 2

    for index, y in enumerate(datas):

        plt.bar(baseline_x + index*bar_span, y, bar_width)

        #plt.ylabel('Scores')

        #plt.title('multi datasets')

        # x轴刻度标签位置与x轴刻度一致

        plt.xticks(ticks, labels,rotation=60,fontsize = 5)

        plt.yticks(np.linspace(0, 1, 20))
    plt.show()
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


mR_VSL = '(on:0.7939) (has:0.7709) (wearing:0.9647) (of:0.6290) (in:0.4048) (near:0.4227) (behind:0.5627) (with:0.1836) (holding:0.6967) (above:0.1542) (sitting on:0.2655) (wears:0.0000) (under:0.4053) (riding:0.3124) (in front of:0.1866) (standing on:0.0713) (at:0.3414) (carrying:0.3473) (attached to:0.0148) (walking on:0.2315) (over:0.1650) (for:0.0622) (looking at:0.0890) (watching:0.3067) (hanging from:0.0996) (laying on:0.0300) (eating:0.3845) (and:0.0118) (belonging to:0.0000) (parked on:0.0598) (using:0.1240) (covering:0.0258) (between:0.0312) (along:0.0459) (covered in:0.2083) (part of:0.0000) (lying on:0.0000) (on back of:0.0000) (to:0.0328) (walking in:0.0070) (mounted on:0.0000) (across:0.0317) (against:0.0081) (from:0.0141) (growing on:0.0000) (painted on:0.0000) (playing:0.0000) (made of:0.0000) (says:0.0000) (flying in:0.0000)'
predicatelist,problist_mR_VSL = filterSRC(mR_VSL)
mR_VS = '(on:0.7969) (has:0.7955) (wearing:0.9683) (of:0.6460) (in:0.3985) (near:0.4441) (behind:0.6023) (with:0.1396) (holding:0.6989) (above:0.1665) (sitting on:0.2985) (wears:0.0000) (under:0.3868) (riding:0.3430) (in front of:0.1548) (standing on:0.0525) (at:0.3263) (carrying:0.2859) (attached to:0.0096) (walking on:0.2031) (over:0.1517) (for:0.0672) (looking at:0.0928) (watching:0.3136) (hanging from:0.0866) (laying on:0.0717) (eating:0.4297) (and:0.0207) (belonging to:0.0000) (parked on:0.0575) (using:0.1929) (covering:0.0345) (between:0.0139) (along:0.0642) (covered in:0.1988) (part of:0.0000) (lying on:0.0102) (on back of:0.0000) (to:0.0205) (walking in:0.0070) (mounted on:0.0000) (across:0.0079) (against:0.0000) (from:0.0141) (growing on:0.0000) (painted on:0.0000) (playing:0.0000) (made of:0.0000) (says:0.0000) (flying in:0.0000)'
_,problist_mR_VS = filterSRC(mR_VS)
tick_label = predicatelist
data = [problist_mR_VSL,problist_mR_VS]
# create_multi_bars(label,data)

x = np.arange(len(tick_label))
width = 0.3
plt.bar(x, problist_mR_VSL, width, align="center", color="b", label="VSL", alpha=0.5)
plt.bar(x+width, problist_mR_VS, width, align="center", color="r", label="VS", alpha=0.5)

plt.xticks(x+width/2, tick_label)
plt.show()