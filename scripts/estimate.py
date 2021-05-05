#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/4 18:07
# @Author  : JJkinging
# @File    : estimate.py

'''该部分是计算准确率、召回率、F1值的评估函数
<pad>   0
B-PER   1
I-PER   2
B-LOC   3
I-LOC   4
B-ORG   5
I-ORG   6
O       7
<START> 8
<EOS>   9
'''
tags = [(1, 2), (3, 4), (5, 6)]

def find_tag(input, B_label_id=1, I_label_id=2):
    '''
    找到指定的label
    :param input: 模型预测输出的路径 shape = [batch的个数, batch_size, rel_seq_len]
    :param B_label_id:
    :param I_label_id:
    :return:
    '''
    result = []
    batch_tag = []
    sentence_tag = []
    for batch in input:
        for out_id_list in batch:
            for num in range(len(out_id_list)):
                if out_id_list[num] == B_label_id:
                    start_pos = num
                if out_id_list[num] == I_label_id and out_id_list[num-1] == B_label_id:
                    length = 2
                    for num2 in range(num, len(out_id_list)):
                        if out_id_list[num2] == I_label_id and out_id_list[num2-1] == I_label_id:
                            length += 1
                            if out_id_list[num2] == 9:  # 到达末尾
                                sentence_tag.append((start_pos, length))
                                break
                        if out_id_list[num2] == 7:
                            sentence_tag.append((start_pos, length))
                            break
            batch_tag.append(sentence_tag)
            sentence_tag = []
        result.append(batch_tag)
        batch_tag = []

    return result

def find_all_tag(input):
    num = 1
    result = {}
    for tag in tags:
        res = find_tag(input, B_label_id=tag[0], I_label_id=tag[1])
        result[num] = res
        num += 1
    return result

def Precision(pre_output, true_output):
    '''
    计算准确率
    :param pre_output:  预测输出
    :param true_output:  真实输出
    :return: 准确率
    '''
    pre = []
    pre_result = find_all_tag(pre_output)
    for num in pre_result:
        for i, batch in enumerate(pre_result[num]):
            for j, seq_path_id in enumerate(batch):
                if len(seq_path_id) != 0:
                    for one_tuple in seq_path_id:
                        if one_tuple:
                            if pre_output[i][j][one_tuple[0]: one_tuple[0]+one_tuple[1]] == \
                                    true_output[i][j][one_tuple[0]: one_tuple[0]+one_tuple[1]]:
                                pre.append(1)
                            else:
                                pre.append(0)
    if len(pre) != 0:
        return sum(pre) / len(pre)
    else:
        return 0

def Recall(pre_output, true_output):
    '''
    计算召回率
    :param pre_output:
    :param true_output:
    :return:
    '''
    recall = []
    true_result = find_all_tag(true_output)
    for num in true_result:
        for i, batch in enumerate(true_result[num]):
            for j, seq_path_id in enumerate(batch):
                if len(seq_path_id) != 0:
                    for one_tuple in seq_path_id:
                        if one_tuple:
                            if pre_output[i][j][one_tuple[0]: one_tuple[0] + one_tuple[1]] == \
                                    true_output[i][j][one_tuple[0]: one_tuple[0] + one_tuple[1]]:
                                recall.append(1)
                            else:
                                recall.append(0)
    if len(recall) != 0:
        return sum(recall) / len(recall)
    else:
        return 0

def F1_score(precision, recall):
    '''
    计算F1值
    :param presion: 准确率
    :param recall:  召回率
    :return: F1值
    '''
    if (precision+recall) != 0:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0