#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/4 12:40
# @Author  : JJkinging
# @File    : config.py
class Config(object):
    '''配置类'''

    def __init__(self):
        self.label_file = '../dataset/tag.txt'
        self.train_file = '../dataset/train.txt'
        self.dev_file = '../dataset/dev.txt'
        self.test_file = '../dataset/test.txt'
        self.vocab = '../dataset/bert/vocab.txt'
        self.max_length = 128
        self.use_cuda = True
        self.gpu = 0
        self.batch_size = 8
        self.bert_path = './data/bert'
        self.rnn_hidden = 128
        self.bert_embedding = 768
        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.rnn_layer = 1
        self.lr = 0.0001
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = '../result/checkpoints/NER_12.pth.tar'
        self.epochs = 64
        self.max_grad_norm = 10
        self.target_dir = '../result/checkpoints'
        self.patience = 5

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)