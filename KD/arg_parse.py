# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/18 20:09 
# @Author : wzy 
# @File : arg_parse.py
# ---------------------------------------
import argparse


def parse_args():
    parse = argparse.ArgumentParser(description="The hyper-parameter of KD")
    parse.add_argument('-b', '--bs', default=128)
    parse.add_argument('-l', '--lr', default=1e-4)
    parse.add_argument('-e', '--epoch', default=3)
    args = parse.parse_args()
    return args
