# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/19 12:12 
# @Author : wzy 
# @File : main.py
# ---------------------------------------
import numpy as np
import torch

import train_s
import train_t
import distillation
from visual import draw
from arg_parse import parse_args

args = parse_args()

if __name__ == '__main__':
    teacher_model, teacher_history = train_t.main_t()
    student_model, student_history = train_s.main_s()
    kd_model, kd_history = distillation.main_kd(teacher_model)

    teacher_history = np.array(torch.tensor(teacher_history, device='cpu'))
    student_history = np.array(torch.tensor(student_history, device='cpu'))
    kd_history = np.array(torch.tensor(kd_history, device='cpu'))

    draw(teacher_history, student_history, kd_history, args.epoch)
