# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/18 20:24 
# @Author : wzy 
# @File : models.py
# ---------------------------------------
import torch.nn as nn


class Teacher(nn.Module):
    def __init__(self, num_cls=10):
        super(Teacher, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_cls)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        output = self.fc3(x)
        return output


class Student(nn.Module):
    def __init__(self, num_cls=10):
        super(Student, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_cls)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        output = self.fc3(x)
        return output


