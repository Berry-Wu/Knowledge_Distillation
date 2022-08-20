# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/18 20:38 
# @Author : wzy 
# @File : train_t.py
# ---------------------------------------
import math

import torch
import torch.nn as nn
from models import Teacher, Student
from arg_parse import parse_args
import datas

args = parse_args()


def train_teacher(model, device, train_loader, optimizer, loss_func, epoch, epochs):
    # 启用 BatchNormalization 和 Dropout
    model.train()
    trained_samples = 0  # 用于记录已经训练的样本数
    for batch_idx, (data, target) in enumerate(train_loader):
        # 搬到指定gpu或者cpu设备上运算
        data, target = data.to(device), target.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 计算误差
        loss = loss_func(output, target)
        # 误差反向传播
        loss.backward()
        # 梯度更新一步
        optimizer.step()

        # 统计已经训练的数据量
        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)

        print('\r【Teacher】 Train epoch: [{}/{}] {}/{} [{}]{}%'.format(epoch, epochs, trained_samples,
                                                                      len(train_loader.dataset),
                                                                      '-' * progress + '>', progress * 2), end='')


def test_teacher(model, device, val_loader, loss_func):
    # 不启用 BatchNormalization 和 Dropout
    model.eval()
    test_loss = 0
    num_correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target)
            # 输出预测类别
            _, predictions = output.max(1)
            num_correct += (predictions == target).sum()
    test_loss /= len(val_loader.dataset)

    print('\n【Teacher】 Test: average loss: {:.4f}, accuracy:{}/{},({:.4f}%)'.format(
        test_loss.item(), num_correct, len(val_loader.dataset), 100 * num_correct / len(val_loader.dataset)))

    return test_loss, num_correct / len(val_loader.dataset)


def main_t():
    torch.manual_seed(1)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_model = Teacher().to(device)
    teacher_history = []  # 记录loss和acc
    optimizer = torch.optim.Adam(t_model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1, args.epoch + 1):
        train_teacher(t_model, device, datas.train_loader, optimizer, loss_func, epoch, args.epoch)
        loss, acc = test_teacher(t_model, device, datas.val_loader, loss_func)
        teacher_history.append((loss, acc))

    # 保存模型
    torch.save(t_model.state_dict(), './pts/teacher.pt')
    return t_model, teacher_history


if __name__ == '__main__':
    main_t()
