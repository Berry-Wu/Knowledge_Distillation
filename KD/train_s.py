# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/18 21:22 
# @Author : wzy 
# @File : train_s.py
# ---------------------------------------
import math

import torch
import torch.nn as nn
from models import Teacher, Student
from arg_parse import parse_args
import datas

args = parse_args()


def train_student(model, device, train_loader, optimizer, loss_func, epoch, epochs):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)

        print('\r【Student】 Train epoch: [{}/{}] {}/{} [{}]{}%'.format(epoch, epochs, trained_samples,
                                                                      len(train_loader.dataset),
                                                                      '-' * progress + '>', progress * 2), end='')


def test_student(model, device, val_loader, loss_func):
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

    print('\n【Student】 Test: average loss: {:.4f}, accuracy: {}/{} ({:.4f}%)'.format(
        test_loss.item(), num_correct, len(val_loader.dataset), 100. * num_correct / len(val_loader.dataset)))

    return test_loss, num_correct / len(val_loader.dataset)


def main_s():
    torch.manual_seed(1)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s_model = Student().to(device)
    student_history = []  # 记录loss和acc
    optimizer = torch.optim.Adam(s_model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(1, args.epoch + 1):
        train_student(s_model, device, datas.train_loader, optimizer, loss_func, epoch, args.epoch)
        loss, acc = test_student(s_model, device, datas.val_loader, loss_func)
        student_history.append((loss, acc))

    # 保存模型,state_dict:Returns a dictionary containing a whole state of the module.
    torch.save(s_model.state_dict(), './pts/student.pt')
    return s_model, student_history


if __name__ == '__main__':
    main_s()
