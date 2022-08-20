# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/18 22:33 
# @Author : wzy 
# @File : distillation.py
# ---------------------------------------
import math
import torch
from torch import nn
from arg_parse import parse_args
from models import Student
import torch.nn.functional as F
import datas

args = parse_args()


# 蒸馏部分：定义kd的loss
def distillation(student_out, target, teacher_out, T, alpha):
    """
    :param student_out: 学生预测的概率分布
    :param target: 实际标签
    :param teacher_out: 老师预测的概率分布
    :param T: 温度系数
    :param alpha: 损失调整因子
    :return:
    """
    hard_loss = F.cross_entropy(student_out, target) * alpha
    soft_loss = nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student_out / T, dim=1),
                                                    F.softmax(teacher_out / T, dim=1)) * T * T * (1 - alpha)
    return soft_loss + hard_loss


def train_student_kd(s_model, t_model, device, train_loader, optimizer, epoch, epochs):
    s_model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # 搬到指定gpu或者cpu设备上运算
        data, target = data.to(device), target.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 学生网络的输出
        student_output = s_model(data)
        # 教师网络的输出
        with torch.no_grad():
            teacher_output = t_model(data)
        # 计算蒸馏误差
        loss = distillation(student_output, target, teacher_output, T=7, alpha=0.3)

        loss.backward()
        optimizer.step()

        # 统计已经训练的数据量
        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)

        print('\r【Distillation】 Train epoch: [{}/{}] {}/{} [{}]{}%'.format(epoch, epochs, trained_samples,
                                                                           len(train_loader.dataset),
                                                                           '-' * progress + '>', progress * 2), end='')


def test_student_kd(model, device, val_loader, loss_func):
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

    print('\n【Distillation】 Test: average loss: {:.4f}, accuracy:{}/{},({:.4f}%)'.format(
        test_loss.item(), num_correct, len(val_loader.dataset), 100 * num_correct / len(val_loader.dataset)))

    return test_loss, num_correct / len(val_loader.dataset)


def main_kd(t_model):
    torch.manual_seed(1)  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kd_model = Student().to(device)
    kd_history = []
    optimizer = torch.optim.Adam(kd_model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(1, args.epoch + 1):
        train_student_kd(kd_model, t_model, device, datas.train_loader, optimizer, epoch, args.epoch)
        loss, acc = test_student_kd(kd_model, device, datas.val_loader, loss_func)

        kd_history.append((loss, acc))
    torch.save(kd_model.state_dict(), './pts/kd.pt')
    return kd_model, kd_history


if __name__ == '__main__':
    main_kd()
