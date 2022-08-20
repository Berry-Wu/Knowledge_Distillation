# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/8/19 12:05 
# @Author : wzy 
# @File : visual.py
# ---------------------------------------
from matplotlib import pyplot as plt


def draw(teacher_history, student_history, kd_history, epochs):
    # 三个模型的loss和acc分析
    x = list(range(1, epochs + 1))

    plt.subplot(2, 1, 1)
    plt.plot(x, [teacher_history[i][1] for i in range(epochs)], label='teacher')
    plt.plot(x, [kd_history[i][1] for i in range(epochs)], label='student with KD')
    plt.plot(x, [student_history[i][1] for i in range(epochs)], label='student without KD')

    plt.title('Test accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, [teacher_history[i][0] for i in range(epochs)], label='teacher')
    plt.plot(x, [kd_history[i][0] for i in range(epochs)], label='student with KD')
    plt.plot(x, [student_history[i][0] for i in range(epochs)], label='student without KD')

    plt.title('Test loss')
    plt.legend()

    plt.savefig("visual.png")