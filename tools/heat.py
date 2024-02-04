# coding: utf-8
import os
from PIL import Image
from torchvision import models
from torchvision import transforms
from models.model import BaseNet
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
import sys
import matplotlib.pyplot as plt


def resnet_cifar(net, x1, x2):

    x1_1, x1_2, x1_3, x1_4, x1_5 = net.backbone.base_forward(x1)
    x2_1, x2_2, x2_3, x2_4, x2_5 = net.backbone.base_forward(x2)

    # feature difference
    d5 = net.TFIM5(x1_5, x2_5)  # 1/32
    d4 = net.TFIM4(x1_4, x2_4)  # 1/16
    d3 = net.TFIM3(x1_3, x2_3)  # 1/8
    d2 = net.TFIM2(x1_2, x2_2)  # 1/4

    # change information guided refinement 1
    d5_p, d4_p, d3_p, d2_p = net.CIEM1(d5, d4, d3, d2)
    d5, d4, d3, d2 = net.GRM1(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

    # # change information guided refinement 2
    # d5_p, d4_p, d3_p, d2_p = net.CIEM2(d5, d4, d3, d2)
    # d5, d4, d3, d2 = net.GRM2(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

    # # change information guided refinement 3
    # d5_p, d4_p, d3_p, d2_p = net.CIEM3(d5, d4, d3, d2)
    # d5, d4, d3, d2 = net.GRM3(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

    # # change information guided refinement 4
    # d5_p, d4_p, d3_p, d2_p = net.CIEM4(d5, d4, d3, d2)
    # d5, d4, d3, d2 = net.GRM4(d5, d4, d3, d2, d5_p, d4_p, d3_p, d2_p)

    # decoder
    out, mask = net.decoder(d5, d4, d3, d2, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)
    output = F.interpolate(out, x1.size()[2:], mode='bilinear', align_corners=True)
    output = torch.sigmoid(output)

    return out, mask, output

def draw_CAM(model, img1, img2):

    out, out_sig, output = resnet_cifar(model, img1, img2)
    # 为了能读取到中间梯度定义的辅助函数
    def extract(g):
        global features_grad
        features_grad = g

    out.register_hook(extract)
    out_sig.register_hook(extract)

    output.backward(torch.ones_like(output), retain_graph=True)  # 计算梯度

    grads = features_grad  # 获取梯度

    pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))

    # 此处batch size默认为1，所以去掉了第0维（batch size维）
    pooled_grads = pooled_grads[0]
    out = out[0]
    out_sig = out_sig[0]

    # 512是最后一层feature的通道数
    for i in range(64):
        out[i, ...] *= pooled_grads[i, ...]
    for i in range(1):
        out_sig[i, ...] *= pooled_grads[i, ...]

    heatmap_out = out.detach().cpu().numpy()
    heatmap_out_sig = out_sig.detach().cpu().numpy()

    heatmap_out = np.mean(heatmap_out, axis=0)
    heatmap_out_sig = np.mean(heatmap_out_sig, axis=0)

    heatmap_out = np.maximum(heatmap_out, 0)
    heatmap_out_sig = np.maximum(heatmap_out_sig, 0)

    heatmap_out /= np.max(heatmap_out)
    heatmap_out_sig /= np.max(heatmap_out_sig)

    heatmap_out = cv2.resize(heatmap_out, (256, 256))  # 将热力图的大小调整为与原始图像相同
    heatmap_out_sig = cv2.resize(heatmap_out_sig, (256, 256))  # 将热力图的大小调整为与原始图像相同

    heatmap_out = np.uint8(255 * heatmap_out)  # 将热力图转换为RGB格式
    heatmap_out_sig = np.uint8(255 * heatmap_out_sig)  # 将热力图转换为RGB格式

    heatmap_out = cv2.applyColorMap(heatmap_out, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    heatmap_out_sig = cv2.applyColorMap(heatmap_out_sig, cv2.COLORMAP_JET)  # 将热力图应用于原始图像

    # superimposed_img = heatmap * 0.4 + img  # 这里的0.4是热力图强度因子
    # cv2.imwrite(save_path, superimposed_img)  # 将图像保存到硬盘

    # cv2.imwrite(save_path, heatmap_5)
    # cv2.imwrite(save_path, heatmap_4)
    # cv2.imwrite(save_path, heatmap_3)
    # cv2.imwrite(save_path, heatmap_2)
    # cv2.imwrite(save_path, heatmap_add)
    # cv2.imwrite(save_path, heatmap_down)

    return heatmap_out, heatmap_out_sig


# model_file_name = '/home/zijun/PycharmProjects/TFI-GR-main(复件)(复件)/tools/log/best_model.pth'
# model = BaseNet(3, 1).cuda()
# state_dict = torch.load(model_file_name)
# model.load_state_dict(state_dict)
# img_path1 = '/home/zijun/datasets/LIVER-CD/256/test/A/test_1_12.png'
# img_path2 = '/home/zijun/datasets/LIVER-CD/256/test/B/test_1_12.png'
# label_path = '/home/zijun/datasets/LIVER-CD/256/test/label/test_1_12.png'
# save_path = '/home/zijun/datasets/mask.1.png'
# # transforms.Normalize(mean=mean, std=std)
# mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
# std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
# transform = transforms.Compose([transforms.ToTensor()])
# draw_CAM(model, img_path1, img_path2, label_path, save_path, transform=transform)