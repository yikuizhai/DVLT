import sys

import cv2

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
from PIL import Image

import os, time
import numpy as np
from argparse import ArgumentParser
from models.model import BaseNet
from heat import draw_CAM


def ValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = BaseNet(3, 1)

    if args.file_root == 'LEVIR':
        args.file_root = '/home/zijun/datasets/LIVER-CD/256'
    elif args.file_root == 'BCDD':
        args.file_root = '/home/zijun/datasets/BCDD'
    elif args.file_root == 'SYSU':
        args.file_root = '/home/zijun/datasets/SYSY-CD'
    elif args.file_root == 'CDD':
        args.file_root = '/home/guan/Documents/Datasets/ChangeDetection/CDD'
    elif args.file_root == 'testLEVIR':
        args.file_root = '../samples'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    args.vis_mask = args.vis_output + 'mask/'
    args.vis_out_cam = args.vis_output + 'out/'
    args.vis_output_cam = args.vis_output + 'output/'

    if not os.path.exists(args.vis_mask):
        os.makedirs(args.vis_mask)
    if not os.path.exists(args.vis_out_cam):
        os.makedirs(args.vis_out_cam)
    if not os.path.exists(args.vis_output_cam):
        os.makedirs(args.vis_output_cam)

    if args.onGPU:
        model = model.cuda()

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])
    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    # load the model
    model_file_name = args.weight
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)
    model.eval()

    # with torch.no_grad():
    for iter, batched_inputs in enumerate(testLoader):
        img, target = batched_inputs
        img_name = testLoader.sampler.data_source.file_list[iter]
        pre_img = img[:, 0:3]
        post_img = img[:, 3:6]

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            target = target.cuda()
            post_img = post_img.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the mdoel
        output = model(pre_img_var, post_img_var)
        cam_out, cam_output = draw_CAM(model, pre_img_var, post_img_var)
        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        # save change maps
        pr = pred[0, 0].cpu().numpy()
        gt = target_var[0, 0].cpu().numpy()
        index_tp = np.where(np.logical_and(pr == 1, gt == 1))
        index_fp = np.where(np.logical_and(pr == 1, gt == 0))
        index_tn = np.where(np.logical_and(pr == 0, gt == 0))
        index_fn = np.where(np.logical_and(pr == 0, gt == 1))

        map = np.zeros([gt.shape[0], gt.shape[1], 3])
        map[index_tp] = [255, 255, 255]
        map[index_fp] = [255, 0, 0]
        map[index_tn] = [0, 0, 0]
        map[index_fn] = [0, 0, 255]
        change_map = Image.fromarray(np.array(map, dtype=np.uint8))

        change_map.save(args.vis_mask + img_name)

        cam_out_name = args.vis_out_cam + img_name
        cam_output_name = args.vis_output_cam + img_name

        cv2.imwrite(cam_out_name, cam_out)
        cv2.imwrite(cam_output_name, cam_output)

        print('{}/{}, {}'.format(iter+1, len(testLoader), img_name))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="SYSU", help='Data directory | LEVIR | BCDD | SYSU ')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--vis_output', default='/home/zijun/PycharmProjects/TFI-GR-main_heat/result_SYSU/测DGAM/120/', help='Directory to save the results')
    # parser.add_argument('--vis_output', default='/home/zijun/PycharmProjects/TFI-GR-main_heat/result/测MLDA/无/0/mask/', help='Directory to save the results')
    # parser.add_argument('--vis_out_cam', default='/home/zijun/PycharmProjects/TFI-GR-main_heat/result/测MLDA/无/0/out/', help='Directory to save the results')
    # parser.add_argument('--vis_output_cam', default='/home/zijun/PycharmProjects/TFI-GR-main_heat/result/测MLDA/无/0/output/', help='Directory to save the results')

    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='/home/zijun/PycharmProjects/TFI-GR-main-消融/tools/log/SYSU/DGAM/120/0.8297.pth', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    ValidateSegmentation(args)
