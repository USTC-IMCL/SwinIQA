import sys
import os
import random
from msssim import MSSSIM
sys.path.append('..')
sys.path.append('.')
from nets.swin_multilevel2 import swin_FR_NR_modified_clic, swin_FR_NR_modified
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from datasets.datasets import Dataset_Test_Compare, Dataset_Test
import argparse
import os
from scipy import stats
import copy
from utils import save_to_json,load_from_json
import torch.optim.lr_scheduler as LS
import csv
from tqdm import tqdm

def test_compare(args):
    model = swin_FR_NR_modified_clic(mode='4').cuda()
    model_weight = model.state_dict()
    pretrained_weights = torch.load('../weights/test_weights.pt')['model']
    new_dict = {}
    if 'module' in list(pretrained_weights.keys())[0]:
        for k in pretrained_weights.keys():
            #     print(k)
            if k.replace('module.', '') in model_weight:
                new_dict[k.replace('module.', '')] = pretrained_weights[k]
        # print(new_dict.keys())
        model_weight.update(new_dict)
        model.load_state_dict(model_weight)
    else:
        model.load_state_dict(pretrained_weights)
    print('load pretrained weight', flush=True)
    if args.mgpu:
        model = torch.nn.DataParallel(model, device_ids=[id for id in range(len(args.gpu_id.split(',')))])
    Test_datasets = Dataset_Test_Compare(test_root='../data/', test_csv='../data/xx.csv',stride_val=args.stride_val,resize=args.resize,given_label=False)
    test_loader = DataLoader(Test_datasets, batch_size=1, shuffle=False, num_workers=1)
    #print(len(val_loader1),len(test_clic_loader))
    print('test len',len(test_loader))
    #---------------------------------------------------------------------------------------------------
    # val clic
    model.eval()
    match_num = 0
    total_num = 0
    name_list = []
    judge_list = []
    with torch.no_grad():
        for ite, (batch1,batch2) in enumerate(test_loader):
            if ite % 100 ==0:
                print('processing{}/{}'.format(ite,len(test_loader)),flush=True)
            o_img, a_img, b_img, label = batch1['o'].cuda(), batch1['a'].cuda(), batch1['b'].cuda(), batch1[
                'label'].type(torch.FloatTensor).cuda()
            # print(o_img.shape)
            name_list.append([batch2['o'][0],batch2['a'][0], batch2['b'][0]])
            prd_scorea = torch.zeros(o_img.shape[0]).cuda()
            prd_scoreb = torch.zeros(o_img.shape[0]).cuda()
            prd_judge = torch.zeros(o_img.shape[0]).cuda()

            for i in range(o_img.shape[0]):
                score_a, score_b, compare_judge = model(a_img[i], b_img[i], o_img[i])
                score_a = score_a.mean()
                score_b = score_b.mean()
                prd_scorea[i] = score_a
                prd_scoreb[i] = score_b
                prd_judge[i] = 0 if score_a <= score_b else 1
            judge_list.extend(prd_judge.cpu().detach().numpy())
            # print(prd_judge,label)
            match_num += torch.eq(prd_judge, label).sum().item()
            total_num += o_img.shape[0]

    save_txt_root = args.save_res_root_4stage
    if os.path.exists(save_txt_root):
        os.remove(save_txt_root)
    fl = open(save_txt_root, 'w')
    writer = csv.writer(fl)
    for names, judge in zip(name_list, judge_list):
        row = (names[0], names[1],names[2], int(judge))
        writer.writerow(row)
    fl.close()



def test(args):
    model = swin_FR_NR_modified(mode='4').cuda()
    model_weight = model.state_dict()
    pretrained_weights = torch.load('../weights/test_weights.pt')['model']
    new_dict = {}
    if 'module' in list(pretrained_weights.keys())[0]:
        for k in pretrained_weights.keys():
            #     print(k)
            if k.replace('module.', '') in model_weight:
                new_dict[k.replace('module.', '')] = pretrained_weights[k]
        # print(new_dict.keys())
        model_weight.update(new_dict)
        model.load_state_dict(model_weight)
    else:
        model.load_state_dict(pretrained_weights)
    print('load pretrained weight', flush=True)
    if args.mgpu:
        model = torch.nn.DataParallel(model, device_ids=[id for id in range(len(args.gpu_id.split(',')))])
    Test_datasets = Dataset_Test(test_root='../data/', test_csv='../data/xx.csv',stride_val=args.stride_val,resize=args.resize,given_label=False)
    test_loader = DataLoader(Test_datasets, batch_size=1, shuffle=False, num_workers=1)
    #print(len(val_loader1),len(test_clic_loader))
    print('test len',len(test_loader))
    #---------------------------------------------------------------------------------------------------
    # val clic
    model.eval()
    total_num = 0
    name_list = []
    score_list = []
    with torch.no_grad():
        for ite, (batch1,batch2) in enumerate(test_loader):
            if ite % 100 ==0:
                print('processing{}/{}'.format(ite,len(test_loader)),flush=True)
            o_img, a_img, label = batch1['o'].cuda(), batch1['a'].cuda(), batch1[
                'label'].type(torch.FloatTensor).cuda()
            # print(o_img.shape)
            name_list.append([batch2['o'][0],batch2['a'][0]])
            prd_scorea = torch.zeros(o_img.shape[0]).cuda()
            
            for i in range(o_img.shape[0]):
                score_a = model(a_img[i], o_img[i])
                score_a = score_a.mean()
                prd_scorea[i] = score_a
                print("==============================score", score_a)
            score_list.extend(prd_scorea.cpu().detach().numpy())
            print(score_list)
            total_num += o_img.shape[0]

    save_txt_root = args.save_res_root_4stage
    if os.path.exists(save_txt_root):
        os.remove(save_txt_root)
    fl = open(save_txt_root, 'w')
    writer = csv.writer(fl)
    for names, score in zip(name_list, score_list):
        row = (names[0], names[1], score)
        writer.writerow(row)
    fl.close()


def psnr(a, b):
  mse = torch.mean((a.flatten() - b.flatten()) ** 2)
  if mse < 1e-10:
      return 100
  return 10 * torch.log10(255**2 / mse)

def psnr_batch(a,b):
    batchsz = a.shape[0]


def psnr(a, b):
  mse = torch.mean((a.flatten() - b.flatten()) ** 2)
  if mse < 1e-10:
      return 100
  return 10 * torch.log10(255**2 / mse)

def psnr_batch(a,b):
    batchsz = a.shape[0]
    res = torch.zeros(batchsz)
    for i in range(batchsz):
        res[i]=psnr(a[i],b[i])
    return res

msssim_calculator = MSSSIM()
def msssim_batch(a,b):
    batchsz = a.shape[0]
    res = torch.zeros(batchsz)
    for i in range(batchsz):
        res[i] = msssim_calculator(a[i].unsqueeze(0), b[i].unsqueeze(0))
    return res




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='implementation of pieAPP')

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--save_res_root_3stage',type=str,default='../clic2022_res/clic2022test_3stage.csv')
    parser.add_argument('--save_res_root_4stage',type=str,default='../clic2022_res/clic2022test_4stage_modified_alldata_epoch1_0.7651_0.7549.csv')
    parser.add_argument('--save_res_root_DTF',type=str,default='../clic2022_res/clic2022test_DTF.csv')
    parser.add_argument('--mgpu', action='store_true', default=False)
    parser.add_argument('--mode',type=str,default='2')
    parser.add_argument('--train_batchsz',type=int,default=48)
    parser.add_argument('--save_model_root',type=str,default='./log/swinFRmode2_clic_alldata_xidianpretrain/models')
    parser.add_argument('--test_freq',type=int,default=1)
    parser.add_argument('--stride_val',type=int,default=224)
    parser.add_argument('--resize',type=int,default=224)
    parser.add_argument('--opts',type=str,default='3stage')
    parser.add_argument('--compare', type=bool, default=False)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(args)
    if args.compare == True:
        test_compare(args)
    else:
        test(args)
   
