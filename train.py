import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import torchvision
import random
from PIL import Image
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from MODELS.resnet_256d import resnet50
from MODELS.resnet import resnet50 as resnet50pretrain
from MODELS.resnet_gfim import resnet50 as resnet50_gfim
from MODELS.resnet_dam import resnet50 as resnet50_dam
from MODELS.resnet_gfim_dam import resnet50 as resnet50_gfim_dam

import argparse
import numpy as np
import copy
from toolkit.dataread import read_data

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def adjust_learning_rate(optimizer, epoch, reducetime, reduced):
    if epoch==reducetime and reduced==0:
        lr = args.lr * 0.1
    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def pretrain(model):
    modelpretrain = resnet50pretrain(num_classes=1000)
    modelpretrain.load_state_dict(torch.load(args.pretrained_model))
    model.conv1 = modelpretrain.conv1
    model.bn1 = modelpretrain.bn1
    model.relu = modelpretrain.relu
    model.maxpool = modelpretrain.maxpool
    model.layer1 = modelpretrain.layer1
    model.layer2 = modelpretrain.layer2
    model.layer3 = modelpretrain.layer3
    model.layer4 = modelpretrain.layer4
    model.avgpool = modelpretrain.avgpool

    return model


def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.model=='resnet50_gfim' or args.model=='resnet50_gfim_dam':
        detr_checkpoint = torch.load('./model/pretrain/detr-r50-e632da11.pth', map_location='cpu')
        names={'transformer.encoder.layers.0.self_attn.in_proj_weight', 'transformer.encoder.layers.0.self_attn.in_proj_bias', 'transformer.encoder.layers.0.self_attn.out_proj.weight', 'transformer.encoder.layers.0.self_attn.out_proj.bias', 'transformer.encoder.layers.0.linear1.weight', 'transformer.encoder.layers.0.linear1.bias', 'transformer.encoder.layers.0.linear2.weight', 'transformer.encoder.layers.0.linear2.bias', 'transformer.encoder.layers.0.norm1.weight', 'transformer.encoder.layers.0.norm1.bias', 'transformer.encoder.layers.0.norm2.weight', 'transformer.encoder.layers.0.norm2.bias', 'transformer.encoder.layers.1.self_attn.in_proj_weight', 'transformer.encoder.layers.1.self_attn.in_proj_bias', 'transformer.encoder.layers.1.self_attn.out_proj.weight', 'transformer.encoder.layers.1.self_attn.out_proj.bias', 'transformer.encoder.layers.1.linear1.weight', 'transformer.encoder.layers.1.linear1.bias', 'transformer.encoder.layers.1.linear2.weight', 'transformer.encoder.layers.1.linear2.bias', 'transformer.encoder.layers.1.norm1.weight', 'transformer.encoder.layers.1.norm1.bias', 'transformer.encoder.layers.1.norm2.weight', 'transformer.encoder.layers.1.norm2.bias', 'transformer.encoder.layers.2.self_attn.in_proj_weight', 'transformer.encoder.layers.2.self_attn.in_proj_bias', 'transformer.encoder.layers.2.self_attn.out_proj.weight', 'transformer.encoder.layers.2.self_attn.out_proj.bias', 'transformer.encoder.layers.2.linear1.weight', 'transformer.encoder.layers.2.linear1.bias', 'transformer.encoder.layers.2.linear2.weight', 'transformer.encoder.layers.2.linear2.bias', 'transformer.encoder.layers.2.norm1.weight', 'transformer.encoder.layers.2.norm1.bias', 'transformer.encoder.layers.2.norm2.weight', 'transformer.encoder.layers.2.norm2.bias'}
        transformer_checkpoint = {key:detr_checkpoint['model'][key] for key in detr_checkpoint['model'].keys() & names}
        if args.model=='resnet50_gfim':
            model = resnet50_gfim(args,num_classes=args.num_class)
        else:
            model = resnet50_gfim_dam(args,num_classes=args.num_class)
        if args.pretrained:
            model=pretrain(model)
        model_state_dict = model.state_dict()
        model_state_dict.update(transformer_checkpoint)
        model.load_state_dict(model_state_dict)
    elif args.model== 'resnet50' or args.model=='resnet50_dam':
        if args.model== 'resnet50' : 
            model = resnet50(num_classes=args.num_class)
        else:
            model = resnet50_dam(num_classes=args.num_class)
        if args.pretrained:
            model=pretrain(model)
    

    # cost
    model = model.to(device)
    cost = nn.CrossEntropyLoss().to(device)
    nllcost = nn.NLLLoss().to(device)
    BCEcost = nn.BCELoss(reduction='mean').to(device)
    logsoftmax_func=nn.LogSoftmax(dim=1)
    softmax_func=nn.Softmax(dim=1)
    bestacc=0

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    mytransforms = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])
    rootdir=args.rootdir
    rootdirval=args.rootdirval
    valcasenames,list1,dirlist,normalboy,normalgirl,reduced,label2list=read_data(rootdir=rootdir,rootdirval=rootdirval)

    print('start training')
    for epoch in range(1, args.iteration + 1):
        model.train()
        start = time.time()
        batchnames=[]
        if epoch==args.lr_reduce_time:
            adjust_learning_rate(optimizer, epoch, args.lr_reduce_time, reduced)
            reduced=1
        sex=random.randint(0,100)%2
        variationed=0
        variation=random.randint(0,100)
        if sex==0:
            namelist=copy.copy(normalboy)
        else:
            namelist=copy.copy(normalgirl)
        if variationed==0 and variation%3==0:# +8
            namelist.append(7)
            variationed=1
        if variationed==0 and variation%10==0:# -7
            del namelist[12]
            variationed=1
        if variationed==0 and variation%5==0 and sex==0:# -y
            del namelist[-1]
            variationed=1
        if variationed==0 and variation%10==0:# +11
            namelist.append(10)
            variationed=1
        if variationed==0 and variation%10==0:# +12
            namelist.append(11)
            variationed=1
        if variationed==0 and variation%10==0:# +21
            namelist.append(20)
            variationed=1
        
        for i in namelist:
            batchnames.append(os.path.join(rootdir,dirlist[i],random.choice(list1[i]))) ##load the data to form a batch
        
        random.shuffle(batchnames)
        img_tensors = torch.empty(len(namelist), 3, 224, 224)###
        labellist=[]
        labelgrouplist=[]
        i=0
        for img_path in batchnames:
            data = Image.open(img_path)
            data = mytransforms(data)
            img_tensors[i,:,:,:] = data
            filelabel=int(img_path.split('/')[-1].split('_')[1])-1
            labellist.append(float(filelabel))
            labelgrouplist.append(float(label2list[filelabel]))
            i=i+1
        label_tensors = torch.from_numpy(np.array(labellist)).long()
        labelgroup_tensors = torch.from_numpy(np.array(labelgrouplist)).long()
        images = img_tensors.to(device)
        labels = label_tensors.to(device)
        labelgroups = labelgroup_tensors.to(device)
        if args.model=='resnet50' or args.model=='resnet50_dam':
            outputs = model(images)
            loss = cost(outputs, labels)
        
        elif args.model=='resnet50_gfim' or args.model=='resnet50_gfim_dam':
            outputs,outputsgroups = model(images)
            lossadd = cost(outputsgroups, labelgroups)
            lossmain = cost(outputs, labels)
            loss = lossmain+0.2*lossadd

        if epoch % 20 == 0:
            print (epoch,loss.data,optimizer.state_dict()['param_groups'][0]['lr'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0:
            torch.save(model, os.path.join(args.model_path, '%s-%d.pth' % (args.model_name, epoch)))
            model.eval()
            num=0
            correctnum=0
            for case in valcasenames:
                root1=os.path.join(rootdirval,case)
                img_list1=os.listdir(root1)
                allprob=[]
                labels=[]
                img_tensors = torch.empty(46, 3, 224, 224)
                casei=0
                for img1 in img_list1:
                    data = Image.open(os.path.join(root1, img1))
                    data = mytransforms(data)
                    img_tensors[casei,:,:,:] = data
                    label=int(img1.split('_')[1])
                    labels.append(label)
                    casei=casei+1
                    num=num+1
                images = img_tensors.to(device)
                if 'gfim' in args.model:
                    outputs, _ = model(images)
                else:
                    outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                for pred in range(predicted.shape[0]):
                    if predicted[pred]+1==labels[pred]:
                        correctnum=correctnum+1
            acc=correctnum / num
            print("Acc: %.4f" % acc)
            model.train()
            if acc>=bestacc:
                torch.save(model, os.path.join(args.model_path, '%s-best.pth' % args.model_name))
                bestacc=acc


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='train hyper-parameter')
    parser.add_argument("--num_class", default=24, type=int)
    parser.add_argument("--iteration", default=40000, type=int)
    parser.add_argument("--lr_reduce_time", default=20000, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=46, type=int)
    parser.add_argument("--model_name", default='model-resnet50_gfim_dam', type=str)
    parser.add_argument("--model_path", default='./model', type=str)
    parser.add_argument("--pretrained", default=True, type=bool)
    parser.add_argument("--pretrained_model", default='./model/pretrain/resnet50-19c8e357.pth', type=str)
    parser.add_argument("--rootdir", default='dataset/train/', type=str)
    parser.add_argument("--rootdirval", default='dataset/val/', type=str)
    parser.add_argument("--model", default='resnet50_gfim_dam', type=str)
    parser.add_argument('--enc_layers', default=3, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nheads', default=1, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    args = parser.parse_args()
    main(args)
