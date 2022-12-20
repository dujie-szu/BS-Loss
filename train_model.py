import copy
import time
from unet import Unet
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader import Dataset
import imageio
import os
from torch import nn
import random
import numpy as np
from torch.nn import init
from BS_loss import BSLoss, BSL_LC
from evaluation_function import evaluate_func, surface_dist
from skimage import img_as_ubyte
from glob import glob
from tqdm import tqdm

manualSeed = 999
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def init_weights(model, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    model.apply(init_func)
def save_data(output, img_path, save_patch):
    output = (output > 0.5).float()
    for i in range(output.shape[0]):
        npName = os.path.basename(img_path[i])
        overNum = npName.find(".png")
        rgbName = npName[0:overNum]
        rgbName = rgbName + ".png"
        if output.shape[1] == 1:
            img_out = torch.squeeze(output[i, :, :, :]).cpu().numpy()
            imageio.imsave(save_patch + rgbName, img_as_ubyte(img_out))
        elif output.shape[1] == 2:    #one-hot
            img_out = torch.squeeze(output[i, 1, :, :]).cpu().numpy()
            imageio.imsave(save_patch + rgbName, img_as_ubyte(img_out))

# Train model
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')
st_time = time.time()
print('Start train model ....')

loss_list = []
train_acc_list = []
train_prec_list = []
train_recal_list = []
train_dice_list = []
val_acc_list = []
val_prec_list = []
val_recal_list = []
val_dice_list = []
val_boundary_dice_list = []
val_hd_list = []

max_dice = -1
best_net = None
max_dice_epoch = -1
val_freq = 2
epochs = 400
lr = 1e-3
bs = 10
model = Unet(3, 1).to(device)
model = nn.DataParallel(model).cuda()
init_weights(model, init_type='normal')
val_img_path = sorted(glob("/public/guankai/Data/LITS/test/mask/*.png"))   #get output name
save_path = '/public/guankai/Data result/LITS/UNet-BS out/'  #create your save floder
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
train_dataset = Dataset(image_root="/public/guankai/Data/LITS/train/image/",
                         gt_root="/public/guankai/Data/LITS/train/mask/", trainsize=512)
test_dataset = Dataset(image_root="/public/guankai/Data/LITS/test/image/",
                        gt_root="/public/guankai/Data/LITS/test/mask/", trainsize=512)
train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=1)
val_dataloader = DataLoader(test_dataset, batch_size=bs)

for epoch in range(epochs):
    bt_st_time = time.time()
    avg_loss, avg_alpha, avg_loss2 = 0., 0., 0.
    for img, mask, _ in train_dataloader:
        optimizer.zero_grad()

        inputs = img.to(device)
        labels = mask.to(device)
        pred_img = model(inputs)

        criterion = BSLoss()
        loss = criterion(pred_img, labels, alpha=0.8)
        # criterion = BSL_LC()
        # loss = criterion(pred_img, labels, alpha=0.8, beta=0.9)

        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
    avg_loss = avg_loss / len(train_dataloader)
    loss_list.append(avg_loss)
    print('Epoch {}:  loss: {:4.4f}  time: {:4.4f}s'.format(epoch, avg_loss, (time.time() - bt_st_time)))

    if (epoch + 1) % val_freq == 0:
        model.eval()
        eval_st_time = time.time()
        with torch.no_grad():
            train_avg_acc, train_avg_prec, train_avg_recal, train_avg_dice = 0., 0., 0., 0.
            for img, mask, _, _ in train_dataloader:
                inputs = img.to(device)
                labels = mask.to(device)
                pred_img = model(inputs)
                pred_img = pred_img.squeeze(1)

                dice_, prec_, recal_ = evaluate_func(pred_img, labels)

                train_avg_prec += prec_.item()
                train_avg_recal += recal_.item()
                train_avg_dice += dice_.item()

            train_avg_prec /= len(train_dataloader)
            train_avg_recal /= len(train_dataloader)
            train_avg_dice /= len(train_dataloader)

            train_prec_list.append(train_avg_prec)
            train_recal_list.append(train_avg_recal)
            train_dice_list.append(train_avg_dice)

            val_avg_asd, val_avg_hd_95 = 0., 0.
            val_avg_prec, val_avg_recal, val_avg_dice = 0., 0., 0.
            for img, mask, id in val_dataloader:
                input = img.to(device)
                label = mask.to(device)
                out = model(input)

                dice_, prec_, recal_ = evaluate_func(out.cpu(), mask.cpu())
                asd, hd_95 = surface_dist(out.cpu(), mask.cpu())

                val_avg_prec += prec_.item()
                val_avg_recal += recal_.item()
                val_avg_dice += dice_.item()
                val_avg_asd += asd.item()
                val_avg_hd_95 += hd_95.item()

            val_avg_prec /= len(val_dataloader)
            val_avg_recal /= len(val_dataloader)
            val_avg_dice /= len(val_dataloader)
            val_avg_asd /= len(val_dataloader)
            val_avg_hd_95 /= len(val_dataloader)

            if val_avg_dice > max_dice:
                max_dice = val_avg_dice
                best_net = copy.deepcopy(model)
                max_dice_epoch = epoch
                for i, (img, _) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                    input = img.to(device)
                    out = model(input)
                    img_paths = val_img_path[bs * i:bs * (i + 1)]
                    save_data(output=out, img_path=img_paths, save_patch=save_path)

            print(
                'Evaluation: dice, prec, recall \n train: {:4.4f}| {:4.4f}| {:4.4f}|'.format(
                    train_avg_dice, train_avg_prec, train_avg_recal))
            print(
                'Evaluation: dice, 95%hd, asd prec, recall, \n test:  {:4.4f}| {:4.4f}| {:4.4f}| {:4.4f}| {:4.4f}'.format(
                    val_avg_dice, val_avg_hd_95, val_avg_asd, val_avg_prec, val_avg_recal))
        model.train()
print('Train model time: {:5.4f}'.format(time.time() - st_time))
print('Max dice epoch {}: {:4.4f}'.format(max_dice_epoch, max_dice))

x = list(range(len(loss_list)))
plt.figure(figsize=(15, 10))
plt.plot(x, loss_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
