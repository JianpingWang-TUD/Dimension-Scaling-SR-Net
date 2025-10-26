from __future__ import print_function
import os
import time
import socket
import pandas as pd
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from net_model.DSSR import DSSR_net
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from data_gen.data_generation import gen_signal
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
DSSR-net训练代码 
DSSR-net training code
"""

# ---------------------------
# Training settings
# ---------------------------
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate. default=0.0001, model:0.002,nomodel:0.0001')
parser.add_argument('--data_augmentation', type=bool, default=True, help='if adopt augmentation when training')
parser.add_argument('--Ispretrained', type=bool, default=False, help='If load checkpoint model')
parser.add_argument('--pretrained_sr', default='net_epoch_upsample_8_layers_3_max_n_R_5_gs_size_0.003_25_final.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', default='./checkpoint/model', help='Location to load checkpoint models')
parser.add_argument("--noiseL", type=float, default=25, help='noise level')
parser.add_argument('--save_folder', default='./checkpoint/', help='Location to save checkpoint models')
parser.add_argument('--statistics', default='./statistics/', help='Location to save statistics')

# Global settings
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--gpus', default=1, type=int, help='number of gpus')
parser.add_argument('--data_dir', type=str, default='./Dataset', help='the dataset dir')
parser.add_argument('--model_type', type=str, default='DSSR-Net', help='the name of model')
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
parser.add_argument('--Isreal', default=False, help='If training/testing on RGB images')
parser.add_argument('--numpy_seed', type=int, default=100)
parser.add_argument('--torch_seed', type=int, default=76)

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)

def findpeak(prediction,distance,d_grid,p_gt,dr):
    """
    Find peaks in the prediction result and evaluate estimation accuracy.
    prediction: model output
    distance: true distance
    d_grid: grid resolution
    p_gt: ground-truth positions
    dr: distance resolution
    """
    k = 0
    for i in range(len(prediction)):
        find_peaks_out,_ = scipy.signal.find_peaks(prediction[i],height=0.1)
        if len(find_peaks_out)==2:
            estimation_R = (find_peaks_out[1]-find_peaks_out[0])*d_grid
            if (estimation_R>= distance- dr/4) & (estimation_R <= distance+ dr/4) & (abs(find_peaks_out[0] - p_gt[i])<=int(dr/4/d_grid)):
                k += 1
    return k/len(prediction)


def train(epoch):
    """
    Training loop for one epoch
    """
    epoch_loss = 0
    model.train()
    for iteration, (clean_signal, target_fr,signal_fft,n_target,fr_ground) in enumerate(training_data_loader):
        
        input =  Variable(clean_signal).cuda()
        target_fr = target_fr.cuda()

        model.zero_grad()
        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)
        # Loss function (MSE between prediction and ground-truth frequency response)
        loss = criterion(prediction, target_fr)

        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.6f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0)))

    scheduler.step(loss)
    
    # Visualization of prediction vs ground truth
    plt.figure()
    plt.plot(target_fr[0,:].cpu().detach().numpy(),'o--', label='truth')
    plt.plot(prediction[0,:].cpu().detach().numpy(),'g', label='estimation')
    plt.legend(fontsize=14,loc='upper left')
    plt.savefig('target_fr.png')
    plt.savefig('results/gussion.pdf', format='pdf')
    print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / len(training_data_loader)))
    return loss.data

def batch_PSNR(img, imclean, data_range):
    """
    Compute average PSNR for a batch
    """
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :], Img[i, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def test(testing_data_loader):
    """
    Test model performance on test dataset
    """
    psnr_test= 0
    model.eval()
    
    for iteration, (clean_signal, target_fr,signal_fft,n_target,fr_ground) in enumerate(testing_data_loader):

        input = clean_signal.cuda()
        target = target_fr.cuda()
        with torch.no_grad():
            prediction = model(input)
            prediction = torch.clamp(prediction, 0., 1.)
        psnr_test += batch_PSNR(prediction, target, 1.)
    print("===> Avg. PSNR: {:.4f} dB".format(psnr_test / len(testing_data_loader)))
    return psnr_test / len(testing_data_loader)


def print_network(net):
    """
    Print network architecture and total number of parameters
    """
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
def adjust_learning_rate(opt, epo, lr):
    """
    Adjust learning rate by decaying every 20 epochs
    """
    lr = lr * (0.5 ** (epo // 20))
    for param_group in opt.param_groups:
        param_group['lr'] = lr

def checkpoint(epoch,psnr):
    """
    Save model checkpoint
    """
    model_out_path = opt.save_folder+hostname+opt.model_type+"_psnr_{}".format(psnr)+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':

    print('===> Loading datasets')

    # ---------------------------
    # Data setting
    # ---------------------------
    num_samples_train = 5000 # 训练样本数 / number of training samples
    num_samples_test = 5000  # 测试样本数 / number of test samples
    signal_dim = 64   # 频点个数 / number of frequency points
    dim = 64  
    max_n_R = 10 # 最大目标数 / maximum number of targets
    min_sep = 0.7  # 最小距离 / minimum separation
    snr_min = 0  # 最小信噪比 / minimum SNR
    snr_max = 50 # 最大信噪比 / maximum SNR
    batch_size = 1024 # batch size
    upsample = 8 # 上采样倍数 / upsampling factor
    double_upsample = upsample*2 # 双倍上采样 / double upsampling
    gs_size = 0.02/signal_dim # 高斯模糊核大小 / Gaussian kernel size
    
    # Create dataset loaders
    training_data_loader = gen_signal(num_samples_train, signal_dim, max_n_R, min_sep,snr_min,snr_max,batch_size,double_upsample,gs_size)
    testing_data_loader = gen_signal(num_samples_test, signal_dim, max_n_R, min_sep,snr_min,snr_max,batch_size,double_upsample,gs_size)
    
    # 针对不同层的网络进行迭代 / Iterate different network depths
    layer_num = 3
    print('===> Building model ', opt.model_type)
    
    # 选择网络模型（模型指导或非模型指导）/ Choose network model (model-guided or non-model-guided)
    model = DSSR_net(dim,layer_num,signal_dim,upsample)

    save_folder_new = opt.save_folder +'/'
    model = torch.nn.DataParallel(model, device_ids=gpus_list)
    criterion = torch.nn.MSELoss(reduction='sum')

    print('---------- Networks architecture -------------')
    print_network(model)
    print('----------------------------------------------')

    # Load pretrained model if specified
    if opt.Ispretrained:
        model_name = os.path.join(opt.pretrained, opt.pretrained_sr)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print(model_name + ' model is loaded.')

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.5, verbose=True)
    PSNR = []
    Acc = []
    loss_table = []
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        
        adjust_learning_rate(optimizer, epoch, lr=opt.lr)
        
        loss_table.append(train(epoch).cpu().detach().numpy())
        
        psnr = test(testing_data_loader)
        PSNR.append(psnr)
        
        # Save training statistics
        data_frame = pd.DataFrame(
            data={'epoch': epoch, 'PSNR': PSNR, 'Loss':loss_table}, index=range(1, epoch+1)
        )
        data_frame.to_csv(os.path.join(opt.statistics, '_layers_'+str(layer_num)+'_gs_size_' +str(gs_size)+'_training_logs_gass.csv'), index_label='index')
        
        # Save model every 25 epochs
        if (epoch) % 25 == 0:
            model.eval()
            SC = 'net_epoch_upsample_'+ str(upsample) +'_layers_' +str(layer_num)+'_max_n_R_' +str(max_n_R)+'_gs_size_' +str(gs_size)+'_'+ str(epoch) + '_final_gass'+'.pth'
            torch.save(model.state_dict(), os.path.join(save_folder_new, SC))
            model.train()
