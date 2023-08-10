
import os
import sys
import time
import logging

import torch.utils.data
#from zero_gradients import zero_gradients
from torch import optim
from torchvision.utils import save_image
from torch.autograd import Variable
from dataloading import make_datasets, make_dataloader
from net import Generator, Discriminator, Encoder, ZDiscriminator_mergebatch, ZDiscriminator
from utils.tracker import LossTracker
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from torch.autograd.gradcheck import zero_gradients
#import dlutils
#from dlutils.pytorch.zero_gradients import zero_gradients
from torch.autograd import Variable
# torch.autograd.set_detect_anomaly(True)

import torch

#rr=np.random.RandomState()
#rnd=rr.randint(2**sys.int_info.bits_per_digit)
#torch.manual_seed(rnd)
#import gc
#gc.collect()
#torch.cuda.empty_cache()


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def compute_jacobian_autograd(inputs, output):
    """
    :param inputs: Batch X Size (e.g. Depth X Width X Height)
    :param output: Batch X Classes
    :return: jacobian: Batch X Classes X Size
    """
    assert inputs.requires_grad

    num_classes = output.size()[1]

    jacobian = torch.zeros(num_classes, *inputs.size())
    grad_output = torch.zeros(*output.size())
    if inputs.is_cuda:
        grad_output = grad_output.cuda()
        jacobian = jacobian.cuda()

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data

    return torch.transpose(jacobian, dim0=0, dim1=1)

# Functional() API for jacobian for same as above compute_jacobian_autograd() results
def batch_jacobian(f, x):
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return torch.autograd.functional.jacobian(f_sum, x, create_graph=True)


def train(folding_id, inliner_classes, ic, cfg, settings_dict):

    min_nsample=settings_dict['min_nsample']
    max_nsample=settings_dict['max_nsample']
    recon_scale1=settings_dict['recon_scale1']
    recon_scale2=settings_dict['recon_scale2']
    lambda_val=settings_dict['lambda_val']

    #torch.manual_seed(123123)
    #rr=np.random.RandomState()
    #rnd=rr.randint(2**sys.int_info.bits_per_digit)
    #torch.manual_seed(rnd)

    logger = logging.getLogger("logger")
    logger.debug('Start training with folding_id: %d', folding_id)

    zsize = cfg.MODEL.LATENT_SIZE
    output_folder = os.path.join('results_' + str(folding_id) + "_" + "_".join([str(x) for x in inliner_classes]))
    output_folder = os.path.join(cfg.OUTPUT_FOLDER, output_folder)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_FOLDER, 'models'), exist_ok=True)

    train_set, _, _ = make_datasets(cfg, folding_id, inliner_classes)

    logger.info("Train set size: %d" % len(train_set))

    G = Generator(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    G.weight_init(mean=0, std=0.02)

    D = Discriminator(channels=cfg.MODEL.INPUT_IMAGE_CHANNELS, img_width=cfg.MODEL.INPUT_IMAGE_SIZE, img_height=cfg.MODEL.INPUT_IMAGE_SIZE)
    D.weight_init(mean=0, std=0.02)

    E = Encoder(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS, img_width=cfg.MODEL.INPUT_IMAGE_SIZE, img_height=cfg.MODEL.INPUT_IMAGE_SIZE)
    E.weight_init(mean=0, std=0.02)

    if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH:
        ZD = ZDiscriminator_mergebatch(zsize, cfg.TRAIN.BATCH_SIZE)
    else:
        ZD = ZDiscriminator(zsize, cfg.TRAIN.BATCH_SIZE)
    ZD.weight_init(mean=0, std=0.02)

    lr = cfg.TRAIN.BASE_LEARNING_RATE

    G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.999))
    ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.999))
    E_optimizer = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.999))

    BCE_loss = nn.BCELoss()
    sample = torch.randn(64, zsize)#.view(-1, zsize, 1, 1)

    os.makedirs(os.path.join(cfg.OUTPUT_FOLDER,"models"), exist_ok=True)
    tracker = LossTracker(output_folder=output_folder)

    for epoch in range(cfg.TRAIN.EPOCH_COUNT):
        G.train()
        D.train()
        E.train()
        ZD.train()

        epoch_start_time = time.time()

        data_loader = make_dataloader(train_set, cfg.TRAIN.BATCH_SIZE, torch.cuda.current_device())
        train_set.shuffle()

        if (epoch + 1) % 30 == 0:
            G_optimizer.param_groups[0]['lr'] /= 4
            D_optimizer.param_groups[0]['lr'] /= 4
            GE_optimizer.param_groups[0]['lr'] /= 4
            ZD_optimizer.param_groups[0]['lr'] /= 4
            print("learning rate change!")

        # kk = 0
        for y, x in data_loader:
            # x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)
            x = x.view(x.shape[0], -1)

            y_real_ = torch.ones(x.shape[0])
            y_fake_ = torch.zeros(x.shape[0])

            y_real_z = torch.ones(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])
            y_fake_z = torch.zeros(1 if cfg.MODEL.Z_DISCRIMINATOR_CROSS_BATCH else x.shape[0])

            #############################################

            D.zero_grad()

            D_result = D(x).squeeze()
            D_real_loss = BCE_loss(D_result, y_real_)

            z = torch.randn((x.shape[0], zsize))#.view(-1, zsize, 1, 1)
            z = Variable(z)
            
            x_fake = G(z).detach()      

            D_result = D(x_fake).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake_)

            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()

            D_optimizer.step()

            tracker.update(dict(D=D_train_loss))


            #############################################

            G.zero_grad()

            z = torch.randn((x.shape[0], zsize)) #.view(-1, zsize, 1, 1)
            z = Variable(z)

            x_fake = G(z)
            D_result = D(x_fake).squeeze()

            G_train_loss = BCE_loss(D_result, y_real_)

            G_train_loss.backward()
            G_optimizer.step()

            tracker.update(dict(G=G_train_loss))

            #############################################

            ZD.zero_grad()

            z = torch.randn((x.shape[0], zsize)).view(-1, zsize)
            z = z.requires_grad_(True)

            ZD_result = ZD(z).squeeze()
            ZD_real_loss = BCE_loss(ZD_result, y_real_z)

            z = E(x).squeeze().detach()

            ZD_result = ZD(z).squeeze()
            ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

            ZD_train_loss = ZD_real_loss + ZD_fake_loss
            ZD_train_loss.backward()

            ZD_optimizer.step()

            tracker.update(dict(ZD=ZD_train_loss))

            # #############################################


            E.zero_grad()
            G.zero_grad()

            z = E(x).squeeze()
            x_rec = G(z)

            ZD_result = ZD(z).squeeze()

            E_train_loss = BCE_loss(ZD_result, y_real_z) * 1.0

            Recon_loss = F.binary_cross_entropy(x_rec.view(x.shape[0], 1, 32, 32), x.detach().view(x.shape[0], 1, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)) * recon_scale1

            # ### Isometric Autoencoder Loss part
            x_hat = Variable(x.data, requires_grad=True)
            z = E(x_hat).squeeze()
            z_hat = Variable(z.data, requires_grad=True)

            ## Monte carlo sampling
            # num_samples= x.shape[0]
            num_samples = np.random.randint(min_nsample, max_nsample, size=1)[0] # sampling only a small number of samples for fast batch_jacobianum_samples = min()
            num_samples = min(num_samples, x_hat.shape[0]-1)
            idx = np.random.choice(x_hat.shape[0], num_samples, replace=False)
            x_hat = x_hat[idx, ...]
            z_hat = z_hat[idx, ...]

            # u_hat from S^(d-1) = {z; ||z||= 1 for z in R^d}
            u_hat = torch.randn(num_samples, zsize)
            # print(u_hat[:2, :3])
            norm  = u_hat.norm(p=2, dim=[1], keepdim=True)
            u_hat = u_hat.div(norm)
            u_hat = u_hat.unsqueeze(2) # [bs, zsize, 1]


            # Reconstruction Loss (isometric autoencoder)
            x_orig = x[idx, ...].detach()
            x_rec  = x_rec[idx, ...]
            Recon_loss_2 = torch.mean((x_rec - x_orig)**2)


            # Eqn 6 (Isometry violation Loss for Decoder)
            df_z = batch_jacobian(G, z_hat).transpose(0, 1) # using functional API for Jacobian
            df_z_uhat = df_z.bmm(u_hat)
            iso_loss = torch.mean((df_z_uhat.norm(p=2, dim=[1], keepdim=True) - 1)**2) # Eqn 6

            # Eqn 7 (Pseudo-inverse violation loss for Encoder)
            dg_x = batch_jacobian(E, x_hat).transpose(0, 1) # using functional API for Jacobian
            uhat_T_dg_x = u_hat.transpose(1,2).bmm(dg_x)
            piso_loss = torch.mean((uhat_T_dg_x.norm(p=2, dim=[1], keepdim=True) - 1)**2) # Eqn 7

           # Final_Isometry_Loss = Recon_loss_2*.5 + 1e-2*(iso_loss + piso_loss)
            Final_Isometry_Loss = Recon_loss_2*recon_scale2  + lambda_val *(iso_loss + piso_loss)

            # ###


            # # ## another method for Jacobian calculation using original Author's compute_jacobian_autograd() method
            # x_hat = Variable(x.data, requires_grad=True)
            # z = E(x_hat).squeeze()
            # z_hat = Variable(z.data, requires_grad=True)            
            # x_hat_rec = G(z_hat)

            # dg_x = compute_jacobian_autograd(x_hat, z) # Encoder
            # uhat_T_dg_x = u_hat.transpose(1,2).bmm(dg_x)
            # piso_loss_2 = torch.mean((uhat_T_dg_x.norm(p=2, dim=[1], keepdim=True) - 1)**2) # Eqn 7

            # df_z = compute_jacobian_autograd(z_hat, x_hat_rec)  # Decoder
            # df_z_uhat = df_z.bmm(u_hat)
            # iso_loss_2 = torch.mean((df_z_uhat.norm(p=2, dim=[1], keepdim=True) - 1)**2) # Eqn 6
            # print("Check same jacobians-vjp: iso/iso_2:{}/{}, piso/piso_2:{}/{}".format(iso_loss, iso_loss_2, piso_loss, piso_loss_2))
            # # ##


            (E_train_loss + Recon_loss + Final_Isometry_Loss).backward()       
            # iso_loss.backward()     
            # piso_loss.backward()
            GE_optimizer.step()
            tracker.update(dict(GE=Recon_loss, E=E_train_loss, GE2=Recon_loss_2, iso=iso_loss, piso=piso_loss))
            # #############################################

            # kk += 1
            # if kk == 2:
            #     break

        x_d = x_rec.view(x_rec.shape[0], 1, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)
        x = x.view(x.shape[0], 1, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)
        comparison = torch.cat([x, x_d])
        save_image(comparison.cpu(), os.path.join(output_folder, 'reconstruction_' + str(epoch) + '.png'), nrow=x.shape[0])

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time

        logger.info('[%d/%d] - ptime: %.2f, %s' % ((epoch + 1), cfg.TRAIN.EPOCH_COUNT, per_epoch_ptime, tracker))

        tracker.register_means(epoch)
        tracker.plot()

        with torch.no_grad():
            resultsample = G(sample).cpu()
            save_image(resultsample.view(64,
                                         cfg.MODEL.INPUT_IMAGE_CHANNELS,
                                         cfg.MODEL.INPUT_IMAGE_SIZE,
                                         cfg.MODEL.INPUT_IMAGE_SIZE),
                       os.path.join(output_folder, 'sample_' + str(epoch) + '.png'))

            

    logger.info("Training finish!... save training results")
    print("Training finish!... save training results")
    torch.save(G.state_dict(), os.path.join(cfg.OUTPUT_FOLDER, "models/Gmodel_%d_%d.pkl" %(folding_id, ic)))
    torch.save(E.state_dict(), os.path.join(cfg.OUTPUT_FOLDER, "models/Emodel_%d_%d.pkl" %(folding_id, ic)))

