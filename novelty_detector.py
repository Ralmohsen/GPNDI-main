
import os
import sys
import logging

import torch.utils.data
from torchvision.utils import save_image
from net import *
from torch.autograd import Variable
from utils.jacobian import compute_jacobian_autograd
import numpy as np
import scipy.optimize
from dataloading import make_datasets, make_dataloader, create_set_with_outlier_percentage
from evaluation import get_f1, evaluate
from utils.threshold_search import find_maximum
from utils.save_plot import save_plot
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import loggamma

import torch
#rr=np.random.RandomState()
#rnd=rr.randint(2**sys.int_info.bits_per_digit)
#torch.manual_seed(rnd)
#torch.manual_seed(123321)

def r_pdf(x, bins, counts):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return max(counts[i], 1e-308)
    if x < bins[0]:
        return max(counts[0] * x / bins[0], 1e-308)
    return 1e-308


def extract_statistics(cfg, train_set, inliner_classes, E, G):
    zlist = []
    rlist = []

    data_loader = make_dataloader(train_set, cfg.TEST.BATCH_SIZE, torch.cuda.current_device())

    # kk = 0
    for label, x in data_loader:
        x = x.view(-1, cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE)
        z = E(x.view(-1, 1, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE))
        z = E(x)
        recon_batch = G(z)
        z = z.squeeze()

        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()

        z = z.cpu().detach().numpy()

        for i in range(x.shape[0]):
            distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())
            rlist.append(distance)

        zlist.append(z)

        # kk += 1
        # if kk == 2:
        #     break

    zlist = np.concatenate(zlist)

    # Removed normed=True (https://numpy.org/doc/stable/reference/generated/numpy.histogram.html)
    counts, bin_edges = np.histogram(rlist, bins=30) # Removed normed=True due to deprecation notice

    if cfg.MAKE_PLOTS:
        plt.plot(bin_edges[1:], counts, linewidth=2)
        save_plot(r"Distance, $\left \|\| I - \hat{I} \right \|\|$",
                  'Probability density',
                  r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$",
                  cfg.OUTPUT_FOLDER + '/mnist_%s_reconstruction_error.pdf' % ("_".join([str(x) for x in inliner_classes])))

    for i in range(cfg.MODEL.LATENT_SIZE):
        plt.hist(zlist[:, i], bins='auto', histtype='step')

    if cfg.MAKE_PLOTS:
        save_plot(r"$z$",
                  'Probability density',
                  r"PDF of embeding $p\left(z \right)$",
                  cfg.OUTPUT_FOLDER + '/mnist_%s_embedding.pdf' % ("_".join([str(x) for x in inliner_classes])))

    def fmin(func, x0, args, disp):
        x0 = [2.0, 0.0, 1.0]
        return scipy.optimize.fmin(func, x0, args, xtol=1e-12, ftol=1e-12, disp=0)

    gennorm_param = np.zeros([3, cfg.MODEL.LATENT_SIZE])
    for i in range(cfg.MODEL.LATENT_SIZE):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i], optimizer=fmin)
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    return counts, bin_edges, gennorm_param


def main(folding_id, inliner_classes, ic, total_classes, mul, folds=5, cfg=None):
    #rr=np.random.RandomState()
    #rnd=rr.randint(2**sys.int_info.bits_per_digit)
    #torch.manual_seed(rnd)
    #torch.manual_seed(456789)

    logger = logging.getLogger("logger")
    logger.debug('Starting Novelty detector for folding_id: %d', folding_id)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.current_device()
    print("Running on ", torch.cuda.get_device_name(device))

    train_set, valid_set, test_set = make_datasets(cfg, folding_id, inliner_classes)

    print('Validation set size: %d' % len(valid_set))
    print('Test set size: %d' % len(test_set))

    train_set.shuffle()
    output_folder = cfg.OUTPUT_FOLDER

    G = Generator(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    E = Encoder(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS, img_width=cfg.MODEL.INPUT_IMAGE_SIZE, img_height=cfg.MODEL.INPUT_IMAGE_SIZE)

    G.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_FOLDER, "models/Gmodel_%d_%d.pkl" %(folding_id, ic))))
    E.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_FOLDER, "models/Emodel_%d_%d.pkl" %(folding_id, ic))))

    G.eval()
    E.eval()

    sample = torch.randn(64, cfg.MODEL.LATENT_SIZE).to(device)
    # sample = G(sample.view(-1, cfg.MODEL.LATENT_SIZE, 1, 1)).cpu()
    sample = G(sample).cpu()
    save_image(sample.view(64, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE), os.path.join(cfg.OUTPUT_FOLDER,'sample.png'))

    counts, bin_edges, gennorm_param = extract_statistics(cfg, train_set, inliner_classes, E, G)

    def run_novely_prediction_on_dataset(dataset, percentage, concervative=False):
        dataset.shuffle()
        dataset = create_set_with_outlier_percentage(dataset, inliner_classes, percentage, concervative)

        result = []
        gt_novel = []

        data_loader = make_dataloader(dataset, cfg.TEST.BATCH_SIZE, torch.cuda.current_device())

        include_jacobian = True

        N = (cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE - cfg.MODEL.LATENT_SIZE) * mul
        logC = loggamma(N / 2.0) - (N / 2.0) * np.log(2.0 * np.pi)

        def logPe_func(x):
            # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
            # \| w^{\perp} \|}^{m-n}
            return logC - (N - 1) * np.log(x) + np.log(r_pdf(x, bin_edges, counts))

        # kk = 0
        for label, x in data_loader:
            # x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS * cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE)
            x = x.view(-1, 1, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)
            if x.shape[0] == 1:
                x = torch.cat([x,x], dim=0)
                label = torch.cat([label, label], dim=0)
            x = Variable(x.data, requires_grad=True)

            # z = E(x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE))
            z = E(x)
            recon_batch = G(z)
            z = z.squeeze()

            if include_jacobian:
                J = compute_jacobian_autograd(x, z)
                J = J.cpu().numpy()

            z = z.cpu().detach().numpy()

            recon_batch = recon_batch.squeeze().cpu().detach().numpy()
            x = x.squeeze().cpu().detach().numpy()

            for i in range(x.shape[0]):
                if include_jacobian:
                    u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                    logD = -np.sum(np.log(np.abs(s)))  # | \mathrm{det} S^{-1} |
                    # logD = np.log(np.abs(1.0/(np.prod(s))))
                else:
                    logD = 0

                p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
                #print("Stats for p: min=%f max=%f avg=%f" % (np.min(p), np.max(p), np.average(p)))
                logPz = np.sum(np.log(p))

                # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
                # In this case, just assign some large negative value to make sure that the sample
                # is classified as unknown.
                if not np.isfinite(logPz):
                    logPz = -1000

                distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())

                logPe = logPe_func(distance)

                P = logD + logPz + logPe

                result.append(P)
                gt_novel.append(label[i].item() in inliner_classes)

            # kk += 1
            # if kk == 2:
            #     break

        result = np.asarray(result, dtype=np.float32)
        ground_truth = np.asarray(gt_novel, dtype=np.float32)
        return result, ground_truth

    def compute_threshold(valid_set, percentage):
        y_scores, y_true = run_novely_prediction_on_dataset(valid_set, percentage, concervative=True)

        minP = min(y_scores) - 1
        maxP = max(y_scores) + 1
        y_false = np.logical_not(y_true)

        def evaluate(e):
            y = np.greater(y_scores, e)
            true_positive = np.sum(np.logical_and(y, y_true))
            false_positive = np.sum(np.logical_and(y, y_false))
            false_negative = np.sum(np.logical_and(np.logical_not(y), y_true))
            return get_f1(true_positive, false_positive, false_negative)

        best_th, best_f1 = find_maximum(evaluate, minP, maxP, 1e-4)

        logger.info("Best e: %f best f1: %f" % (best_th, best_f1))
        return best_th

    def test(test_set, percentage, threshold):
        y_scores, y_true = run_novely_prediction_on_dataset(test_set, percentage, concervative=True)
        return evaluate(logger, percentage, inliner_classes, y_scores, threshold, y_true, output_folder)

    percentages = [cfg.DATASET.PERCENTAGES]
    # percentages = [50]

    results = {}

    for p in percentages:
        plt.figure(num=None, figsize=(8, 6), dpi=180, facecolor='w', edgecolor='k')
        e = compute_threshold(valid_set, p)
        results[p] = test(test_set, p, e)

    return results

