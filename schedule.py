#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import json

from save_to_csv import save_results
import utils.multiprocessing
from defaults import get_cfg_defaults

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def f(setting):
    import train_AAE
    import novelty_detector

    global idx
    idx = 0
    import torch
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(idx)
    device0 = torch.cuda.current_device()
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #CUDA_VISIBLE_DEVICES= 5,6,7
    #device = [5,6]
    print("Running on GPU: %d, %s" % (idx, torch.cuda.get_device_name(device0)))
    print("Running on GPU: %d, %s" % (idx, torch.cuda.get_device_name(device1)))

    fold_id = setting['fold']
    inliner_classes = setting['digit'] 
    logger.debug('Using fold_id: %d', fold_id)
    logger.debug('Using inliner_classes: %s', inliner_classes)
    logger.debug('Percentage used: %d', cfg.DATASET.PERCENTAGES)

    train_AAE.train(fold_id, [inliner_classes], inliner_classes, cfg, setting )

    res = novelty_detector.main(fold_id, [inliner_classes], inliner_classes, classes_count, mul, cfg=cfg)
    return res

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Scheduler for Deep Learning Model')
    parser.add_argument('--min_nsample', metavar='N', type=int, help='Minimal size of Sample (mandatory)', required=True)
    parser.add_argument('--max_nsample', metavar='N', type=int, help='Maximal size of Sample (mandatory)', required=True)
    parser.add_argument('--batch_size', metavar='N', type=int, help='Batch Size (optional)')

    parser.add_argument('--recon_scale1', metavar='X', type=float, help='Reconstruction scale factor 1 (default=2.5)', default=2.5)
    parser.add_argument('--recon_scale2', metavar='X', type=float, help='Reconstruction scale factor 2 (default=2.5)', default=2.5)
    parser.add_argument('--lambda_val', metavar='X', type=float, help='Lambda value (default=0.01)', default=0.01)
    parser.add_argument('--percentage', metavar='X', type=int, help='Percentage (default=50)', default=50)
    parser.add_argument('--full_run', metavar='V', type=bool, help='True and will run 5-fold', default=False)
  
    parser.add_argument('--input_folder', metavar='PATH', type=str, help='Basename for Input Folder (optional)', default='INPUT') 
    parser.add_argument('--output_folder', metavar='PATH', type=str, help='Basename for Output Folder (optional)', default='OUTPUT' ) 
    parser.add_argument('--config_file', metavar='PATH', type=str, help="Configure File (default='configs/mnist.yaml')", default='configs/mnist.yaml')

    args = parser.parse_args()

    #if len(sys.argv) > 1:
    #    cfg_file = 'configs/' + sys.argv[1]
    #else:
    #    cfg_file = 'configs/mnist.yaml'

    cfg_file=args.config_file
    
    mul = 0.2

    settings = []

    classes_count = 10

    for fold in range(5 if args.full_run else 1):
        for i in range(classes_count):
            settings.append(dict(fold=fold, digit=i))

    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    #cfg.freeze()

    if args.batch_size is None:
        args.batch_size=cfg.TRAIN.BATCH_SIZE
    else:
        cfg.TRAIN.BATCH_SIZE=args.batch_size

    if args.percentage is None:
        args.percentage=cfg.DATASET.PERCENTAGES
    else:
        cfg.DATASET.PERCENTAGES=args.percentage

    if args.input_folder is None:
        args.input_folder=cfg.DATASET.PATH
    else:
        cfg.DATASET.PATH=args.input_folder

    if args.output_folder is None:
        args.output_folder=cfg.OUTPUT_FOLDER
    else:
        cfg.OUTPUT_FOLDER=args.output_folder

    cfg.freeze()

    logger.debug("Min Size of Sample: %d", args.min_nsample)
    logger.debug("Max Size of Sample: %d", args.max_nsample)
    logger.debug("Reconstruction scale factor 1: %f", args.min_nsample)
    logger.debug("Reconstruction scale factor 2: %f", args.max_nsample)
    logger.debug("Lambda value: %f", args.min_nsample)
    logger.debug("Batch Size        : %d", args.batch_size)
    logger.debug("Output folder     : %s", args.output_folder)
    logger.debug('Percentage        : %d', args.percentage)
    logger.debug("Output folder     : %s", args.output_folder)
    logger.debug('Config file       : %s', cfg_file)

    print("CONFIGURATION FILE")
    print(cfg)
    print()

    print("SETTINGS")
    print(settings)
    print()

    #sys.exit(1)

    #gpu_count = utils.multiprocessing.get_gpu_count()
    #gpu_count = min(utils.multiprocessing.get_gpu_count(), 4)
    gpu_count = 1

    #CUDA_VISIBLE_DEVICES=4,5,6,7
    #results = utils.multiprocessing.map(f, gpu_count, settings)
    results=[]
    for iset in settings:
        
        iset['min_nsample']=args.min_nsample
        iset['max_nsample']=args.max_nsample
        iset['batch_size']=args.batch_size 
        iset['recon_scale1']=args.recon_scale1
        iset['recon_scale2']=args.recon_scale2
        iset['lambda_val']=args.lambda_val
        iset['percentage']=args.percentage

        if not os.path.isdir(cfg.OUTPUT_FOLDER):
            os.mkdir(cfg.OUTPUT_FOLDER)

        wf=open(cfg.OUTPUT_FOLDER+os.sep+'settings.json','w')
        json.dump(iset, wf, sort_keys=True, indent=4)
        wf.close()

        results.append( f(iset) )
        save_results(results, os.path.join(cfg.OUTPUT_FOLDER, cfg.RESULTS_NAME+'_'+str(iset['fold'])+'_'+str(iset['digit'])+'.csv'))

