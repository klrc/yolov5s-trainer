"""Train a YOLOv5 model on a custom dataset

Usage:
    $ python scripts/train.py --data coco.yaml --img 416 --weights prototype/ancient.pt --hyp data/hyps/hyp.finetune.yaml --batch-size 32 --epochs 100
"""

from _utils.loggers import Loggers
from _utils.metrics import fitness
from _utils.loggers.wandb.wandb_utils import check_wandb_resume
from _utils.torch_utils import select_device, de_parallel
from _utils.plots import plot_labels
from _utils.loss import ComputeLoss
from _utils.general import labels_to_class_weights, increment_path, init_seeds, \
    get_latest_run, check_dataset, check_file, check_img_size, set_logging, one_cycle, colorstr
from _utils.datasets import create_dataloader
from _utils.autoanchor import check_anchors
from _utils.melt4 import shrink
from _utils.counter import Counter
from _utils.models import load_model

import val
import argparse
import logging
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path


LOGGER = logging.getLogger(__name__)


def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, resume, noval, nosave, workers, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, \
        opt.resume, opt.noval, opt.nosave, opt.workers

    # Detect anomaly
    if opt.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)

    # Config
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(0)
    data_dict = check_dataset(data)  # check
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    print(data_dict)
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # Loggers
    loggers = Loggers(save_dir, weights, opt, hyp, data_dict, LOGGER).start()  # loggers dict
    if loggers.wandb and resume:
        weights, epochs, hyp, data_dict = opt.weights, opt.epochs, opt.hyp, loggers.wandb.data_dict

    # Model
    model, ckpt, csd = load_model(weights, device, nc)
    LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight with decay
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight without decay
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2  # free memory

    # Scheduler
    if opt.linear_lr:
        def lf(x): return (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if weights.endswith('.pt') and 'optimizer' in ckpt and 'epoch' in ckpt:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        # free memory
        del ckpt, csd

    # Image sizes
    gs = 32  # grid size (max stride)
    nl = 3  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Trainloader
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, single_cls,
                                              hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=-1,
                                              workers=workers, quad=opt.quad,
                                              prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    val_loader = create_dataloader(val_path, imgsz, batch_size * 2, gs, single_cls,
                                   hyp=hyp, cache=opt.cache_images and not noval, rect=True, rank=-1,
                                   workers=workers, pad=0.5,
                                   prefix=colorstr('val: '))[0]

    if not resume:
        labels = np.concatenate(dataset.labels, 0)
        if plots:
            plot_labels(labels, names, save_dir, loggers)

        # Anchors
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
        model.half().float()  # pre-reduce anchor precision

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    # early-stop counter initialization
    fitness_counter = Counter(patience=10)
    is_training_stuck = False

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if opt.quad:
                    loss *= 4.

            # Backward
            with torch.autograd.set_detect_anomaly(True):
                scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                last_opt_step = ni

            # Log
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
            loggers.on_train_batch_end(ni, model, imgs, targets, paths, plots)

            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        # mAP
        loggers.on_train_epoch_end(epoch)
        final_epoch = epoch + 1 == epochs
        if not noval or final_epoch:  # Calculate mAP
            results, maps, _ = val.run(data_dict,
                                       batch_size=batch_size * 2,
                                       imgsz=imgsz,
                                       model=model.eval(),
                                       single_cls=single_cls,
                                       dataloader=val_loader,
                                       save_dir=save_dir,
                                       save_json=is_coco and final_epoch,
                                       verbose=nc < 50 and final_epoch,
                                       plots=plots and final_epoch,
                                       loggers=loggers,
                                       compute_loss=compute_loss,
                                       nc=nc)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi
        loggers.on_train_val_end(mloss, results, lr, epoch, best_fitness, fi)
        is_training_stuck = fitness_counter.step(fi)

        # Save model
        last, best = w / 'last.pt', w / 'best.pt'
        if (not nosave) or (final_epoch and not evolve):  # if save
            ckpt = {'epoch': epoch,
                    'best_fitness': best_fitness,
                    'state_dict': deepcopy(de_parallel(model)).half().state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt
            loggers.on_model_save(last, epoch, final_epoch, best_fitness, fi)

        # check if need fpgm
        # FPGM for Xyolov5s only ------------------------
        if is_training_stuck:
            model.cpu().dsp()
            model = shrink(
                model,
                mode='fpgm',
                blacklist=['backbone_stage_1.0.focus', 'detect.m.0', 'detect.m.1', 'detect.m.2'],
            )
            model.to(device).torch()
            born = w / 'born.pt'
            ckpt = {'epoch': epoch,
                    'state_dict': deepcopy(de_parallel(model)).half().state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}
            torch.save(ckpt, born)
            return born
        # ----------------------------------------------

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    LOGGER.info(f'{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')
    torch.cuda.empty_cache()
    return


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='default', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--detect_anomaly', action='store_true', help='enable detect anomaly for debugging')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    set_logging(-1)
    print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))

    # Resume
    _epochs = opt.epochs  # keep new target epochs
    if opt.resume and not check_wandb_resume(opt):  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate

        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.name = 'evolve' if opt.evolve else opt.name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok or opt.evolve))
    opt.epochs = _epochs

    # Train
    device = select_device(opt.device, batch_size=opt.batch_size)
    return train(opt.hyp, opt, device)


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    return main(opt)


def fpgm_run():
    options = {
        'data': 'coco-l.yaml',
        'weights': 'runs/train/exp91/weights/last.pt',
        'imgsz': 416,
        'batch-size': 32,
        'hyp': 'data/hyps/hyp.finetune.yaml',
    }
    next_gen_seed = run(**options)
    while next_gen_seed is not None:
        options['weights'] = str(next_gen_seed)
        next_gen_seed = run(**options)


if __name__ == "__main__":
    # opt = parse_opt()
    # main(opt)
    fpgm_run()
