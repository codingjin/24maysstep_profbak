import warnings
import argparse
import torch
import torch.utils.data as data
import numpy as np

from collections import defaultdict
import torch.nn as nn
import time
from myutils import read_csv, write_csv, create_plt, csv_dict_writer, CNNTimers, Timers_CUDA, Timers_Time, \
    CNNBasicMetrics, convert_state_dict_ddp, convert_state_dict_noddp, set_worker_sharing_strategy

# import tensorboard
import monai.transforms as mtransforms
import sys
import random
import logging
import os
from torchmetrics import AUROC, MetricCollection, Accuracy, Precision, Recall
import io
# from utils import Timers_Time as mytimer #see if __name__ == '__main__': for dymanic usage
import gc
#import tracemalloc
#import signal
import math

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from torch.utils.data import Dataset

import torchvision.models.vgg as vgg_tv
import torchvision.models.resnet as resnet
import torchvision.models as tvm #weights
from torchvision.models import VGG16_Weights

import data_utils.wsi_dataset as wsi_dataset
from data_utils.wsi_dataset import collate_WSIImage #Todo, hide this from user?
from data_utils.image_loader import CuImageLoader
from data_utils.wsi_image import WSIImage
from data_utils.wsi_dataset import WSIList

from sstep_module_wrapper import SstepModuleWrapper

from myutils import convert_state_dict_ddp, convert_state_dict_noddp


from torch.profiler import profile, record_function, ProfilerActivity

# torch.multiprocessing.set_sharing_strategy("file_system")

#TODO: Hide this from user
def set_worker_sharing_strategy(worker_id: int) -> None:
    #torch.multiprocessing.set_sharing_strategy("file_descriptor")
    torch.multiprocessing.set_sharing_strategy('file_system')

# debug fake data set
class RanTensorDataset(Dataset):
    def __init__(self, B, chanel, H, W):
        self.samples = []
        for i in range(0, B):
            image = torch.rand(chanel, H, W)
            label = torch.rand(1)
            self.samples.append((image, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def disktest(model, criterion, optimizer, train_loader, test_loader, num_epochs=25, device="cpu", modprint=1,
             fn="saved_model", is_tiled_version=False, tile_size=240, is_DDP=False):
    for epoch in range(args.nepochs):
        t0 = time.time()
        for i, (data, target) in enumerate(train_loader):
            t1 = time.time()
            print("Data Loader Sync Time: %lf" % ((t1 - t0) * 1000))
            t0 = time.time()
        for i, (data, target) in enumerate(test_loader):
            t1 = time.time()
            print("Data Loader Sync Time: %lf" % ((t1 - t0) * 1000))
            t0 = time.time()
    return ({}, {}, {})


#TILED READER IMPL: This will likly fail for input as we dont have a tensor -- Its never called so might
# be dead code anyway, but add an assert just to make sure
def input_image_preprocess(input):

    if isinstance(input,WSIImage):
        raise NotImplementedError("input_image_preprocess not implemented for tiled WSIImage class")

    img_size = input.size()
    tile_size = 1024
    H_round = math.ceil(img_size[2] / tile_size)
    W_round = math.ceil(img_size[3] / tile_size)
    H_diff = H_round * tile_size - img_size[2]
    W_diff = W_round * tile_size - img_size[3]

    p2d = (W_diff // 2, W_diff - (W_diff // 2), H_diff // 2, H_diff - (H_diff // 2))
    # pad last dim(W) by first 2 element[0,1] and 2nd to last dim(H) by next 2 element[2,3]

    input = torch.nn.functional.pad(input, p2d, mode='constant', value=0.0)
    new_img_size = input.size()
    # print("input new size", input.size())
    # nTh, nTw
    return input, new_img_size[0], new_img_size[1], new_img_size[2], new_img_size[3], H_round, W_round

class wsiLogger:
    def __init__(self, fn_base):
        self.fn_base = fn_base
        self.results = {}
        self.results["epoch"] = defaultdict(list)
        # self.results["test_epoch"] = defaultdict(list)
        # self.results["train_epoch"] = defaultdict(list)
        self.results["train"] = defaultdict(list)
        self.results["test"] = defaultdict(list)
        self.results["train_instances"] = defaultdict(list)
        self.results["test_instances"] = defaultdict(list)

        self.epoch_writer = csv_dict_writer(fn_base + ".epoch.csv")
        self.train_writer = csv_dict_writer(fn_base + ".train.csv")
        self.test_writer = csv_dict_writer(fn_base + ".test.csv")
        self.traininst_writer = csv_dict_writer(fn_base + ".train_instance.csv")
        self.testinst_writer = csv_dict_writer(fn_base + ".test_instance.csv")

    def logInstances(self, rank, epoch_id, batch_id, model_out, target, img_file, t="train"):
        """
        Instances are stored locally when logged. All stats are synced at the end of an epoch for performance.
        """
        if (t == "train"):
            results = self.results["train_instances"]
        else:
            results = self.results["test_instances"]
        batch_len = target.shape[0]

        for i in range(batch_len):
            results['epoch_id'].append(epoch_id)
            results['rank'].append(rank)
            results['batch_id'].append(batch_id)
            # results["instance_id"].append(instance_id)
            results['output'].append(model_out.detach().cpu()[i].numpy()[:2]) #only store the first 2 elements for now. 
            results['label'].append(target.detach().cpu()[i].item())
            results['img'].append(img_file)

    def logBatch(self, rank, epoch_id, batch_id, blen, shape, loss, batch_metrics, basic_metrics, overall_metrics,
                 timer, t="train", pr=True, tileInfo=[0,0]):
        if (t == "train"):
            results = self.results["train"]
        else:
            results = self.results["test"]

        # store results and timing
        results['epoch_id'].append(epoch_id)
        results['rank'].append(rank)
        results['batch_id'].append(batch_id)
        results['size'].append(blen)
        results['shape'].append("(%d,%d,%d)" % (shape[0], shape[1], shape[2]))
        results['tile_count'].append("(%d,%d)" % (tileInfo[0], tileInfo[1]))


        results['loss'].append(loss)
        results['acc'].append(batch_metrics["BinaryAccuracy"].item())

        results['load_ms'].append(timer.get("load"))
        results['fwd_ms'].append(timer.get("fwd"))
        if (t == "train"):
            results['bwd_ms'].append(timer.get("bwd"))
        results['batch_ms'].append(timer.get("batch"))

        if (pr):
            bwd_str = ""; origTiles=0; filteredTiles=0
            if (t == "train"):
                bwd_str = " Bwd: %0.3fms" % results['bwd_ms'][-1]

            print(
                "\t%s - Rank:%d Epoch: %d Batch: %d BatchLoss: %0.3f BatchAcc: %0.2f OverallLoss: %0.3f OverallAcc: %0.2f AUC-ROC:%0.2f Loader: %0.3fms Fwd: %0.3fms%s W:%d H:%d Pixels:%d origTiles:%d filteredTiles:%d" %
                (t.capitalize(), rank, epoch_id, batch_id, loss, batch_metrics["BinaryAccuracy"].item(),
                 basic_metrics.compute("loss"), overall_metrics["BinaryAccuracy"], overall_metrics["BinaryAUROC"],
                 results['load_ms'][-1], results['fwd_ms'][-1], bwd_str, shape[1], shape[2], shape[1] * shape[2], tileInfo[0], tileInfo[1]))

    def logEpoch(self, rank, epoch_id, num_batches, basic_metrics, overall_metrics, timer, t="train", pr=True):
        results = self.results["epoch"]

        if (t == "train"):  # this is fragile, assumes train is always run ...
            results['epoch_id'].append(epoch_id)
        results[t + '_num_batches'].append(num_batches)
        results[t + '_num_instances'].append(int(basic_metrics.compute("instances")))
        results[t + '_loss'].append(basic_metrics.compute("loss"))
        for k in overall_metrics.keys():
            results[t + '_' + k].append(overall_metrics[k].item())
        results[t + '_load_ms'].append(basic_metrics.compute("load_ms"))
        results[t + '_fwd_ms'].append(basic_metrics.compute("fwd_ms"))
        if (t == "train"):
            results[t + '_bwd_ms'].append(basic_metrics.compute("bwd_ms"))
        results[t + '_batch_ms'].append(basic_metrics.compute(
            "batch_ms"))  # expect the sum of the batches to be VERY close to the total train time, record anyway
        results[t + '_train_ms'].append(timer.get("train"))

        if rank == 0:  # this should work distirbuted now that we are using torchmetrics
            bwd = ""
            if (t == "train"):
                bwd = "bwd: %0.3f " % basic_metrics.compute("bwd_ms")
            tot = ""

            print(
                "%s\tEpoch: %d NumBatches: %d Instances: %d Loss: %0.3f Acc: %0.2f AUC-ROC: %0.2f Load: %0.3fms Fwd: %0.3fms %sSumBatch: %0.3f Total: %0.3fms" %
                (t.capitalize(), epoch_id, num_batches, int(basic_metrics.compute("instances")),
                 basic_metrics.compute("loss"), overall_metrics["BinaryAccuracy"], overall_metrics["BinaryAUROC"],
                 basic_metrics.compute("load_ms"), basic_metrics.compute("fwd_ms"), bwd,
                 basic_metrics.compute("batch_ms"), timer.get(t)))

    def write(self, rank):
        self.testinst_writer.write_rows(self.results["test_instances"])
        self.traininst_writer.write_rows(self.results["train_instances"])
        self.train_writer.write_rows(self.results["train"])
        self.test_writer.write_rows(self.results["test"])
        if (rank == 0):
            self.epoch_writer.write_rows(self.results["epoch"])


# training function(train + test)
def train(model, criterion, optimizer, train_loader, test_loader, nepochs=25, device="cpu", modprint=1,
          fn="saved_model",
          rank=0, is_tiled_version=False, tile_size=1, is_DDP=False, is_mxp=False, logger=None):
    metrics = MetricCollection([AUROC(task='BINARY', num_classes=2), Accuracy(task='BINARY'), Precision(task='BINARY', num_classes=2, average='macro'),
                                Recall(task='BINARY', num_classes=2, average='macro')])
    basic_metrics = CNNBasicMetrics()

    # timers
    if (args.use_gpu):
        timer = CNNTimers(Timers_CUDA)
    else:
        timer = CNNTimers(Timers_Time)

    scaler = torch.cuda.amp.GradScaler()
    
    tbprof_folder_name = args.resnet_model
    if (args.use_sstep):
        tbprof_folder_name = tbprof_folder_name + "_sstep"
    if (args.read_full_tensor == False):
        tbprof_folder_name = tbprof_folder_name + "_tile"
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('./tblog/' + tbprof_folder_name),
                 record_shapes=True
    ) as prof:
        for epoch in range(nepochs):
            timer.start("epoch")

            if (args.run_train):
                model.train()

                metrics.reset()
                basic_metrics.reset()

                timer.start("train")
                timer.start("load")

                batch_id = 0
                


                #TILED READER IMPL: Data is now a WSIImage
                for data, target in train_loader:
                    timer.stop("load")
                    timer.start("batch")
                    target = target.to(device)

                    #TILED READER IMPL (data has no method "to" as its a WSIImage -- If statment should catch but...
                    if (not args.use_sstep):
                        assert(not isinstance(data, WSIImage))
                        data = data.to(device)  # The tiled loader moves one tile at a time, dont load the entire image!

                    timer.start("fwd")

                    # this logic can be simplified, but not till after we sort out the mxp stuff.
                    if is_tiled_version and not is_DDP:  # regular tiled version
                        if is_mxp:
                            with torch.cuda.amp.autocast():
                                output = model(data)
                        else:
                            output = model(data)
                    elif is_tiled_version and is_DDP:

                        output = model(data)
                    elif not is_tiled_version and is_DDP:
                        # print("regular + ddp")

                        assert(not isinstance(data,WSIImage))
                        output = model(data)
                    else:
                        # torchvision.utils.save_image(data, 'img1.png')
                        # Prefer to not duplicate the code in with autocast().
                        # I think it is best to use autocast=torch.cuda.amp.autocast(); autocast.__enter__();do stuff; autocast.__exit__(), maybe adding the try to
                        # this way the autocast code can be inside conditionals, and only one "do stuff" code is requiured.
                        # want to wait until we have it working first though ;)
                        if is_mxp:
                            with torch.cuda.amp.autocast():
                                output = model(data)
                        else:                
                            output = model(data)

                    predicted = output.data.max(1)[1]  # get index of max
                    loss = criterion(output, target)

                    #print(f"predicted: {predicted} output:{output[0][:2]} target:{target}")

                    timer.stop("fwd")
                    if(isinstance(data, WSIList)):
                        slidefn = data[0].filename
                    else:
                        slidefn = "fn_dummy_tensor"
                    logger.logInstances(rank, epoch, batch_id, output, target, slidefn)

                    # TODO: call gc every 20 images, requiured? helpful? hide from user?
                    if batch_id % 20 == 0:
                        gc.collect()

                    # TODO: I expect the detach version to work with and without mxp?
                    if is_mxp:
                        c_loss = loss.detach().cpu().numpy()  # get the loss as a float for display
                    else:
                        c_loss = loss.item()  # get the loss as a float for display

                    optimizer.zero_grad()  # Zero out any cached gradients

                    timer.start("bwd")

                    if is_mxp:
                        scaler.scale(loss).backward()  # Backward pass
                    else:
                        loss.backward()  # Backward pass
    
                    optimizer.step()  # Update the weights
                    timer.stop("bwd")
                    basic_metrics.update("instances", target.size(0))  # total samples in mini batch

                    # calc timers
                    timer.stop("batch")
                    timer.synchronize() #sync with GPU for timing

                    #TODO: Hacked output to get it to run, thiunk i have a problem with the number of classes ...
                    metrics.update(output[0][:1].detach().cpu(), target.detach().cpu())
                    basic_metrics.update("loss", c_loss)
                    basic_metrics.update("load_ms", timer.get("load"))
                    basic_metrics.update("fwd_ms", timer.get("fwd"))
                    basic_metrics.update("bwd_ms", timer.get("bwd"))
                    basic_metrics.update("batch_ms", timer.get("batch"))

                    overall_metrics = metrics.compute()
                    batch_metrics = metrics(output[0][:1].detach().cpu(), target.detach().cpu())
                    timer.synchronize()
                    tileCount=[0,0]
                    if isinstance(data, WSIList):
                        tileCount[0]=data[0].original_tile_count
                        tileCount[1]=data[0].filtered_tile_count

                    logger.logBatch(rank, epoch, batch_id, target.size(0), data[0].shape, c_loss, batch_metrics, basic_metrics,
                                    overall_metrics, timer, t="train", pr=((batch_id + 1) % modprint == 0), tileInfo=tileCount)
                    timer.start("load")
                    batch_id+=1

                timer.stop("train")
                timer.synchronize()

                overall_metrics = metrics.compute()
                logger.logEpoch(rank, epoch, len(train_loader.dataset), basic_metrics, overall_metrics, timer, t="train")

            if (args.run_test):
                ####### Test Phase #######
                timer.start("test")
                metrics.reset()
                basic_metrics.reset()
                model.eval()

                batch_id = 0  #TODO: want the number of iterations at the end ... better to enumerate?
                timer.start("load")
                for data, target in test_loader:
                    timer.stop("load")
                    timer.start("batch")

                    target = target.to(device)
                    if (not args.use_sstep): data = data.to(
                        device)  # The tiled loader moves one tile at a time, dont load the entire image!

                    timer.start("fwd")
                    if is_tiled_version:  # and not is_DDP: #regular tiled version
                        output = model(data) 
                    else:  # not is_tiled_version
                        output = model(data)
    
                    predicted = output.data.max(1)[1]  # get index of max
                    loss = criterion(output, target)
                    timer.stop("fwd")

                    if(isinstance(data, WSIList)):
                        slidefn = data[0].filename
                    else:
                        slidefn = "fn_dummy_tensor"
                    logger.logInstances(rank, epoch, batch_id, output, target, slidefn, t="test")

                    metrics.update(output[0][:1].detach().cpu(), target.detach().cpu())
                    basic_metrics.update("loss", loss.detach().cpu())
    
                    c_loss = loss.item()
                    basic_metrics.update("instances", target.size(0))  # total samples in mini batch
        
                    timer.stop("batch")
                    timer.synchronize()

                    basic_metrics.update("load_ms", timer.get("load"))
                    basic_metrics.update("fwd_ms", timer.get("fwd"))
                    basic_metrics.update("batch_ms", timer.get("batch"))

                    overall_metrics = metrics.compute()
                    batch_metrics = metrics(output[0][:1].detach().cpu(), target.detach().cpu())


                    if isinstance(data, WSIList):
                        tileCount[0]=data[0].original_tile_count
                        tileCount[1]=data[0].filtered_tile_count
                    logger.logBatch(rank, epoch, batch_id, target.size(0), data[0].shape, c_loss, batch_metrics, basic_metrics,
                                    overall_metrics, timer, t="test", pr=((batch_id + 1) % modprint == 0), tileInfo=tileCount)

                    timer.start("load")
                    batch_id+=1

                timer.stop("test")
                timer.stop("epoch")
                timer.synchronize()

                logger.logEpoch(rank, epoch, len(test_loader.dataset), basic_metrics, overall_metrics, timer, t="test")
                if (args.save_stats):  # does this handle the race condition?
                    logger.write(rank)

            if (args.save_model and (not is_DDP or rank == 0)):
                torch.save(model.state_dict(), "%s.epoch%d" % (fn, epoch + 1))
    
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))

    return

def warmup_train(model, criterion, optimizer, train_loader, test_loader, nepochs=25, device="cpu", modprint=1,
          fn="saved_model",
          rank=0, is_tiled_version=False, tile_size=1, is_DDP=False, is_mxp=False, logger=None):
    metrics = MetricCollection([AUROC(task='BINARY', num_classes=2), Accuracy(task='BINARY'), Precision(task='BINARY', num_classes=2, average='macro'),
                                Recall(task='BINARY', num_classes=2, average='macro')])

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(nepochs):
        if (args.run_train):
            model.train()
            batch_id = 0

            #TILED READER IMPL: Data is now a WSIImage
            for data, target in train_loader:
                target = target.to(device)

                #TILED READER IMPL (data has no method "to" as its a WSIImage -- If statment should catch but...
                if (not args.use_sstep):
                    assert(not isinstance(data, WSIImage))
                    data = data.to(device)  # The tiled loader moves one tile at a time, dont load the entire image!

                # this logic can be simplified, but not till after we sort out the mxp stuff.
                if is_tiled_version and not is_DDP:  # regular tiled version
                    if is_mxp:
                        with torch.cuda.amp.autocast():
                            output = model(data)
                    else:
                        output = model(data)
                elif is_tiled_version and is_DDP:

                    output = model(data)
                elif not is_tiled_version and is_DDP:
                    # print("regular + ddp")

                    assert(not isinstance(data,WSIImage))
                    output = model(data)
                else:
                    # torchvision.utils.save_image(data, 'img1.png')
                    # Prefer to not duplicate the code in with autocast().
                    # I think it is best to use autocast=torch.cuda.amp.autocast(); autocast.__enter__();do stuff; autocast.__exit__(), maybe adding the try to
                    # this way the autocast code can be inside conditionals, and only one "do stuff" code is requiured.
                    # want to wait until we have it working first though ;)
                    if is_mxp:
                        with torch.cuda.amp.autocast():
                            output = model(data)
                    else:                
                        output = model(data)

                predicted = output.data.max(1)[1]  # get index of max
                loss = criterion(output, target)

                #print(f"predicted: {predicted} output:{output[0][:2]} target:{target}")


                if(isinstance(data, WSIList)):
                    slidefn = data[0].filename
                else:
                    slidefn = "fn_dummy_tensor"

                # TODO: call gc every 20 images, requiured? helpful? hide from user?
                if batch_id % 20 == 0:
                    gc.collect()

                # TODO: I expect the detach version to work with and without mxp?
                if is_mxp:
                    c_loss = loss.detach().cpu().numpy()  # get the loss as a float for display
                else:
                    c_loss = loss.item()  # get the loss as a float for display

                optimizer.zero_grad()  # Zero out any cached gradients

                if is_mxp:
                    scaler.scale(loss).backward()  # Backward pass
                else:
                    loss.backward()  # Backward pass
   
                optimizer.step()  # Update the weights

                tileCount=[0,0]
                if isinstance(data, WSIList):
                    tileCount[0]=data[0].original_tile_count
                    tileCount[1]=data[0].filtered_tile_count

                batch_id+=1

    return

# every single process handle this function; entry point of mp.Process()
def indiv_train(gpu, args, logger, fn="saved_model", ):
    """ Initialize the distributed environment. """
    world_size = args.gpus * args.nodes
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank)

    block = args.model.block1

    # Wrap the model
    model = args.model

    device = torch.device(
        f'cuda:{rank % args.gpus + args.gpu_id}')  # presumes nr=1? replaced rank with (rank%args.ddp_gpus+args.gpu_id)
    model = model.to(device) 
    if (args.use_sstep and not args.vgg16_custom):
        #TODO: This needs debuged. Does the wrapper work the DDP? Biggest concern is parameters being tracked...
        model = SstepModuleWrapper(model, args.train_dataset[0], tile_size=args.tile_size)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], tt_model=args.use_sstep)

    if (args.load_model != None):
        # we can not currently use a ddp saved model in a non-ddp instance and vice versa, should be able to fix this by add/removing module. from each key?
        print("TODO: modify the state_dict to be able to use ddp saved state in a non-ddp module and vice versa!")
        model.load_state_dict(convert_state_dict_ddp(torch.load(args.load_model)))
    elif (args.preload_model):
        if (args.use_sstep):
            print( "Warning!!!!!!!! Need to implement vggtt.get_state_dict() in order to preload model!!!! Currently we attempt to preload during initialization!?")
        else:
            # need to rewrite the state dict to include the 'module." prefix"
            sd = args.model.get_state_dict()
            model.load_state_dict(convert_state_dict_ddp(sd), strict=False)

    criterion = nn.CrossEntropyLoss()  # nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)  # each thread must creaet its own optimzer!

    # create data loaders for train and test
    train_sampler = torch.utils.data.distributed.DistributedSampler(args.train_dataset, num_replicas=world_size, shuffle=args.shuffle, rank=rank)
    train_dataloader = DataLoader(args.train_dset, batch_size=args.batch_size, num_workers=args.workers, worker_init_fn=set_worker_sharing_strategy, collate_fn=args.collate_fn, pin_memory=True, sampler=train_sampler)

    test_sampler = torch.utils.data.distributed.DistributedSampler(args.test_dataset, num_replicas=world_size, rank=rank)
    test_dataloader = DataLoader(args.test_dset, batch_size=args.batch_size, num_workers=args.workers, worker_init_fn=set_worker_sharing_strategy, collate_fn=args.collate_fn, pin_memory=True, sampler=test_sampler)


    if rank == 0:
        print("d_set size", len(args.train_dataset), len(train_dataloader), args.use_sstep)

    # train
    train(model, criterion, optimizer, train_dataloader, test_dataloader, args.nepochs, device,
          fn=fn, modprint=args.modprint, rank=rank, is_tiled_version=args.use_sstep,
          tile_size=args.tile_size, is_DDP=True, is_mxp=args.mxprecision, logger=logger)
    
    
    dist.destroy_process_group()

    return


def main(args):
    torch.set_num_threads(1) #TODO: This is a hold over, need to verify usage is still needed

    if args.seed is not None:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    base_fn = get_fn(args) #build base filename from user arguments
    full_filename = os.getcwd() + base_fn #full path

    device = torch.device("cuda:%d" % args.gpu_id if args.use_gpu else "cpu")

    #TOOD: Pretty this up and improve usability
    image_loader = None
    collate_fn = collate_WSIImage
    if(args.read_full_tensor):
        image_loader=CuImageLoader() #TODO: Improve usability. This is currently how the DataLoader decides if it is loading full images or times.
        collate_fn=None

    #TODO: This happens on the CPU, develop an API to move this to the GPU
    #TODO: Define tilewise vs. global transform API, and throw errors when a global opp is done on a tile
    if(args.use_preproc_transforms):
        wsi_preproc = mtransforms.Compose([
                        mtransforms.NormalizeIntensity(subtrahend=0, divisor=255), #[0,1]
                        mtransforms.NormalizeIntensity(subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225], channel_wise=True)])
    else:
        wsi_preproc = None

    #TODO: Get rid of image_roundup, is it even needed anymore? Should not be. Needs tested.
    wsi_dset = wsi_dataset.WSIDataSet(args.class_folders, power=args.power, transform = wsi_preproc, img_roundup=args.img_roundup, image_loader=image_loader, filter_empty_tiles=args.use_masking, tensor_type="float32", tile_workers=args.tile_workers)
    (wsi_train_dset, wsi_test_dset) = wsi_dataset.splitDataSet(wsi_dset, args.train_split, seed=args.seed)

    print("Total Images: %d, Train Images: %d, Test Images: %d" % (len(wsi_dset), len(wsi_train_dset), len(wsi_test_dset)))

    num_classes=len(args.class_folders)
    num_classes=1000
    if (args.resnet_model == 'resnet34'):
        model = resnet.resnet34(weights=resnet.ResNet34_Weights.DEFAULT, num_classes=num_classes)
    elif (args.resnet_model == 'resnet50'):
        model = resnet.resnet50(weights=resnet.ResNet50_Weights.DEFAULT, num_classes=num_classes)
    elif (args.resnet_model == 'resnet101'):
        model = resnet.resnet101(weights=resnet.ResNet101_Weights.DEFAULT, num_classes=num_classes)
    elif (args.resnet_model == 'resnet152'):
        model = resnet.resnet152(weights=resnet.ResNet152_Weights.DEFAULT, num_classes=num_classes)
    elif (args.resnet_model == 'vgg16'):
        model = vgg_tv.vgg16(weights=vgg_tv.VGG16_Weights.DEFAULT, num_classes=num_classes)
    elif (args.resnet_model == 'vgg16_custom'):
        num_classes=1000
        if(args.use_sstep):
            from vgg import vgg16
            model = vgg16(weights=vgg_tv.VGG16_Weights.DEFAULT, num_classes=num_classes, tile_size=args.tile_size, tile_device=device)
        else:
            from vision.torchvision.models.vgg import vgg16
            model = vgg16(weights=vgg_tv.VGG16_Weights.DEFAULT, num_classes=num_classes)

        model.classifier[6] = nn.Linear(4096, 100)
    else:
        if(args.resnet_model != 'resnet18'):
            print(f"Warning: Unknown model {args.resnet_model}. Defaulting to resnet18.")
    
        model = resnet.resnet18(weights=resnet.ResNet18_Weights.DEFAULT, num_classes=num_classes)



    with open(full_filename + ".config.json", 'w') as data:
        data.write(str(args))
        data.write(str(model))

    start = time.time()
    # for ddp specific
    if args.is_ddp:
        #### for debug
        # train_dset = RanTensorDataset(len(train_dset), 3,102,102)  
        # test_dset = train_dset      
        #print("########### ddp ??", args.is_ddp)

        # TODO: Split args into args and ddp_args. Prefer that args only contain command line arguments.
        os.environ['MASTER_ADDR'] = args.ddp_addr
        os.environ['MASTER_PORT'] = str(args.ddp_port)
        args.model = model
        args.train_dataset = wsi_train_dset
        args.test_dataset = wsi_test_dset
        args.base_fn = base_fn
        args.collate_fn = collate_fn

        # result_queue = mp.Queue()
        procs = []
        for rank in range(args.gpus):
            proc = mp.Process(target=indiv_train, args=(rank, args, wsiLogger(full_filename)))
            proc.start()
            procs.append(proc)

        for p in procs:
            p.join()
          
        # assume P0 is the last one!!
        # procs[0].join()
        # torch.multiprocessing.spawn(fn=indiv_train, args=(args,None ), nprocs=args.gpus, join=True)

    else:
        train_dataloader = DataLoader(wsi_train_dset, batch_size=args.batch_size, num_workers=args.workers, worker_init_fn=set_worker_sharing_strategy, collate_fn=collate_fn)
        test_dataloader = DataLoader(wsi_test_dset, batch_size=args.batch_size, num_workers=args.workers, worker_init_fn=set_worker_sharing_strategy, collate_fn=collate_fn)


        if (args.load_model != None):
            # we can not currently use a ddp saved model in a non-ddp instance and vice versa, should be able to fix this by add/removing module. from each key?
            model.load_state_dict(convert_state_dict_noddp(torch.load(args.load_model)))
        elif (args.preload_model):
            if (args.preload_model):
                if (args.use_sstep):
                    print(
                        "Warning!!!!!!!! Need to implement vggtt.get_state_dict() in order to preload model!!!! Currently we attempt to preload during initialization!?")
                else:
                    model.load_state_dict(model.get_state_dict(), strict=False)
        model.to(device)
        if (args.use_sstep and not args.resnet_model == "vgg16_custom"):
            model = SstepModuleWrapper(model, tile_size=args.tile_size)

        criterion = nn.CrossEntropyLoss()  # nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        
        # Warmup
        warmup_train(model, criterion, optimizer, train_dataloader, test_dataloader, args.nepochs, device,
              fn=full_filename + ".pth", modprint=args.modprint, is_tiled_version=args.use_sstep,
              tile_size=args.tile_size,
              is_DDP=args.is_ddp, is_mxp=args.mxprecision,
              logger=wsiLogger(full_filename))  # last arg reserve for tiled version
        

        train(model, criterion, optimizer, train_dataloader, test_dataloader, args.nepochs, device,
            fn=full_filename + ".pth", modprint=args.modprint, is_tiled_version=args.use_sstep,
            tile_size=args.tile_size,
            is_DDP=args.is_ddp, is_mxp=args.mxprecision,
            logger=wsiLogger(full_filename))  # last arg reserve for tiled version

    elapsed = time.time() - start
    print("Total time: %lf seconds" % (elapsed))


def get_fn(args):
    tt_str = "no_tiled"
    ddp_str = "no_ddp"
    gpu_str = "cpu"
    shuffle_str = "no_shuffle"
    mxp_str = "no_mxp"
    preproc_str = "no_preproc"

    model = args.resnet_model
    if (args.use_sstep):
        tt_str = "tiled.tilesize%d" % args.tile_size
    if (args.is_ddp):
        ddp_str = "ddp.ddp_gpus%d.ddp_nodes%d" % (args.gpus, args.nodes)
    if (args.use_gpu):
        gpu_str = "gpu%d" % args.gpu_id
    if (args.shuffle):
        shuffle_str = "shuffle"
    if (args.mxprecision):
        mxp_str = "mxp"
    if (args.use_preproc_transforms):
        preproc_str = "preproc"
    #if (args.resnet_model == "vgg_wsi"):
    #    model = args.resnet_model + ".gpool_%s" % args.vgg_gpool

    return "%swsi%ld.%s.train%0.3f.batch_size%d.nepochs%d.lr%0.6f.power%0.3f.roundup%d.%s.%s.%s.%s.%s.%s" % \
           (args.fnbase, int(time.time()), model, args.train_split, args.batch_size, args.nepochs,
            args.learning_rate, args.power, args.tile_size, shuffle_str, gpu_str, tt_str, ddp_str, mxp_str, preproc_str)


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=UserWarning, module="torchmetrics.utilities.prints")


    parser = argparse.ArgumentParser(description='WSI Openslide VGG Demo')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='mini-batch size (default: 1)')
    parser.add_argument('-e', '--nepochs', type=int, default=20, help='number of epochs (default: 20)')
    parser.add_argument('-w', '--workers', default=4, type=int, help='number of data loading workers for the WSI DL(default: 4)')
    parser.add_argument('-W', '--tile_workers', default=16, type=int, help='number of data loading workers for tiles inside of run(default: 16)')

    parser.add_argument('-r', '--learning_rate', type=float, default=5e-4, help='Set the learning rate (default: 5e-4)')
    parser.add_argument('-p', '--power', type=float, default=1.25,
                        help='Scale OpenSlide images to this objective power (default:1)')
    parser.add_argument('-f', '--fnbase', type=str, default="", help='Base string for output file names (default:"")')
    parser.add_argument('-c', '--class_folders', nargs='+', default=[])#["/data/wsi_data/lusc_test", "/data/wsi_data/luad_test"])
    parser.add_argument('-l', '--load_model', type=str, default=None,
                        help='Location of pth file to load (default:None)')
    parser.add_argument('-v', '--resnet_model', type=str, default="vgg16_custom",
                        help='resnet18, resnet34, resnet50, resnet101, resnet152, vgg16_custom (default: resnet18).')
    parser.add_argument('-t', '--train_split', type=float, default=0.75,
                        help='percent train split as a float, vgg_wsi (default: 0.75)')
    parser.add_argument('-m', '--modprint', type=int, default=1,
                        help='Control printing of each batch stats (default: 1)')
    parser.add_argument('--seed', type=int, default=None, help='Seed the test/train randomizer (default: None)')
    parser.add_argument('--img_roundup', type=int, default=1,
                        help='Pad images to be divisible by this number, this can be set to the number of desired tiles to end up with even tiles (default: 1)')
    parser.add_argument('--ddp_port', type=int, default=8888, help='DDP Connection Network Port (default: 8888)')
    parser.add_argument('--ddp_addr', type=str, default='172.18.0.1',
                        help='DDP Connection Adrress/IP (default: 172.18.0.1)')
    #parser.add_argument('--gpool', type=str, default='max',
    #                    help='Use max or avg for the global pool stage in Model (default:max)')
    parser.add_argument("-T", "--tile_size", type=int, default=1300, help="Tile size")

    parser.add_argument('--nodes', default=1, type=int, metavar='N', help=' ')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')

    parser.add_argument('--use_sstep', dest='use_sstep', action='store_true', help='Tiled execution (default:False)')
    parser.add_argument('--no_sstep', dest='use_sstep', action='store_false', help='Tiled execution (default:False)')
    parser.set_defaults(use_sstep=True)

    parser.add_argument('--use_ddp', dest='is_ddp', action='store_true', help='Using DDP in execution (default:False)')
    parser.add_argument('--no_ddp', dest='is_ddp', action='store_false', help='Using DDP in execution (default:False)')
    parser.set_defaults(is_ddp=False)

    parser.add_argument('-g', '--use_gpu', dest='use_gpu', action='store_true', help='enable cuda (default:False)')
    parser.add_argument('--use_cpu', dest='use_gpu', action='store_false', help='enable cuda (default:False)')
    parser.set_defaults(use_gpu=True)
    parser.add_argument('-i', '--gpu_id', type=int, default=0,
                        help='Select a GPU to use. Only used if use_gpu==True (default: 0)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s', '--save_stats', dest='save_stats', action='store_true',
                        help='Save train batch, test batch, and epoch results to csv? (default:True)')
    group.add_argument('--no_stats', dest='save_stats', action='store_false',
                        help='Save train batch, test batch, and epoch results to csv? (default:True)')
    group.set_defaults(save_stats=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--save_model', dest='save_model', action='store_true',
                        help='Save the best model found? (default:False)')
    group.add_argument('--no_save_model', dest='save_model', action='store_false',
                        help='Save the best model found? (default:False)')
    group.set_defaults(save_model=False)

    #This has not been tesetd in sstep_demo, it is a hangover from an older app. Hence, it is ignored here. The model is always preloaded at the moment. 
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--preload_model', dest='preload_model', action='store_true',
                        help='Preload the model weights from a public source? (default:True)')
    group.add_argument('--no_preload_model', dest='preload_model', action='store_false',
                        help='Preload the model weights from a public source? (default:True)')
    group.set_defaults(preload_model=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffle during batching? (default:True)')
    group.add_argument('--no_shuffle', dest='shuffle', action='store_false',
                        help='Shuffle during batching? (default:True)')
    group.set_defaults(shuffle=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--mxp', dest='mxprecision', action='store_true',
                        help='mxprecision during training? (default:True)')
    group.add_argument('--no_mxp', dest='mxprecision', action='store_false',
                        help='mxprecision during training? (default:True)')
    group.set_defaults(mxprecision=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', dest='run_train', action='store_true',
                        help='Preform training in each epoch? (default:True)')
    group.add_argument('--no_train', dest='run_train', action='store_false',
                        help='Preform training in each epoch? (default:True)')
    group.set_defaults(run_train=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--test', dest='run_test', action='store_true',
                        help='Preform test evaluation in each epoch? (default:True)')
    group.add_argument('--no_test', dest='run_test', action='store_false',
                        help='Preform test evaluation in each epoch? (default:True)')
    group.set_defaults(run_test=True)
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use_masking', dest='use_masking', action='store_true',
                        help='Use WSI masking for the images? (default:False)')
    group.add_argument('--no_masking', dest='use_masking', action='store_false',
                        help='Use WSI masking for the images? (default:False)')
    group.set_defaults(use_masking=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--tile_reader', dest='read_full_tensor', action='store_false',
                        help='Read tiles (i.e., do not read the full tensor). Exclusive with --read_full_tensor.')
    group.add_argument('--no_tile_reader', dest='read_full_tensor', action='store_true',
                        help='Read the full tensor before tiling. Exclusive with --tile_reader.  (default:True)')
    group.set_defaults(read_full_tensor=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use_preproc_transforms', dest='use_preproc_transforms', action='store_true',
                        help='Use standard image preprocessing transforms. (default:True) Exclusive with --no_use_preproc_transforms.')
    group.add_argument('--no_preproc_transforms', dest='use_preproc_transforms', action='store_false',
                        help='Disable standard image preprocessing transforms. Exclusive with --use_preproc_transforms.')
    group.set_defaults(use_preproc_transforms=True)

    args = parser.parse_args()
    print(args)
    main(args)
