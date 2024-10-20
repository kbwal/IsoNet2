import numpy as np

from .unet import Unet
import torch
import os
from .data_sequence import Train_sets, Predict_sets
import mrcfile
from IsoNet.preprocessing.img_processing import normalize
import torch.nn as nn
import logging
from IsoNet.utils.toTile import reform3D
import sys
from tqdm import tqdm
import socket
import copy
import random
from IsoNet.utils.utils import debug_matrix
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time
#import torch._dynamo as dynamo
def find_unused_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 0))
    _, port = sock.getsockname()
    sock.close()
    return port

def ddp_train(rank, world_size, port_number, model, data_path, batch_size, acc_batches, epochs, steps_per_epoch, learning_rate, mixed_precision, model_path, fsc3d):
    
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_number
    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    batch_size = batch_size // acc_batches
    batch_size_gpu = batch_size // world_size

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()

    model = DDP(model, device_ids=[rank])
    # if torch.__version__ >= "2.0.0":
    #     GPU_capability = torch.cuda.get_device_capability()
    #     if GPU_capability[0] >= 7:
    #         torch.set_float32_matmul_precision('high')
    #         model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    #from chatGPT: The DistributedSampler shuffles the indices of the entire dataset, not just the portion assigned to a specific GPU. 
    train_dataset = Train_sets(data_path)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_gpu, persistent_workers=True,
                                             num_workers=4, pin_memory=True, sampler=train_sampler)

    steps_per_epoch_train = steps_per_epoch
    total_steps = min(len(train_loader)//acc_batches, steps_per_epoch)
    average_loss_list = []
    loss_fn = nn.L1Loss()
    from IsoNet.utils.utils import debug_matrix
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        with tqdm(total=total_steps, unit="batch", disable=(rank!=0)) as progress_bar:
            model.train()
            # have to convert to tensor because reduce needed it
            average_loss = torch.tensor(0, dtype=torch.float).to(rank)
            for i, batch in enumerate(train_loader):
                x1, x2 = batch[0], batch[1]
                x1 = x1.cuda()
                x2 = x2.cuda()
                optimizer.zero_grad(set_to_none=True)
                if mixed_precision:
                    pass
                else:
                    preds = model(x1)
                    loss = loss_fn(x2,preds)
                    #print(loss)
                    loss = loss / acc_batches
                    loss.backward()
                    # debug_matrix(x1,'x1.mrc')
                    # debug_matrix(x2,'x2.mrc')
                    # debug_matrix(preds,'preds.mrc')
                loss_item = loss.item()
                              
                if ( (i+1)%acc_batches == 0 ) or (i+1) == min(len(train_loader), steps_per_epoch_train * acc_batches):
                    if mixed_precision:
                        pass
                    else:
                        optimizer.step()

                if rank == 0 and ( (i+1)%acc_batches == 0 ):
                   progress_bar.set_postfix({"Loss": loss_item})#, "t1": time2-time1, "t2": time3-time2, "t3": time4-time3})
                   progress_bar.update()
                average_loss += loss_item
                
                if i + 1 >= steps_per_epoch_train*acc_batches:
                    break
            average_loss = average_loss / (i+1.)
        
                                      
        dist.barrier()
        dist.reduce(average_loss, dst=0)

        average_loss =  average_loss / dist.get_world_size()
        if rank == 0:
            average_loss_list.append(average_loss.cpu().numpy())
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {average_loss:.4f}")
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'average_loss': average_loss_list,
                }, model_path)
    dist.destroy_process_group()

def ddp_predict(rank, world_size, port_number, model, data, tmp_data_path):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port_number
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = model.to(rank)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])
    model.eval()

    num_data_points = data.shape[0]
    steps_per_rank = (num_data_points + world_size - 1) // world_size

    output = torch.zeros(steps_per_rank,data.shape[1],data.shape[2],data.shape[3],data.shape[4]).to(rank)
    with torch.no_grad():
        for i in tqdm(range(rank * steps_per_rank, min((rank + 1) * steps_per_rank, num_data_points)),disable=(rank!=0)):
            batch_input  = data[i:i+1]
            batch_output  = model(batch_input.to(rank))
            output[i - rank * steps_per_rank] = batch_output
    gathered_outputs = [torch.zeros_like(output) for _ in range(world_size)]
    dist.all_gather(gathered_outputs, output)
    dist.barrier()
    if rank == 0:
        gathered_outputs = torch.cat(gathered_outputs).cpu().numpy()
        gathered_outputs = gathered_outputs[:data.shape[0]]
        np.save(tmp_data_path,gathered_outputs)
    dist.destroy_process_group()

class Net:
    def __init__(self, filter_base=64, unet_depth=4, add_last=False):
        torch.backends.cudnn.benchmark = True
        self.model = Unet(filter_base = filter_base,unet_depth=unet_depth, add_last=add_last)
        self.world_size = torch.cuda.device_count()
        self.port_number = str(find_unused_port())
        logging.info(f"Port number: {self.port_number}")
        self.metrics = {"average_loss":[],
                        "avg_val_loss":[] }

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.metrics["average_loss"] = checkpoint['average_loss']

    def load_jit(self, path):
        #Using the TorchScript format, you will be able to load the exported model and run inference without defining the model class.
        self.model = torch.jit.load(path)
    
    def save(self, path):
        state = self.model.state_dict()
        torch.save(state, path)

    def save_jit(self, path):
        model_scripted = torch.jit.script(self.model) # Export to TorchScript
        model_scripted.save(path) # Save

    def train(self, data_path, output_dir, batch_size=None, outmodel_path='tmp.pt',
              epochs = 10, steps_per_epoch=200, acc_batches =2,
              mixed_precision=False, learning_rate=3e-4, fsc3d = None):
        print('learning rate',learning_rate)

        self.model.zero_grad()

        #if os.path.exists(model_path):
        #    os.remove(model_path)
        try: 
            mp.spawn(ddp_train, args=(self.world_size, self.port_number, self.model,
                                       data_path, batch_size, acc_batches, epochs, steps_per_epoch, learning_rate, 
                                       mixed_precision, outmodel_path, fsc3d), nprocs=self.world_size)

        except KeyboardInterrupt:
           logging.info('KeyboardInterrupt: Terminating all processes...')
           dist.destroy_process_group() 
           os.system("kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')")

        checkpoint = torch.load(outmodel_path)
        self.metrics['average_loss'].extend(checkpoint['average_loss'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.save({
            'model_state_dict': checkpoint['model_state_dict'],
            'average_loss': self.metrics['average_loss'],
            }, outmodel_path)
        
    def predict_subtomos(self, mrc_list, result_folder):
        from IsoNet.utils.utils import read_mrc
        first_map, pixel_size = read_mrc(mrc_list[0])
        shape = first_map.shape
        data = np.zeros((len(mrc_list),shape[0], shape[1], shape[2]), dtype=np.float32)
        for i, file_name in enumerate(mrc_list):
            subtomo, pixel_size = read_mrc(file_name)
            data[i] = subtomo
        tmp_data_path = f"{result_folder}/tmp.npy"
        outData = self.predict(data, tmp_data_path=tmp_data_path)
        os.remove(tmp_data_path)
        return outData

    def predict(self, data, tmp_data_path):    
        data = data[:,np.newaxis,:,:]
        data = torch.from_numpy(data)
        print('data_shape',data.shape)
        mp.spawn(ddp_predict, args=(self.world_size, self.port_number, self.model, data, tmp_data_path), nprocs=self.world_size)
        outData = np.load(tmp_data_path)
        outData = outData.squeeze()
        return outData

    
    def predict_map(self, data, output_dir, cube_size = 64, crop_size=96, output_base=None, wedge=None):
        reform_ins = reform3D(data,cube_size,crop_size,7)
        data = reform_ins.pad_and_crop()
        print(data.shape)
        
        tmp_data_path = f"{output_dir}/tmp.npy"
        outData = self.predict(data, tmp_data_path=tmp_data_path)
        outData = outData.squeeze()
        outData=reform_ins.restore(outData)
        os.remove(tmp_data_path)
        return outData