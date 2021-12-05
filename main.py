
import argparse

from tqdm import tqdm
import numpy as np

import torch as T
import torchvision as TV
import torch.distributed as DIST

from model import Model

class Dist:
    def __init__(self):
        super().__init__()
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--local_rank", type=int)
        args = parser.parse_args()
        self.rank_local = args.local_rank
        
        T.cuda.set_device(self.rank_local)
        DIST.init_process_group(backend='nccl')
        DIST.barrier()

def tqdm_iter(it):
    if DIST.get_rank()==0:
        return tqdm(it, ascii=True)
    else:
        return it        

def reduce_val(v):
    v = T.tensor(v).cuda()
    DIST.all_reduce(v)
    v = v.item()/DIST.get_world_size()
    return v

if __name__=='__main__':
    dist = Dist()
    
    if DIST.get_rank()==0:
        TV.datasets.CIFAR10('_data/', download=True)
    DIST.barrier()
    
    ds_tr, ds_ts = [TV.datasets.CIFAR10('_data/', transform=TV.transforms.ToTensor(), train=typ) \
                    for typ in [True, False]]
    sp_tr, sp_ts = [T.utils.data.distributed.DistributedSampler(ds, shuffle=(ds is ds_tr)) \
                    for ds in [ds_tr, ds_ts]]
    dl_tr, dl_ts = [T.utils.data.DataLoader(ds, batch_size=64, num_workers=8, sampler=sp, pin_memory=True) \
                    for ds, sp in [[ds_tr, sp_tr], [ds_ts, sp_ts]]]
    
    if DIST.get_rank()==0:
        print('Tr:', len(dl_tr.dataset), 'Ts:', len(dl_ts.dataset))
    
    model = T.nn.parallel.DistributedDataParallel(Model().cuda(), 
                                                  device_ids=[dist.rank_local], 
                                                  output_device=dist.rank_local)
    loss_func = T.nn.CrossEntropyLoss().cuda()
    optzr = T.optim.Adam(model.parameters(), lr=0.003)
    
    tq = tqdm_iter(range(20))
    for e in tq:
        model.train()
        sp_tr.set_epoch(e)
        for img, lbl in tqdm_iter(dl_tr):
            optzr.zero_grad()
            out = model(img.cuda())
            ls = loss_func(out, lbl.cuda())
            ls.backward()
            optzr.step()
            
            if DIST.get_rank()==0:
                ls, ac = ls.item(), (T.argmax(out, dim=1)==lbl.cuda()).float().mean().item()
                tq.set_postfix(ls='%.4f'%(ls), ac='%.4f'%(ac))
        
        model.eval()
        ep = {'loss': [], 'acc': []}
        for img, lbl in tqdm_iter(dl_ts):
            out = model(img.cuda())
            ls = loss_func(out, lbl.cuda())
            
            ls, ac = ls.item(), (T.argmax(out, dim=1)==lbl.cuda()).float().mean().item()
            ls, ac = reduce_val(ls), reduce_val(ac)
            ep['loss'].append(ls), ep['acc'].append(ac)
        
        if DIST.get_rank()==0:
            ep = {k: float(np.average(ep[k])) for k in ep}
            print('Ep %d: loss=%.4f, acc=%.4f'%(e+1, ep['loss'], ep['acc']))
            T.save(model.module.state_dict(), '_snapshot/model_%d.pt'%(e+1))
        
        DIST.barrier()
        
        for pg in optzr.param_groups:
            pg['lr'] *= 0.8
            