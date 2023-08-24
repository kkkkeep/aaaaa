import datetime
import logging
import os
import argparse

from math import ceil
from random import Random

import torch
import torch.distributed as dist
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 
import torch.utils.data
import torch.utils.data.distributed
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.autograd import Variable
from torch.nn.modules import Module
from torchvision import datasets, transforms

gbatch_size = 128
epochs = 10
world_size = os.environ.get("WORLD_SIZE", "{}")
rank = os.environ.get("RANK", "{}")

class DistributedDataParallel(Module):
  def __init__(self, module):
    super(DistributedDataParallel, self).__init__()
    self.module = module
    self.first_call = True

    def allreduce_params():
      if self.needs_reduction:
        self.needs_reduction = False 
        buckets = {}
        for param in self.module.parameters():
          if param.requires_grad and param.grad is not None:
            tp = type(param.data)
            if tp not in buckets:
              buckets[tp] = []
            buckets[tp].append(param)
        for tp in buckets:
          bucket = buckets[tp]
          grads = [param.grad.data for param in bucket]
          coalesced = _flatten_dense_tensors(grads)
          dist.all_reduce(coalesced)
          coalesced /= dist.get_world_size()
          
          for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)

    for param in list(self.module.parameters()):
      def allreduce_hook(*unused): 
        Variable._execution_engine.queue_callback(allreduce_params)  

      if param.requires_grad:
        param.register_hook(allreduce_hook)

  def weight_broadcast(self):
    for param in self.module.parameters():
      dist.broadcast(param.data, 0)

  def forward(self, *inputs, **kwargs):  
    if self.first_call:
      logging.info("first broadcast start")
      self.weight_broadcast()
      self.first_call = False
      logging.info("first broadcast done")
    self.needs_reduction = True  
    return self.module(*inputs, **kwargs)

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x): 
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x)


def partition_dataset(rank):
  dataset = datasets.MNIST(
    './data{}'.format(rank),
    train=True,
    download=True,
    transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ]))
  size = dist.get_world_size()
  bsz = int(gbatch_size / float(size))
  train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
  train_set = torch.utils.data.DataLoader(
    dataset, batch_size=bsz, shuffle=(train_sampler is None), sampler=train_sampler)
  return train_set, bsz


def average_gradients(model):
  size = float(dist.get_world_size())
  group = dist.new_group([0])
  for param in model.parameters():
    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=group)
    
    param.grad.data /= size


def run(gpu):
  rank = dist.get_rank()
  torch.manual_seed(1234)
  train_set, bsz = partition_dataset(rank)
  model = Net()
  if gpu:
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
  else:
    model = DistributedDataParallel(model)
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

  num_batches = ceil(len(train_set.dataset) / float(bsz))
  logging.info("num_batches = %s", num_batches)
  time_start = datetime.datetime.now()
  for epoch in range(epochs):
    epoch_loss = 0.0
    for data, target in train_set:
      if gpu:
        data, target = Variable(data).cuda(), Variable(target).cuda()
      else:
        data, target = Variable(data), Variable(target)
      optimizer.zero_grad()
      output = model(data)
      loss = F.nll_loss(output, target)
      epoch_loss += loss.item()
      loss.backward()
      average_gradients(model)
      optimizer.step()
    logging.info('Epoch {} Loss {:.6f} Global batch size {} on {} ranks'.format(
      epoch, epoch_loss / num_batches, gbatch_size, dist.get_world_size()))
  if gpu:
    logging.info("GPU training time= {}".format(  
      str(datetime.datetime.now() - time_start)))  
  else:
    logging.info("CPU training time= {}".format(  
      str(datetime.datetime.now() - time_start))) 


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO,
                      format=('%(levelname)s|%(asctime)s'
                              '|%(pathname)s|%(lineno)d| %(message)s'),
                      datefmt='%Y-%m-%dT%H:%M:%S',
                      )
  logging.getLogger().setLevel(logging.INFO)

  parser = argparse.ArgumentParser(description='Train Pytorch model using DDP')
  parser.add_argument('--gpu', action='store_true',
                      help='Use GPU and CUDA')
  parser.set_defaults(gpu=False)
  args = parser.parse_args()
  if args.gpu:
    logging.info("\n======= CUDA INFO =======")
    logging.info("CUDA Availability: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
      logging.info("CUDA Device Name: %s", torch.cuda.get_device_name(0))
      logging.info("CUDA Version: %s", torch.version.cuda)
    logging.info("=========================\n")
  dist.init_process_group(backend='gloo', init_method='env://',world_size=int(world_size),rank=int(rank))
  run(gpu=False)
  dist.destroy_process_group()