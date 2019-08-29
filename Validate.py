
from __future__ import  print_function, division, absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.feeder_skeleton import DataFeeder_Skeleton
import time
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
from model.action_dnn_trian import MLP_TRN

print_freq = 5



# engine = context.get_engine()
# assert(engine.get_nb_bindings() == 2)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def inference(inputs, output, batch_size, context, stream, d_input, d_output, bindings):
    inputs = inputs.astype(np.float32)
    cuda.memcpy_htod_async(d_input, inputs, stream)
    # batch_size = 1 in default
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # print(output)
    stream.synchronize()


def validate_trt(val_loader, context, batch_size, criterion, is_initialized=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.numpy()

        if not is_initialized:
            output = np.empty((batch_size, 5), dtype=np.float32)
            d_input = cuda.mem_alloc(input.size * input.dtype.itemsize)
            d_output = cuda.mem_alloc(output.size * output.dtype.itemsize)
            bindings = [int(d_input), int(d_output)]
            stream = cuda.Stream()
            is_initialized = True

        # compute output
        inference(input, output, batch_size, context, stream, d_input, d_output, bindings)

        # measure elapsed time
        batch_time.update(time.time() - end)


        output_tensor = torch.from_numpy(output)
        input_tensor = torch.from_numpy(input)
        loss = criterion(output_tensor, target)
        # measure accuracy and record loss
        prec1, prec2 = accuracy(output_tensor.data, target, topk=(1, 2))
        losses.update(loss.data, input_tensor.size(0))
        top1.update(prec1, input_tensor.size(0))
        top2.update(prec2, input_tensor.size(0))



        if i % print_freq == 0:
            line = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top2=top2))
            print(line)

        end = time.time()

    line = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top2=top2, loss=losses))
    print(line)


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.reshape(input.shape[0],-1)
        input = input.cuda()

        # compute output
        output = model(input)

        # measure elapsed time
        batch_time.update(time.time() - end)

        loss = criterion(output, target)
        # measure accuracy and record loss
        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top2.update(prec2, input.size(0))

        if i % print_freq == 0:
            output = ('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top2=top2))
            print(output)

        end = time.time()

    line = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top2=top2, loss=losses))
    print(line)



def main():
    is_initialized = False

    data_path = './dataset/label_action_five.txt'
    label_path = './dataset/label_category.txt'
    root_path = './dataset/ContinuousFrame/'
    num_segments = 1
    batch_size = 10
    val_loader = DataLoader(
        DataFeeder_Skeleton(data_path, label_path, root_path, transform=None, data_type="Single",
                            num_segment=num_segments),
        batch_size=batch_size,
        shuffle=True)
    criterion = nn.CrossEntropyLoss()

    engine_path = './tensorrt/test.trt'
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as engine_file, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_file.read())
    context = engine.create_execution_context()
    validate_trt(val_loader, context, batch_size, criterion, is_initialized)

    model_path = './checkpoint/MLP_TRN/100_85.789474.pt'
    model = MLP_TRN(num_segments=1, class_nums=5)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    validate(val_loader, model, criterion)


if __name__ == '__main__':
    main()
