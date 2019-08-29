from __future__ import print_function, division, absolute_import

import os
import sys
sys.path.append(os.getcwd())

import torch

import tensorrt as trt
from model.action_dnn_trian import MLP_TRN


def torch_to_onnx(torch_path, onnx_path):
    model = MLP_TRN(num_segments=1, class_nums=5)
    model.load_state_dict(torch.load(torch_path))
    model.cuda()

    dummpy_input = torch.randn((10, 32), device='cuda')
    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(model, (dummpy_input), onnx_path, verbose=True, input_names=input_names, output_names=output_names, export_params=True)


def build_engine_onnx(onnx_path, engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 100
        builder.max_workspace_size = 1 << 30

        print('[INFO] Start parsing ONNX file from path {}...'.format(onnx_path))
        with open(onnx_path, 'rb') as onnx_file:
            parser.parse(onnx_file.read())
        print('[INFO] End parsing ONNX file')

        print(network.num_layers)
        for i in range(network.num_layers):
            print(network.get_layer(i).name)
            # print("input shape:", network.get_layer(i).get_input(0).shape,
            #       "input name:", network.get_layer(i).get_input(0).name)
            # print("output shape:", network.get_layer(i).get_output(0).shape,
                  # "output name:", network.get_layer(i).get_output(0).name)

        print("input shape:", network.get_layer(2).get_input(0).shape,
              "input name:", network.get_layer(2).get_input(0).name)
        print("output shape:", network.get_layer(network.num_layers - 1).get_output(0).shape,
              "output name:", network.get_layer(network.num_layers - 1).get_output(0).name)

        print('[INFO] Start building an engine from file {}; this may take a while...'.format(onnx_path))
        engine = builder.build_cuda_engine(network)
        print('[INFO] End building Engine')

        # trt.utils.write_engine_to_file(engine_path, engine.serialize())
        with open(engine_path, "wb") as engine_file:
            engine_file.write(engine.serialize())

        return engine


def main():
    torch_path = './checkpoint/MLP_TRN/100_85.789474.pt'
    onnx_path = './tensorrt/test.onnx'
    # torch_to_onnx(torch_path, onnx_path)

    engine_path = './tensorrt/test.trt'
    build_engine_onnx(onnx_path, engine_path)


if __name__ == '__main__':
    main()
