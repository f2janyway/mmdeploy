#!/bin/bash

cd build

cmake -DCMAKE_CXX_COMPILER=g++ -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=/usr -DCUDNN_DIR=/usr/lib/aarch64-linux-gnu ..

make -j$(nproc) && make install

# trt_plugin_0 << 120,80,1
# trt_plugin_1 << 192,120,1
# trt_plugin_2 << 192,120,64?

cp /home/nano/code/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so ~/code/stereo_5_split/trt_plugin_1.so
cp /home/nano/code/mmdeploy/mmdeploy/lib/libmmdeploy_tensorrt_ops.so ~/code/sequor_vfm_deploy/trt_engine/trt_plugin_1.so


