#!/bin/bash

cd build

cmake -DCMAKE_CXX_COMPILER=g++ -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=/usr -DCUDNN_DIR=/usr/lib/aarch64-linux-gnu ..

make -j$(nproc) && make install


cp lib/libmmdeploy_tensorrt_ops.so ~/code/stereo_5_split
cp lib/libmmdeploy_tensorrt_ops.so ~/code/sequor_vfm_deploy/trt_engine/


