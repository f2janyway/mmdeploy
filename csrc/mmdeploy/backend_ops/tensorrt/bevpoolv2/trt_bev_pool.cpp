// Copyright (c) OpenMMLab. All rights reserved.
#include "trt_bev_pool.hpp"

#include <assert.h>

#include <chrono>

#include "trt_bev_pool_kernel.hpp"
#include "trt_plugin_helper.hpp"
#include "trt_serialize.hpp"

namespace mmdeploy {
namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"bev_pool_v2"};
}  // namespace

TRTBEVPoolV2::TRTBEVPoolV2(const std::string &name, int outWidth, int outHeight,int outZ) :
      TRTPluginBase(name),
      mOutWidth(outWidth),
      mOutHeight(outHeight),
      mOutZ(outZ){}

TRTBEVPoolV2::TRTBEVPoolV2(const std::string name, const void *data, size_t length)
    : TRTPluginBase(name) {
  deserialize_value(&data, &length, &mOutWidth);
  deserialize_value(&data, &length, &mOutHeight);
  deserialize_value(&data, &length, &mOutZ);
}

nvinfer1::IPluginV2DynamicExt *TRTBEVPoolV2::clone() const TRT_NOEXCEPT {
  TRTBEVPoolV2 *plugin = new TRTBEVPoolV2(mLayerName, mOutWidth, mOutHeight, mOutZ);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs TRTBEVPoolV2::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) TRT_NOEXCEPT {
  // input[0] == depth
  // input[1] == feat
  // input[2] == ranks_depth
  // input[3] == ranks_feat
  // input[4] == ranks_bev
  nvinfer1::DimsExprs ret;
  // original
  // ret.nbDims = 4;
  // ret.d[0] = exprBuilder.constant(1); //Todo support batch>1
  // ret.d[1] = exprBuilder.constant(mOutHeight);
  // ret.d[2] = exprBuilder.constant(mOutWidth);
  // ret.d[3] = inputs[1].d[3];

  // for vfm case, standard from onnx bev_pool_v2 
  //N=1, Z=1, H=80, W=128, C=64
  //ret.nbDims = 5;
  //ret.d[0] = inputs[0].d[0]; // N:depth_probs
  //// ret.d[1] = exprBuilder.constant(1); // Z: single level
  //ret.d[1] = exprBuilder.constant(mOutZ);
  //ret.d[2] = exprBuilder.constant(mOutHeight); // H // 여기 값이 80이어야 함?
  //ret.d[3] = exprBuilder.constant(mOutWidth);  // W // 여기 값이 128이어야 함?
  //ret.d[4] = inputs[1].d[2]; // C: feat
                             //

  // 원하는 출력: (B, C, Z, H, W)
  //ret.nbDims = 5;
  //ret.d[0] = inputs[0].d[0];                 // B
  //ret.d[1] = inputs[1].d[4];                 // C  (feat: B N H W C)
  //ret.d[2] = exprBuilder.constant(mOutZ);    // Z
  //ret.d[3] = exprBuilder.constant(mOutHeight); // H
  //ret.d[4] = exprBuilder.constant(mOutWidth);  // W

  // ONNX의 /m/bev_pool_v2_output_0 기대: (B, Z, H, W, C)
  ret.nbDims = 5;
  ret.d[0] = inputs[0].d[0];                 // B
  ret.d[1] = exprBuilder.constant(mOutZ);    // Z
  ret.d[2] = exprBuilder.constant(mOutHeight); // H
  ret.d[3] = exprBuilder.constant(mOutWidth);  // W
  ret.d[4] = inputs[1].d[4];                 // C  (feat: B N H W C)

  std::cout << "[bev_pool_v2] getOutputDimensions called. "
          << "outZ=" << mOutZ << " outH=" << mOutHeight << " outW=" << mOutWidth << std::endl;


  std::cout << "[bev_pool_v2] out dims will be: "
          << "B=" << inputs[0].d[0]->getConstantValue()
          << " C=" << inputs[1].d[4]->getConstantValue()
          << " Z=" << mOutZ
          << " H=" << mOutHeight
          << " W=" << mOutWidth
          << std::endl;

  return ret;
}

//bool TRTBEVPoolV2::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc *ioDesc,
//                                               int nbInputs, int nbOutputs) TRT_NOEXCEPT {
//  // input[0] == depth->kFLOAT
//  // input[1] == feat->kFLOAT
//  // input[2] == ranks_depth->kINT32
//  // input[3] == ranks_feat->kINT32
//  // input[4] == ranks_bev->kINT32
//  // input[5] == interval_starts->kINT32
//  // input[6] == interval_lengths->kINT32
//  // output[0] == bev_feat->kFLOAT
//  if (pos == 0 || pos==1 || pos == 7) {
//    return (ioDesc[pos].type == nvinfer1::DataType::kFLOAT &&
//            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
//  } else {
//    return (ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
//            ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR);
//  }
//}


bool TRTBEVPoolV2::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* ioDesc,
    int nbInputs, int nbOutputs) TRT_NOEXCEPT
{
  // total = nbInputs + nbOutputs = 7 + 1 = 8
  // inputs: 0..6, output: 7

  const auto& in0 = ioDesc[0];

  // depth_probs (pos0): FP16 or FP32, LINEAR
  if (pos == 0) {
    return ( (ioDesc[pos].type == nvinfer1::DataType::kHALF ||
              ioDesc[pos].type == nvinfer1::DataType::kFLOAT) &&
             ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR );
  }

  // feat (pos1): must match input0 type/format
  if (pos == 1) {
    return ( ioDesc[pos].type   == in0.type &&
             ioDesc[pos].format == in0.format );
  }

  // ranks/interval (pos2..6): INT32, LINEAR
  if (pos >= 2 && pos <= 6) {
    return ( ioDesc[pos].type == nvinfer1::DataType::kINT32 &&
             ioDesc[pos].format == nvinfer1::TensorFormat::kLINEAR );
  }

  // output (pos7): must match input0 type/format
  if (pos == 7) {
    return ( ioDesc[pos].type   == in0.type &&
             ioDesc[pos].format == in0.format );
  }

  return false;
}


void TRTBEVPoolV2::configurePlugin(const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
                                     const nvinfer1::DynamicPluginTensorDesc *outputs,
                                     int nbOutputs) TRT_NOEXCEPT {
  // Validate input arguments
  ASSERT(nbInputs == 7);
  ASSERT(nbOutputs == 1);
  for (int i=0;i<nbInputs;i++){
    auto t = inputs[i].desc.type;
    auto f = inputs[i].desc.format;
    printf("[bev_pool_v2] input %d type=%d format=%d nbDims=%d\n",
           i, (int)t, (int)f, inputs[i].desc.dims.nbDims);
  }
  printf("[bev_pool_v2] output type=%d format=%d nbDims=%d\n",
         (int)outputs[0].desc.type, (int)outputs[0].desc.format, outputs[0].desc.dims.nbDims);
  auto d = inputs[1].desc.dims;
  printf("[bev_pool_v2] feat dims: nbDims=%d : ", d.nbDims);
  for (int k=0;k<d.nbDims;k++) printf("%d ", d.d[k]);
  printf("\n");

}

size_t TRTBEVPoolV2::getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
                                        const nvinfer1::PluginTensorDesc *outputs,
                                        int nbOutputs) const TRT_NOEXCEPT {
  return 0;
}

int TRTBEVPoolV2::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                            const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                            void *const *outputs, void *workSpace,
                            cudaStream_t stream) TRT_NOEXCEPT {
  nvinfer1::Dims feat_dims = inputDesc[1].dims; // bnhwc
  nvinfer1::Dims interval_dims = inputDesc[5].dims; // n
  nvinfer1::Dims out_dims = outputDesc[0].dims; //bhwc
  // int num_points = out_dims.d[0]*out_dims.d[1]*out_dims.d[2]*out_dims.d[3];
  auto data_type = inputDesc[0].type;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:{

      int numel = 1;
      for(int i=0; i<out_dims.nbDims; ++i){
        numel *= out_dims.d[i];
      }
      bev_pool_v2_set_zero(numel, (float *)outputs[0]);
  
      int C = feat_dims.d[feat_dims.nbDims -1];
  
      bev_pool_v2(C, interval_dims.d[0], (float *)inputs[0], (float *)inputs[1],
        (int *)inputs[2], (int *)inputs[3], (int *)inputs[4], (int *)inputs[5],(int *)inputs[6], (float *)outputs[0],
        stream);
      break;
    }
    
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType TRTBEVPoolV2::getOutputDataType(int index,
                                                     const nvinfer1::DataType *inputTypes,
                                                     int nbInputs) const TRT_NOEXCEPT {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *TRTBEVPoolV2::getPluginType() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTBEVPoolV2::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

int TRTBEVPoolV2::getNbOutputs() const TRT_NOEXCEPT { return 1; }

size_t TRTBEVPoolV2::getSerializationSize() const TRT_NOEXCEPT {
  return serialized_size(mOutWidth) + serialized_size(mOutHeight) + serialized_size(mOutZ);
}

void TRTBEVPoolV2::serialize(void *buffer) const TRT_NOEXCEPT {
  serialize_value(&buffer, mOutWidth);
  serialize_value(&buffer, mOutHeight);
  serialize_value(&buffer, mOutZ);
}

////////////////////// creator /////////////////////////////

TRTBEVPoolV2Creator::TRTBEVPoolV2Creator() {
  mPluginAttributes = std::vector<nvinfer1::PluginField>(
      {nvinfer1::PluginField("output_z"), nvinfer1::PluginField("output_height"), nvinfer1::PluginField("output_width")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *TRTBEVPoolV2Creator::getPluginName() const TRT_NOEXCEPT { return PLUGIN_NAME; }

const char *TRTBEVPoolV2Creator::getPluginVersion() const TRT_NOEXCEPT { return PLUGIN_VERSION; }

nvinfer1::IPluginV2 *TRTBEVPoolV2Creator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) TRT_NOEXCEPT {
  // int outWidth = 128;
  // int outHeight = 128;
  int outWidth = 128;
  int outHeight = 80;
  int outZ = 1;

  std::cout << "TRTBEVPoolV2Creator::createPlugin" << std::endl;
  for (int i = 0; i < fc->nbFields; i++) {
    std::cout << "field name: " << fc->fields[i].name << std::endl;
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name == "output_z") {
      outZ = static_cast<const int*>(fc->fields[i].data)[0];
    }
    if (field_name.compare("output_height") == 0) {
      outHeight = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("output_width") == 0) {
      outWidth = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }
  ASSERT(outHeight > 0);
  ASSERT(outWidth > 0);
  std::cout << "outZ: " << outZ
            << ", outHeight: " << outHeight
            << ", outWidth: " << outWidth << std::endl;

  TRTBEVPoolV2 *plugin = new TRTBEVPoolV2(name, outWidth, outHeight,outZ);
  std::cout << "after new plugin" << std::endl;
  std::cout << "plugin namespace: " << plugin->getPluginNamespace() << std::endl;
  std::cout << "creator namespace: " << getPluginNamespace() << std::endl;
  plugin->setPluginNamespace(getPluginNamespace());
  std::cout << "end" << std::endl;
  return plugin;
}

nvinfer1::IPluginV2 *TRTBEVPoolV2Creator::deserializePlugin(const char *name,
                                                              const void *serialData,
                                                              size_t serialLength) TRT_NOEXCEPT {
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new TRTBEVPoolV2(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

REGISTER_TENSORRT_PLUGIN(TRTBEVPoolV2Creator);
}  // namespace mmdeploy
