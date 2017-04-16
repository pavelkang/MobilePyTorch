#ifndef BNNSNETBUILDER_H
#define BNNSNETBUILDER_H

#include <Accelerate/Accelerate.h>
#include <iostream>
#include <vector>

using namespace std;

struct BNNSShape {
  size_t width;
  size_t height;
  size_t channels;
}

class LayerConfiguration
{
public:
  BNNSFilter build();

private:
  BNNSShape _inputShape;
  BNNSShape _outputShape;
}

class Conv2dLayerConfiguration
{
public:
  Conv2dLayerConfiguration(
    size_t kernel_width,
    size_t kernel_height,
    size_t stride_x,
    size_t stride_y,
    void *weights,
    void *biases,
    BNNSActivation activation,
    size_t padding_x, size_t padding_y
    );
private:
  size_t _kernel_width, _kernel_height, _stride_x, _stride_y,
    padding_x, padding_y;
  void *_weights;
  void *_biases;
  BNNSActivation _activation;
}

class BNNSNetBuilder
{
public:
  BNNSNetBuilder(size_t width, size_t height, size_t channels);
  BNNSNetBuilder(size_t width, size_t height, size_t channels,
                 BNNSDataType datatype);
  void *apply(void *in);
  BNNSNetBuilder *conv2d(
    size_t in_channels,
    size_t out_channels,
    size_t kernel_size,
    string weights_data_path,
    string bias_data_path,
    BNNSActivation activation,
    size_t stride = 1,
    size_t padding = 0
    );
  BNNSNetBuilder *MaxPool2d(
    size_t kernel_size,
    BNNSActivation activation,
    size_t stride = 0,
    size_t padding = 0,
    );
  BNNSNetBuilder *Linear(
    size_t in_features,
    size_t out_features,
    string weights_data_path,
    string bias_data_path,
    BNNSActivation activation
    );
private:
  vector<LayerConfiguration> _configurations;
  vector<BNNSFilter> _filters;
  void *load_data(string data_path);
  size_t _width, _height, channels;
  BNNSDataType _type;
};


#endif
