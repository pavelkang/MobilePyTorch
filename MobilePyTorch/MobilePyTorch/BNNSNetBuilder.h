#ifndef BNNSNETBUILDER_H
#define BNNSNETBUILDER_H

#include <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <vector>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;


typedef struct BNNSShape {
    size_t width;
    size_t height;
    size_t channels;
    bool operator== (const BNNSShape &other) {
        return (width == other.width && height == other.height
                && channels == other.channels);
    }
} BNNSShape;

enum Activ_Enum { relu, none };

class LayerConfiguration
{
public:
  LayerConfiguration(
    BNNSShape input_shape,
    BNNSShape output_shape,
    BNNSDataType data_type
    ) {
      _inputShape = input_shape;
      _outputShape = output_shape;
      _data_type = data_type;
  };
    // For Linear layers
    LayerConfiguration(
        size_t in_features, size_t out_features, BNNSDataType data_type
                       ) {
        BNNSShape input_shape = {
            .width = in_features,
            .height = 1,
            .channels = 1,
        };
        BNNSShape output_shape = {
            .width = out_features,
            .height = 1,
            .channels = 1,
        };
        _inputShape = input_shape;
        _outputShape = output_shape;
        _data_type = data_type;
    };
  virtual BNNSFilter build() {return NULL;};
    virtual string toStr() {return "...";};
  BNNSShape _outputShape;
  size_t _out_img_stride;

protected:
  BNNSShape _inputShape;
  BNNSDataType _data_type;
};

class LinearLayerConfiguration : public LayerConfiguration
{
public:
    LinearLayerConfiguration(size_t in_features, size_t out_features, BNNSDataType data_type,
                             Activ_Enum activation,
                             void *weights, void *biases): LayerConfiguration(in_features, out_features, data_type){
        _activation = activation;
        _weights = weights;
        _biases = biases;
    }
    BNNSFilter build() override;
    string toStr() override;
private:
    Activ_Enum _activation;
    void *_weights;
    void *_biases;
};

class MaxPool2dLayerConfiguration : public LayerConfiguration
{
public:
    MaxPool2dLayerConfiguration(BNNSShape input_shape,
                                BNNSShape output_shape,
                                BNNSDataType data_type,
                                size_t kernel_size,
                                size_t stride) : LayerConfiguration(input_shape, output_shape, data_type) {
        _kernel_size = kernel_size;
        _stride = stride;
    };
    BNNSFilter build() override;
    string toStr() override;
private:
    size_t _kernel_size;
    size_t _stride;
};

class Conv2dLayerConfiguration : public LayerConfiguration
{
public:
Conv2dLayerConfiguration(
    BNNSShape input_shape,
    BNNSShape output_shape,
    BNNSDataType data_type,
    size_t kernel_width,
    size_t kernel_height,
    size_t stride_x,
    size_t stride_y,
    void *weights,
    void *biases,
    BNNSActivationFunction activation,
    size_t padding_x,
    size_t padding_y
  ) : LayerConfiguration(input_shape, output_shape, data_type) {
    _kernel_width = kernel_width;
    _kernel_height = kernel_height;
    _stride_x = stride_x;
    _stride_y = stride_y;
    _weights = weights;
    _biases = biases;
    _activation = activation;
    _padding_x = padding_x;
    _padding_y = padding_y;
  };
    BNNSFilter build() override;
    string toStr() override;
private:
  size_t _kernel_width, _kernel_height, _stride_x, _stride_y,
    _padding_x, _padding_y;
  void *_weights;
  void *_biases;
  BNNSActivationFunction _activation;
};

class BNNSNetBuilder
{
public:
  BNNSNetBuilder(size_t width, size_t height, size_t channels);
  BNNSNetBuilder(size_t width, size_t height, size_t channels,
                 BNNSDataType datatype);
  BNNSNetBuilder *conv2d(
    size_t in_channels,
    size_t out_channels,
    size_t kernel_size,
    string weights_data_path,
    string bias_data_path,
    BNNSActivationFunction activation,
    size_t stride = 1,
    size_t padding = 0
    );
  BNNSNetBuilder *MaxPool2d(
    size_t kernel_size,
    size_t stride = 2);
  BNNSNetBuilder *Linear(
    size_t in_features,
    size_t out_features,
    string weights_data_path,
    string bias_data_path,
    Activ_Enum activation
    );
  void build();
  void *apply(void *in);
    void inspect();
private:
  vector<BNNSFilter> _filters;
  vector<LayerConfiguration *> _configurations;
  void *load_data(string data_path);
  BNNSDataType _type;
  BNNSShape _finalShape;
};

#endif
