#include "BNNSNetBuilder.h"
#include <iostream>

using namespace std;

string shape2str(BNNSShape shape) {
    return "<" + to_string(shape.width) + ", " + to_string(shape.height) + ", " + to_string(shape.channels) + ">";
}

string MaxPool2dLayerConfiguration::toStr() {
    return "MaxPool2d: kernel: " + to_string(_kernel_size) + ", stride: " + to_string(_stride) + " \n  IN: " + shape2str(_inputShape) + "\n  OUT: " + shape2str(_outputShape);
}

string Conv2dLayerConfiguration::toStr() {
    string configStr = "kwidth: " + to_string(_kernel_width) + ", kheight: " + to_string(_kernel_height) +
    ", stridex: " + to_string(_stride_x) + ", stridey: " + to_string(_stride_y) + ", padx: " + to_string(_padding_x) + ", pady: " + to_string(_padding_y);
    return "Conv2d: " + configStr + "\n  IN: " + shape2str(_inputShape) + "\n  OUT: " + shape2str(_outputShape);
}

string LinearLayerConfiguration::toStr() {
    return "Linear: \n  IN: " + shape2str(_inputShape) + "\n  OUT: " + shape2str(_outputShape);
}

BNNSFilter LinearLayerConfiguration::build() {
    BNNSVectorDescriptor i_desc = {
        .size = _inputShape.width,
        .data_type = _data_type,
        .data_scale = 0,
        .data_bias = 0,
    };
    BNNSVectorDescriptor o_desc = {
        .size = _outputShape.width,
        .data_type = _data_type,
        .data_scale = 0,
        .data_bias = 0,
    };
    
    BNNSFullyConnectedLayerParameters layer_params = {
        .in_size = i_desc.size,
        .out_size = o_desc.size,
        .weights.data = _weights,
        .weights.data_type = _data_type,
        .bias.data_type = _data_type,
        .bias.data = _biases,
    };
    
    if (_activation == relu) {
        BNNSActivation activation = {
            .function = BNNSActivationFunctionRectifiedLinear,
            .alpha = 0,
            .beta = 0,
        };
        layer_params.activation = activation;
    }
    BNNSFilterParameters filter_params = {};
    _out_img_stride = o_desc.size;
    return BNNSFilterCreateFullyConnectedLayer(&i_desc, &o_desc, &layer_params, &filter_params);
}

BNNSFilter MaxPool2dLayerConfiguration::build() {
    BNNSImageStackDescriptor i_desc = {
        .width = _inputShape.width,
        .height = _inputShape.height,
        .channels = _inputShape.channels,
        .row_stride = _inputShape.width,
        .image_stride = _inputShape.width * _inputShape.height,
        .data_type = _data_type,
    };
    BNNSImageStackDescriptor o_desc = {
        .width = _outputShape.width,
        .height = _outputShape.height,
        .channels = _outputShape.channels,
        .row_stride = _outputShape.width,
        .image_stride = _outputShape.width * _outputShape.height,
        .data_type = _data_type,
    };
    BNNSPoolingLayerParameters layer_params = {
        .x_stride = _stride,
        .y_stride = _stride,
        .x_padding = 0,
        .y_padding = 0,
        .k_width = _kernel_size,
        .k_height = _kernel_size,
        .in_channels = i_desc.channels,
        .out_channels = o_desc.channels,
        .pooling_function = BNNSPoolingFunctionMax,
    };
    BNNSFilterParameters filter_params = {};
    _out_img_stride = o_desc.image_stride;
    return BNNSFilterCreatePoolingLayer(&i_desc, &o_desc, &layer_params, &filter_params);
}

BNNSFilter Conv2dLayerConfiguration::build() {
  BNNSImageStackDescriptor i_desc = {
    .width = _inputShape.width,
    .height = _inputShape.height,
    .channels = _inputShape.channels,
    .row_stride = _inputShape.width + _padding_x,
    .image_stride = (_inputShape.width + _padding_x)
    * (_inputShape.height + _padding_y),
    .data_type = _data_type,
  };
  BNNSImageStackDescriptor o_desc = {
    .width = _outputShape.width,
    .height = _outputShape.height,
    .channels = _outputShape.channels,
    .row_stride = _outputShape.width,
    .image_stride = _outputShape.width * _outputShape.height,
    .data_type = _data_type,
  };
  BNNSConvolutionLayerParameters layer_params = {
    .k_width = _kernel_width,
    .k_height = _kernel_height,
    .in_channels = _inputShape.channels,
    .out_channels = _outputShape.channels,
    .x_stride = _stride_x,
    .y_stride = _stride_y,
    .x_padding = _padding_x,
    .y_padding = _padding_y,
    .activation = _activation,
  };
  BNNSFilterParameters filter_params = {};
  _out_img_stride = o_desc.image_stride;
    assert(_weights != NULL);
    assert(_biases != NULL);
  layer_params.weights.data = _weights;
  layer_params.bias.data = _biases;
  layer_params.weights.data_type = _data_type;
  layer_params.bias.data_type = _data_type;
  return BNNSFilterCreateConvolutionLayer(&i_desc, &o_desc,
                                          &layer_params,
                                          &filter_params);
}

BNNSNetBuilder::BNNSNetBuilder(size_t width, size_t height, size_t channels) {
  _finalShape = {
    .width  = width,
    .height = height,
    .channels = channels,
  };
  _type = BNNSDataTypeFloat32;
}

BNNSNetBuilder::BNNSNetBuilder(size_t width, size_t height, size_t channels,
               BNNSDataType type) {
  _finalShape = {
    .width  = width,
    .height = height,
    .channels = channels,
  };
  _type = type;
}

void *BNNSNetBuilder::load_data(string data_path) {
    NSString* path = [NSString stringWithUTF8String:data_path.c_str()];
    NSString *filepath = [[NSBundle mainBundle] pathForResource:path ofType:@""];
    NSData *weights = [NSData dataWithContentsOfFile:filepath];
    Float32 *ws = (Float32 *)calloc(weights.length / 4, sizeof(float));
    memcpy(ws, weights.bytes, weights.length / 4);
    return (void *)ws;
}

/**
 @brief Calculates the output dimension of a convolution layer
 The formula comes from http://cs231n.github.io/convolutional-networks/
 
 @param W input size
 @param F kernel size
 @param P amount of zero padding
 @param S stride
 @return output size of a conv layer
 */
static size_t calcConvLayerOutputSize(size_t W, size_t F, size_t P, size_t S) {
    size_t _numerator = W - F + 2 * P;
    assert(_numerator % S == 0);
    return _numerator / S + 1;
}

static size_t calcPoolingLayerOutputSize(size_t W, size_t F, size_t S) {
    return (size_t) (floor( (W - F) / S ) + 1);
}

BNNSNetBuilder *BNNSNetBuilder::MaxPool2d(
    size_t kernel_size,
    size_t stride) {
    BNNSShape inputShape = {
        .width = _finalShape.width,
        .height = _finalShape.height,
        .channels = _finalShape.channels,
    };
    
    BNNSShape outputShape = {
        .width = calcPoolingLayerOutputSize(inputShape.width,
                                            kernel_size, stride),
        .height = calcPoolingLayerOutputSize(inputShape.height, kernel_size, stride),
        .channels = _finalShape.channels,
    };
    MaxPool2dLayerConfiguration *config = new MaxPool2dLayerConfiguration(inputShape, outputShape, _type,
                                    kernel_size, stride);
    _configurations.push_back(config);
    _finalShape = outputShape;
    return this;
}

BNNSNetBuilder *BNNSNetBuilder::Linear(
                       size_t in_features,
                       size_t out_features,
                       string weights_data_path,
                       string bias_data_path,
                       Activ_Enum activation
                                       ) {
    void *weights = load_data(weights_data_path);
    void *biases = load_data(bias_data_path);
    LinearLayerConfiguration *config = new LinearLayerConfiguration(in_features, out_features, _type,
                                                                    activation, weights, biases);
    _configurations.push_back(config);
    BNNSShape outputShape = {
        .width = out_features,
        .height = 1,
        .channels = 1,
    };
    _finalShape = outputShape;
    return this;
}

BNNSNetBuilder *BNNSNetBuilder::conv2d(
  size_t in_channels,
  size_t out_channels,
  size_t kernel_size,
  string weights_data_path,
  string bias_data_path,
  BNNSActivationFunction activation,
  size_t stride,
  size_t padding
  ) {

  assert(_finalShape.channels == in_channels);

  void *weights = load_data(weights_data_path);
  void *biases = load_data(bias_data_path);

  BNNSShape inputShape = {
    .width = _finalShape.width,
    .height= _finalShape.height,
    .channels = _finalShape.channels,
  };

  BNNSShape outputShape = {
    .width = calcConvLayerOutputSize(inputShape.width,
                                     kernel_size, padding, stride),
    .height = calcConvLayerOutputSize(inputShape.height,
                                      kernel_size, padding, stride),
    .channels = out_channels,
  };

  Conv2dLayerConfiguration *config = new Conv2dLayerConfiguration(
    inputShape, outputShape, _type,
    kernel_size, kernel_size,
    stride, stride,
    weights, biases,
    activation,
    padding, padding
    );
  _configurations.push_back(config);
  _finalShape = outputShape;
  return this;
}

void BNNSNetBuilder::build() {
  for (int i = 0; i < _configurations.size(); ++i) {
    _filters.push_back(_configurations[i]->build());
  }
}

void *BNNSNetBuilder::apply(void *in) {
  void *out = in;
  for (int i = 0; i < _configurations.size(); ++i) {
    in = out;
    LayerConfiguration *config = _configurations[i];
    out = (void *)calloc(config->_out_img_stride * config->_outputShape.channels,
                         sizeof(_type));
    BNNSFilterApply(_filters[i], in, out);
  }
  return out;
}

void BNNSNetBuilder::inspect() {
    for (int i = 0; i < _configurations.size(); ++i) {
        LayerConfiguration *config = _configurations[i];
        cout << config->toStr() << endl;
        
        BNNSFilter filter = _filters[i];
        cout << filter << endl;
    }
}
