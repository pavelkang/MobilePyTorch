#include "BNNSNetBuilder.h"
#include <iostream>

using namespace std;

/*
    These functions help print information about each layer
 */

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

static size_t calcPoolingLayerOutputSize(size_t W, size_t F, size_t S, size_t P) {
    if (W % 2 == 0) {
        return W / 2;
    } else {
        return (W + 1) / 2;
    }
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

void print_desc(BNNSImageStackDescriptor desc) {
    cout << desc.width << ", " << desc.height << ", " << desc.channels << ", " << desc.row_stride << ", " << desc.image_stride << endl;
}

BNNSFilter MaxPool2dLayerConfiguration::build() {
    
    if ((_inputShape.width % 2 != 0) || (_inputShape.height % 2 != 0)) {
        cout << "BNNS MaxPool2d layer is not compatible with tensorflow pooling layer. Tensorflow pads only what is necessary, but BNNS pads equally on both left and right. Your max pool layer is of odd dimension. Use at your own risk!" << endl;
    }
    
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
        .x_padding = _inputShape.width % 2 == 0 ? 0 : _padding,
        .y_padding = _inputShape.height % 2 == 0 ? 0 : _padding,
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
      .weights.data = _weights,
      .bias.data = _biases,
      .weights.data_type = _data_type,
      .bias.data_type = _data_type,
  };
  BNNSFilterParameters filter_params = {};
  _out_img_stride = o_desc.image_stride;
    assert(_weights != NULL);
    assert(_biases != NULL);
    BNNSFilter filter = BNNSFilterCreateConvolutionLayer(&i_desc, &o_desc,
                                                        &layer_params,
                                                        &filter_params);
    return filter;
}

BNNSNetBuilder::BNNSNetBuilder(size_t width, size_t height, size_t channels) {
  _finalShape = {
    .width  = width,
    .height = height,
    .channels = channels,
  };
  _type = BNNSDataTypeFloat32;
    _softmax = false;
}

BNNSNetBuilder::BNNSNetBuilder(size_t width, size_t height, size_t channels,
               BNNSDataType type) {
  _finalShape = {
    .width  = width,
    .height = height,
    .channels = channels,
  };
  _type = type;
    _softmax = false;
}

void *BNNSNetBuilder::load_data(string data_path, size_t expected_size) {
    NSString* path = [NSString stringWithUTF8String:data_path.c_str()];
    NSString *filepath = [[NSBundle mainBundle] pathForResource:path ofType:@""];
    NSData *weights = [NSData dataWithContentsOfFile:filepath];
    
    if (weights.length / sizeof(Float32) != expected_size) {
        cout << "expected size: " << expected_size << ", but got " << weights.length / sizeof(Float32) << endl;
        return NULL;
    }
    Float32 *ws = (Float32 *)calloc(weights.length / 4, sizeof(float));
    memcpy(ws, weights.bytes, weights.length);
    return (void *)ws;
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
                                            kernel_size, stride, kernel_size / 2),
        .height = calcPoolingLayerOutputSize(inputShape.height, kernel_size, stride, kernel_size / 2),
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
    void *weights = load_data(weights_data_path, in_features * out_features);
    void *biases = load_data(bias_data_path, out_features);
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

BNNSNetBuilder *BNNSNetBuilder::softmax() {
    _softmax = true;
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

    size_t expected_weight_size = in_channels * kernel_size * kernel_size * out_channels;
    
  void *weights = load_data(weights_data_path, expected_weight_size);
  void *biases = load_data(bias_data_path, out_channels);

  BNNSShape inputShape = {
    .width = _finalShape.width,
    .height= _finalShape.height,
    .channels = _finalShape.channels,
  };

  BNNSShape outputShape = {
    .width = calcConvLayerOutputSize(inputShape.width,
                                     kernel_size, kernel_size/2, stride),
    .height = calcConvLayerOutputSize(inputShape.height,
                                      kernel_size, kernel_size/2, stride),
    .channels = out_channels,
  };

  Conv2dLayerConfiguration *config = new Conv2dLayerConfiguration(
    inputShape, outputShape, _type,
    kernel_size, kernel_size,
    stride, stride,
    weights, biases,
    activation,
    kernel_size/2, kernel_size/2
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
    assert (_configurations.size() == _filters.size());
    void *out = in;
    
    for (int i = 0; i < _filters.size(); ++i) {
        in = out;
        LayerConfiguration *config = _configurations[i];
        out = (void *)calloc(config->_out_img_stride * config->_outputShape.channels,
                             sizeof(_type));
        Float32 *inf = (Float32 *) in;
        int res = BNNSFilterApply(_filters[i], in, out);
        if (res != 0) {
            cout << "Failure!" << endl;
        }
        Float32 *outf = (Float32 *) out;
    }
    if (!_softmax) {
        return out;
    }
    std::cout << "Calculating softmax" << std::endl;
    Float32 *outf = (Float32 *)out;
    
    // Calculate Softmax
    // find max
    Float32 maxval = outf[0];
    for (int i = 1; i < _finalShape.width; ++i) {
        if (outf[i] > maxval) {
            maxval = outf[i];
        }
    }
    // subtract
    for (int i = 0; i < _finalShape.width; ++i) {
        outf[i] = outf[i] - maxval;
    }
    // sum
    Float32 denom = 0.0;
    for (int i = 0; i < _finalShape.width; ++i) {
        denom += exp(outf[i]);
    }
    assert(denom != 0);
    for (int i = 0; i < _finalShape.width; ++i) {
        outf[i] = exp(outf[i]) / denom;
    }
    return (void *)outf;
}

void BNNSNetBuilder::inspect() {
    for (int i = 0; i < _configurations.size(); ++i) {
        LayerConfiguration *config = _configurations[i];
        cout << config->toStr() << endl;
        BNNSFilter filter = _filters[i];
        assert(filter != NULL);
    }
    assert(_configurations[_configurations.size() - 1]->_out_img_stride == _finalShape.width * _finalShape.height);
    if (_softmax) {
        std::cout << "softmax vector of length " << _configurations[_configurations.size() - 1]->_out_img_stride << std::endl;
    }
}
