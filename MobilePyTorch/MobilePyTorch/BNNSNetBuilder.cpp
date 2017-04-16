#include "BNNSNetBuilder.h"

Conv2dLayerConfiguration::Conv2dLayerConfiguration(
    size_t kernel_width,
    size_t kernel_height,
    size_t stride_x,
    size_t stride_y,
    void *weights,
    void *biases,
    BNNSActivation activation,
    size_t padding_x, size_t padding_y
  ) {
  _kernel_width = kernel_width;
  _kernel_height = kernel_height;
  _stride_x = stride_x;
  _stride_y = stride_y;
  _weights = weights;
  _biases = biases;
  _activation = activation;
  _padding_x = padding_x;
  _padding_y = padding_y;
}

BNNSFilter Conv2dLayerConfiguration::build() {

}


BNNSNetBuilder(size_t width, size_t height, size_t channels) {
  _width = width;
  _height = height;
  _channels = channels;
  _type = BNNSDataTypeFloat16;
}

BNNSNetBuilder(size_t width, size_t height, size_t channels,
               BNNSDataType type) {
  _width = width;
  _height = height;
  _channels = channels;
  _type = type;
}

void *BNNSNetBuilder::load_data(string data_path) {

}

BNNSNetBuilder *BNNSNetBuilder::conv2d(
  size_t in_channels,
  size_t out_channels,
  size_t kernel_size,
  string weights_data_path,
  string bias_data_path,
  BNNSActivation activation,
  size_t stride,
  size_t padding
  ) {

  void *weights = load_data(weights_data_path);
  void *biases = load_data(bias_data_path);

  config = Conv2dLayerConfiguration(
    kernel_size, kernel_size,
    stride, stride,
    weights, biases,
    activation,
    padding, padding
    );
  _configurations.append(config);
  return this;
}

void *apply(void *in) {

}

int main() {
  BNNSNetBuilder *mnist_net = new BNNSNetBuilder(28, 28, 1);
  mnist_net
    ->conv2d(1, 32, 5,
             "conv1weights", "conv1biases",
             BNNSActivationFunctionRectifiedLinear)
    ->MaxPool2d();
  mnist_net->apply(input);
  return 0;
}
