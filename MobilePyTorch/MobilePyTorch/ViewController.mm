//
//  ViewController.m
//  MobilePyTorch
//
//  Created by Kai Kang on 4/12/17.
//  Copyright © 2017 Kai Kang. All rights reserved.
//

#import "ViewController.h"
#include <Accelerate/Accelerate.h>

@interface ViewController ()

@end

@implementation ViewController


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

- (void)viewDidLoad {
    [super viewDidLoad];
    
    NSString *myNSString = @"pretrained_model";    
    BNNSNetBuilder *net = [[BNNSNetBuilder alloc] initWithInput:input andOutput:output andModelPath:"datapath"];
    // Do any additional setup after loading the view, typically from a nib.
    
    const size_t input_width = 28,
    input_height = 28;
    
    // input layer
    BNNSImageStackDescriptor i_desc = {
        .width = input_width,
        .height = input_height,
        .channels = 1,
        .row_stride = input_width,
        .image_stride = input_width * input_height,
        .data_type = BNNSDataTypeFloat32,
    };
    
    // First convolution layer
    size_t conv1_output_size = calcConvLayerOutputSize(input_width, 5, 0, 1);
    BNNSImageStackDescriptor conv1_output = {
        .width = conv1_output_size,
        .height = conv1_output_size,
        .channels = 10,
        .row_stride = conv1_output_size,
        .image_stride = conv1_output_size * conv1_output_size,
        .data_type = BNNSDataTypeFloat32,
    };
    BNNSConvolutionLayerParameters conv1_layer_params = {
        .k_width = 5,
        .k_height = 5,
        .in_channels = 1,
        .out_channels = 10,
        .x_stride = 1,
        .y_stride = 1,
        .x_padding = 0,
        .y_padding = 0,
        .activation = BNNSActivationFunctionRectifiedLinear,
    };
    BNNSFilterParameters conv1_filter_params = {};
    conv1_layer_params.weights.data = NULL; // TODO fill with values
    conv1_layer_params.bias.data = NULL; // TODO fill with values
    conv1_layer_params.weights.data_type = BNNSDataTypeFloat32;
    conv1_layer_params.bias.data_type = BNNSDataTypeFloat32;
    
    BNNSFilter conv1 = BNNSFilterCreateConvolutionLayer(&i_desc, &conv1_output, &conv1_layer_params, &conv1_filter_params);
    BNNSPoolingLayerParameters
    // max_pool2d

}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
