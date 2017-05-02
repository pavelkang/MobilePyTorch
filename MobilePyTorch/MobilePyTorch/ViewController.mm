//
//  ViewController.m
//  MobilePyTorch
//
//  Created by Kai Kang on 4/12/17.
//  Copyright © 2017 Kai Kang. All rights reserved.
//

#import "ViewController.h"
#include "BNNSNetBuilder.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
@interface ViewController ()

@end

using namespace std;

@implementation ViewController

void *load_data(string data_path) {
    NSString* path = [NSString stringWithUTF8String:data_path.c_str()];
    NSString *filepath = [[NSBundle mainBundle] pathForResource:path ofType:@""];
    NSData *weights = [NSData dataWithContentsOfFile:filepath];
    
    cout << "len: " << weights.length / 4 << endl;
    Float32 *ws = (Float32 *)calloc(weights.length / 4, sizeof(float));
    
    memcpy(ws, weights.bytes, weights.length);
    cout << data_path << ": " << ws << endl;
    for (int i = 0; i < 3; i++) {
        cout << ws[i] << endl;
    }
    cout << "---data---" << endl;
    return (void *)ws;
}

size_t eye_predict() {
    
    Float32 *input = (Float32 *) calloc(48*56, sizeof(Float32));
    cout << "in: " << endl;
    for (int i = 0; i < 48*56; i++) {
        input[i] = 1.0;
    }
    
    Float32 *out;
    // First Conv Layer
    BNNSImageStackDescriptor i_desc_conv1 = {
        .width = 56,
        .height = 48,
        .channels = 1,
        .row_stride = 56,
        .image_stride = 56 * 48,
        .data_type = BNNSDataTypeFloat32,
    };
    BNNSImageStackDescriptor o_desc_conv1 = {
        .width = 56,
        .height = 48,
        .channels = 24,
        .row_stride = 56,
        .image_stride = 56*48,
        .data_type = BNNSDataTypeFloat32,
    };
    BNNSConvolutionLayerParameters layer_params_conv1 = {
        .k_width = 7,
        .k_height = 7,
        .in_channels = 1,
        .out_channels = 24,
        .x_stride = 1,
        .y_stride = 1,
        .x_padding = 3,
        .y_padding = 3,
        .activation = BNNSActivationFunctionRectifiedLinear,
        .weights.data = load_data("model-W_conv1-7x7x1x24"),
        .bias.data = load_data("model-b_conv1-24"),
        .weights.data_type = BNNSDataTypeFloat32,
        .bias.data_type = BNNSDataTypeFloat32,
    };
    BNNSFilterParameters filter_params_conv1 = {};
    BNNSFilter conv1 = BNNSFilterCreateConvolutionLayer(&i_desc_conv1, &o_desc_conv1,
                                                         &layer_params_conv1,
                                                         &filter_params_conv1);
    
    out = (Float32 *)calloc(56*48*24, sizeof(Float32));
    BNNSFilterApply(conv1, (void *)input, out);
    cout << "out: " << endl;
    for (int i = 0; i < 5; ++i) {
        cout << out[i] << endl;
    }
    
    // Max Pool 2d
    BNNSImageStackDescriptor i_desc_pool1 = {
        .width = 56,
        .height = 48,
        .channels = 24,
        .row_stride = 56,
        .image_stride = 56*48,
        .data_type = BNNSDataTypeFloat32,
    };
    
    BNNSImageStackDescriptor o_desc_pool1 = {
        .width = 28,
        .height = 24,
        .channels = 24,
        .row_stride = 28,
        .image_stride = 28*24,
        .data_type = BNNSDataTypeFloat32,
    };
    BNNSPoolingLayerParameters layer_params_pool1 = {
        .x_stride = 2,
        .y_stride = 2,
        .x_padding = 0,
        .y_padding = 0,
        .k_width = 2,
        .k_height = 2,
        .in_channels = 24,
        .out_channels = 24,
        .pooling_function = BNNSPoolingFunctionMax,
    };
    BNNSFilterParameters filter_params_pool1 = {};
    BNNSFilter pool1 = BNNSFilterCreatePoolingLayer(&i_desc_pool1, &o_desc_pool1, &layer_params_pool1, &filter_params_pool1);
    
    
    Float32 *input2 = out;
    Float32 *out2 = (Float32 *)calloc(28*24*24, sizeof(Float32));
    BNNSFilterApply(pool1, (void *)input2, out2);
    cout << "out2: " << endl;
    for (int i = 0; i < 10; ++i) {
        cout << out2[i] << endl;
    }
    
    BNNSImageStackDescriptor i_desc_conv2;
    bzero(&i_desc_conv2,sizeof(i_desc_conv2));
    i_desc_conv2.width = 28;
    i_desc_conv2.height = 24;
    i_desc_conv2.channels = 24;
    i_desc_conv2.row_stride = 28;
    i_desc_conv2.image_stride = 28 * 24;
    i_desc_conv2.data_type = BNNSDataTypeFloat32;
    i_desc_conv2.data_bias = 0;
    i_desc_conv2.data_scale = 0;

    BNNSImageStackDescriptor o_desc_conv2;
    bzero(&o_desc_conv2,sizeof(o_desc_conv2));
    o_desc_conv2.width = 28;
    o_desc_conv2.height = 24;
    o_desc_conv2.channels = 24;
    o_desc_conv2.row_stride = 28;
    o_desc_conv2.image_stride = 28*24;
    o_desc_conv2.data_type = BNNSDataTypeFloat32;
    o_desc_conv2.data_scale = 0;
    o_desc_conv2.data_bias = 0;
    
    BNNSConvolutionLayerParameters layer_params_conv2;
    bzero(&layer_params_conv2, sizeof(layer_params_conv2));
    layer_params_conv2.k_width = 5;
    layer_params_conv2.k_height = 5;
    layer_params_conv2.in_channels = 24;
    layer_params_conv2.out_channels = 24;
    layer_params_conv2.x_stride = 1;
    layer_params_conv2.y_stride = 1;
    layer_params_conv2.x_padding = 2;
    layer_params_conv2.y_padding = 2;
    layer_params_conv2.activation.function = BNNSActivationFunctionRectifiedLinear;
    layer_params_conv2.weights.data = load_data("model-W_conv2-5x5x24x24");
    layer_params_conv2.bias.data = load_data("model-b_conv2-24");
    layer_params_conv2.weights.data_type = BNNSDataTypeFloat32;
    layer_params_conv2.bias.data_type = BNNSDataTypeFloat32;

    BNNSFilterParameters filter_params_conv2 = {};
    bzero(&filter_params_conv2, sizeof(filter_params_conv2));
    BNNSFilter conv2 = BNNSFilterCreateConvolutionLayer(&i_desc_conv2, &o_desc_conv2,
                                                        &layer_params_conv2,
                                                        &filter_params_conv2);
    
    Float32 *input3 = out2;
    Float32 *out3 = (Float32 *)calloc(28*24*24, sizeof(Float32));
    int r = BNNSFilterApply(conv2, input3, out3); if (r != 0) { cout << "Fail!" << endl; }
    
    cout << "out3: " << endl;
    for (int i = 0; i < 10; ++i) {
        cout << out3[i] << endl;
    }
    
    // Max Pool 2d
    
    BNNSImageStackDescriptor i_desc_pool2 = {
        .width = 28,
        .height = 24,
        .channels = 24,
        .row_stride = 28,
        .image_stride = 28*24,
        .data_type = BNNSDataTypeFloat32,
    };
    
    BNNSImageStackDescriptor o_desc_pool2 = {
        .width = 14,
        .height = 12,
        .channels = 24,
        .row_stride = 14,
        .image_stride = 14*12,
        .data_type = BNNSDataTypeFloat32,
    };
    BNNSPoolingLayerParameters layer_params_pool2 = {
        .x_stride = 2,
        .y_stride = 2,
        .x_padding = 0,
        .y_padding = 0,
        .k_width = 2,
        .k_height = 2,
        .in_channels = 24,
        .out_channels = 24,
        .pooling_function = BNNSPoolingFunctionMax,
    };
    BNNSFilterParameters filter_params_pool2 = {};
    BNNSFilter pool2 = BNNSFilterCreatePoolingLayer(&i_desc_pool2, &o_desc_pool2, &layer_params_pool2, &filter_params_pool2);
    
    
    Float32 *input4 = out3;
    Float32 *out4 = (Float32 *)calloc(14*12*24, sizeof(Float32));
    BNNSFilterApply(pool2, (void *)input4, out4);
    cout << "out4: " << endl;
    for (int i = 0; i < 5; ++i) {
        cout << out4[i] << endl;
    }
    
    BNNSImageStackDescriptor i_desc_conv3 = {
        .width = 14,
        .height = 12,
        .channels = 24,
        .row_stride = 14,
        .image_stride = 14 * 12,
        .data_type = BNNSDataTypeFloat32,
    };
    BNNSImageStackDescriptor o_desc_conv3 = {
        .width = 14,
        .height = 12,
        .channels = 32,
        .row_stride = 14,
        .image_stride = 14 * 12,
        .data_type = BNNSDataTypeFloat32,
    };
    BNNSConvolutionLayerParameters layer_params_conv3 = {
        .k_width = 3,
        .k_height = 3,
        .in_channels = 24,
        .out_channels = 32,
        .x_stride = 1,
        .y_stride = 1,
        .x_padding = 1,
        .y_padding = 1,
        .activation = BNNSActivationFunctionRectifiedLinear,
        .weights.data = load_data("model-W_conv3-3x3x24x32"),
        .bias.data = load_data("model-b_conv3-32"),
        .weights.data_type = BNNSDataTypeFloat32,
        .bias.data_type = BNNSDataTypeFloat32,
    };
    BNNSFilterParameters filter_params_conv3 = {};
    BNNSFilter conv3 = BNNSFilterCreateConvolutionLayer(&i_desc_conv3, &o_desc_conv3,
                                                        &layer_params_conv3,
                                                        &filter_params_conv3);
    Float32 *input5 = out4;
    Float32 *out5 = (Float32 *)calloc(14*12*32, sizeof(Float32));
    BNNSFilterApply(conv3, (void *)input5, out5);
    cout << "out5: " << endl;
    for (int i = 0; i < 5; ++i) {
        cout << out5[i] << endl;
    }
    
    
    BNNSImageStackDescriptor i_desc_pool3 = {
        .width = 14,
        .height = 12,
        .channels = 32,
        .row_stride = 14,
        .image_stride = 14*12,
        .data_type = BNNSDataTypeFloat32,
    };
    
    BNNSImageStackDescriptor o_desc_pool3 = {
        .width = 7,
        .height = 6,
        .channels = 32,
        .row_stride = 7,
        .image_stride = 7*6,
        .data_type = BNNSDataTypeFloat32,
    };
    BNNSPoolingLayerParameters layer_params_pool3 = {
        .x_stride = 2,
        .y_stride = 2,
        .x_padding = 0,
        .y_padding = 0,
        .k_width = 2,
        .k_height = 2,
        .in_channels = 32,
        .out_channels = 32,
        .pooling_function = BNNSPoolingFunctionMax,
    };
    BNNSFilterParameters filter_params_pool3 = {};
    BNNSFilter pool3 = BNNSFilterCreatePoolingLayer(&i_desc_pool3, &o_desc_pool3, &layer_params_pool3, &filter_params_pool3);
    
    
    Float32 *input6 = out5;
    Float32 *out6 = (Float32 *)calloc(7*6*32, sizeof(Float32));
    BNNSFilterApply(pool3, (void *)input6, out6);
    cout << "out6: " << endl;
    for (int i = 0; i < 5; ++i) {
        cout << out6[i] << endl;
    }
    
    
    
    
    return 0;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    
    Float32 *input = (Float32 *) calloc(48*56, sizeof(Float32));
    for (int i = 0; i < 48*56; i++) {
        input[i] = 1.0;
    }
    
    BNNSNetBuilder *lefteye_net = new BNNSNetBuilder(56, 48, 1);
    lefteye_net->conv2d(1, 24, 7, "model-W_conv1-7x7x1x24", "model-b_conv1-24", BNNSActivationFunctionRectifiedLinear)
               ->MaxPool2d(2, 2)
               ->conv2d(24, 24, 5, "model-W_conv2-5x5x24x24", "model-b_conv2-24", BNNSActivationFunctionRectifiedLinear)
               ->MaxPool2d(2, 2)
               ->conv2d(24, 32, 3, "model-W_conv3-3x3x24x32", "model-b_conv3-32", BNNSActivationFunctionRectifiedLinear)
               ->MaxPool2d(2, 2)
               ->Linear(1344, 256, "model-W_fc1-1344x256", "model-b_fc1-256", relu)
               ->Linear(256, 7, "model-W_nn1-256x7", "model-b_nn1-7", none)
               ->softmax()->build();

    lefteye_net->inspect();
    
    Float32 *output = (Float32 *) lefteye_net->apply(input);
    for (int i = 0; i < 7; i++) {
        cout << output[i] << endl;
    }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
