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

- (void)viewDidLoad {
    [super viewDidLoad];
    BNNSNetBuilder *mnist_net = new BNNSNetBuilder(28, 28, 1);
    mnist_net->conv2d(1, 32, 5, "conv-weights-0", "biases-0", BNNSActivationFunctionRectifiedLinear)
             ->MaxPool2d(2)
             ->conv2d(32, 64, 5, "conv-weights-1", "biases-1", BNNSActivationFunctionRectifiedLinear)
             ->MaxPool2d(2)
             ->Linear(4*4*64, 1024, "fc-weights-2", "fc-biases-2", relu)
             ->Linear(1024, 10, "fc-weights-3", "fc-biases-3", none)
             ->build();
    Float32 *input = (Float32 *) calloc(28*28, sizeof(Float32));
    Float32 *output = (Float32 *) mnist_net->apply(input);
    for (int i = 0; i < 10; i++) {
        cout << output[i] << endl;
    }
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
