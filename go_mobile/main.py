import torch

def main():
    #TODO: use argparse
    path = "../pretrained_models/mnist.pth"
    model = torch.load(path)
    for child in model.children():
        print child
    return 0

if __name__ == '__main__':
    main()
