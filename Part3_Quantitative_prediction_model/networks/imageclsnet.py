from config.config_ import *
from utils.common_util import *
from networks.densenet import class_densenet121_dropout, class_densenet121_large_dropout
# from networks.sedensenet import class_densenet121_dropout, class_densenet121_large_dropout
from networks.inception_v3 import class_inceptionv3_dropout
from networks.resnet import class_resnet34_dropout, class_resnet18_dropout,class_resnet50_dropout

model_names = {
    'class_densenet121_dropout': 'densenet121-a639ec97.pth',
    'class_densenet121_large_dropout': 'densenet121-a639ec97.pth',
    'class_inceptionv3_dropout': 'inception_v3_google-1a9a5a14.pth',
    'class_resnet34_dropout': 'resnet34-333f7ec4.pth',
    'class_resnet18_dropout': 'resnet18-5c106cde.pth',
    'class_resnet50_dropout': 'resnet50-19c8e357.pth',
}

def init_network(params):
    architecture = params.get('architecture', 'class_resnet50_dropout')
    num_classes = params.get('num_classes', 5)
    in_channels = params.get('in_channels', 3)

    pretrained_file = opj(PRETRAINED_DIR, model_names[architecture])
    print(">> Using pre-trained model.")
    net = eval(architecture)(num_classes=num_classes, in_channels=in_channels, pretrained_file=pretrained_file)
    return net