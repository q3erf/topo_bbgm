import torch.nn as nn
from torchvision import models


class VGG16_base(nn.Module):
    def __init__(self, batch_norm=True):
        super(VGG16_base, self).__init__()
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone(batch_norm)
        # print('node_layers=', self.node_layers)
        # print('edge_layers=', self.edge_layers)
        # print('final_layers=', self.final_layers)

    def forward(self, *input):
        raise NotImplementedError

    @staticmethod
    def get_backbone(batch_norm):
        """
        Get pretrained VGG16 models for feature extraction.
        :return: feature sequence
        """
        # print('batch_norm=', batch_norm)
        # if batch_norm:
        #     model = models.vgg16_bn(pretrained=True)
        # else:
        #     model = models.vgg16(pretrained=True)

        model = models.vgg16(pretrained=True)

        # with open('results/spair/vgg16_base.txt', 'w') as f:
        #     f.write(str(model))

        conv_layers = nn.Sequential(*list(model.features.children()))
        conv_list = node_list = edge_list = []

        # get the output of relu4_2(node features) and relu5_1(edge features)
        cnt_m, cnt_r = 1, 0
        for layer, module in enumerate(conv_layers):
            if isinstance(module, nn.Conv2d): 
                cnt_r += 1
            if isinstance(module, nn.MaxPool2d):
                cnt_r = 0
                cnt_m += 1

            conv_list += [module]

            if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
                node_list = conv_list
                conv_list = []
            elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU):
                edge_list = conv_list
                conv_list = []

        assert len(node_list) > 0 and len(edge_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)
        # 剩余的层+自适应最大池化
        final_layers = nn.Sequential(*conv_list, nn.AdaptiveMaxPool2d(1, 1))
        return node_layers, edge_layers, final_layers


class VGG16_bn(VGG16_base):
    def __init__(self):
        super(VGG16_bn, self).__init__(True)

class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.node_layers, self.edge_layers, self.final_layers = self.get_backbone()

    def forward(self, *input):
        raise NotImplementedError

    @staticmethod
    def get_backbone():
        """
        Get pretrained ResNet101 models for feature extraction.
        :return: feature sequence
        """
        resnet = models.resnet101(pretrained=True)

        # print('model=', type(model))
        # print(model)

        # with open('results/spair/resnet101.txt', 'w') as f:
        #     f.write(str(model))

        conv_layers = nn.Sequential(*list(resnet.children()))

        conv_list = node_list = edge_list = []

        resnet_module_list = [resnet.conv1,
                                resnet.bn1,
                                resnet.relu,
                                resnet.maxpool,
                                resnet.layer1,
                                resnet.layer2,
                                resnet.layer3,
                                resnet.layer4]


        node_list = [resnet.conv1,
                        resnet.bn1,
                        resnet.relu,
                        resnet.maxpool,
                        resnet.layer1,
                        resnet.layer2,
                        resnet.layer3]
        
        edge_list = [resnet.layer4[:1]]
        # print('resnet.layer4', type(resnet.layer4))
        # print(resnet.layer4)
        # final_list = [resnet.layer4[1:], resnet.avgpool]
        final_list = [resnet.layer4[1:]]
        # get the output of relu4_2(node features) and relu5_1(edge features)
        # cnt_m, cnt_r = 1, 0
        # for layer, module in enumerate(conv_layers):
        #     # 判断一个对象是否是一个已知的类型
        #     if isinstance(module, nn.Conv2d): 
        #         cnt_r += 1
        #     if isinstance(module, nn.MaxPool2d):
        #         cnt_r = 0
        #         cnt_m += 1

        #     conv_list += [module]

        #     if cnt_m == 4 and cnt_r == 2 and isinstance(module, nn.ReLU):
        #         node_list = conv_list
        #         conv_list = []
        #     elif cnt_m == 5 and cnt_r == 1 and isinstance(module, nn.ReLU):
        #         edge_list = conv_list
        #         conv_list = []

        assert len(node_list) > 0 and len(edge_list) > 0

        # Set the layers as a nn.Sequential module
        node_layers = nn.Sequential(*node_list)
        edge_layers = nn.Sequential(*edge_list)
        # 剩余的层+自适应最大池化
        # final_layers = nn.Sequential(*conv_list, nn.AdaptiveMaxPool2d(1, 1))
        # final_layers = nn.Sequential(*final_list)
        final_layers = nn.Sequential(*final_list, nn.AdaptiveMaxPool2d(1, 1))
        # print('node_layers=', node_layers)
        # print('edge_layers=', edge_layers)
        # print('final_layers=', final_layers)
        return node_layers, edge_layers, final_layers
    