import torch
import torch.nn.functional as F



class VGG19(torch.nn.Module):

    def __init__(self, input_channels, num_classes):
        super(VGG19, self).__init__()
        self.input_channels = input_channels
        if type(num_classes) is list:
            self.num_classes = num_classes
        else:
            self.num_classes = [num_classes]
        self.block1_conv1 = torch.nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.block1_bn1 = torch.nn.BatchNorm2d(64)
        self.block1_conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block1_bn2 = torch.nn.BatchNorm2d(64)
        self.block1_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block2_conv1 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.block2_bn1 = torch.nn.BatchNorm2d(128)
        self.block2_conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.block2_bn2 = torch.nn.BatchNorm2d(128)
        self.block2_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block3_conv1 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.block3_bn1 = torch.nn.BatchNorm2d(256)
        self.block3_conv2 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block3_bn2 = torch.nn.BatchNorm2d(256)
        self.block3_conv3 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block3_bn3 = torch.nn.BatchNorm2d(256)
        self.block3_conv4 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block3_bn4 = torch.nn.BatchNorm2d(256)
        self.block3_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block4_conv1 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.block4_bn1 = torch.nn.BatchNorm2d(512)
        self.block4_conv2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block4_bn2 = torch.nn.BatchNorm2d(512)
        self.block4_conv3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block4_bn3 = torch.nn.BatchNorm2d(512)
        self.block4_conv4 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block4_bn4 = torch.nn.BatchNorm2d(256)
        self.block4_pool = torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.block5_conv1 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_bn1 = torch.nn.BatchNorm2d(512)
        self.block5_conv2 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_bn2 = torch.nn.BatchNorm2d(512)
        self.block5_conv3 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_bn3 = torch.nn.BatchNorm2d(512)
        self.block5_conv4 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block5_bn4 = torch.nn.BatchNorm2d(256)
        self.block5_pool = torch.nn.AdaptiveMaxPool2d((7, 7))
        self.classifier_linear1 = torch.nn.Linear(512*7*7, 4096)
        self.classifier_bn1 = torch.nn.BatchNorm1d(4096)
        self.classifier_linear2 = torch.nn.Linear(4096, 4096)
        self.classifier_bn2 = torch.nn.BatchNorm1d(4096)
        self.classifier_linear3 = torch.nn.Linear(4096, sum(self.num_classes))        

    def forward(self, x):
        h = x
        h = F.relu(self.block1_bn1(self.block1_conv1(h)))
        h = F.relu(self.block1_bn2(self.block1_conv2(h)))
        h = self.block1_pool(h)
        h = F.relu(self.block2_bn1(self.block2_conv1(h)))
        h = F.relu(self.block2_bn2(self.block2_conv2(h)))
        h = self.block2_pool(h)
        h = F.relu(self.block3_bn1(self.block3_conv1(h)))
        h = F.relu(self.block3_bn2(self.block3_conv2(h)))
        h = F.relu(self.block3_bn3(self.block3_conv3(h)))
        h = F.relu(self.block3_bn4(self.block3_conv4(h)))
        h = self.block3_pool(h)
        h = F.relu(self.block4_bn1(self.block4_conv1(h)))
        h = F.relu(self.block4_bn2(self.block4_conv2(h)))
        h = F.relu(self.block4_bn3(self.block4_conv3(h)))
        h = F.relu(self.block4_bn4(self.block4_conv4(h)))
        h = self.block4_pool(h)
        h = F.relu(self.block5_bn1(self.block5_conv1(h)))
        h = F.relu(self.block5_bn2(self.block5_conv2(h)))
        h = F.relu(self.block5_bn3(self.block5_conv3(h)))
        h = F.relu(self.block5_bn4(self.block5_conv4(h)))
        h = self.block5_pool(h)
        h = h.view(h.size(0), -1)
        h = F.relu(self.classifier_bn1(self.classifier_linear1(h)))
        h = F.relu(self.classifier_bn2(self.classifier_linear2(h)))
        h = self.classifier_linear3(h)
        if len(self.num_classes) > 1:
            y = []
            c_start = 0
            for c in self.num_classes:
                y.append(h[:, c_start:c_start+c])
                c_start += c
        else:
            y = h
        return y