"""***********************
***** Classification *****
***********************"""

"""***** VGG *****"""
from components.models.vgg.vgg11 import VGG11
from components.models.vgg.vgg13 import VGG13
from components.models.vgg.vgg16 import VGG16
from components.models.vgg.vgg19 import VGG19

"""***** ResNet *****"""
from components.models.resnet.resnet18 import ResNet18
from components.models.resnet.resnet34 import ResNet34
from components.models.resnet.resnet50 import ResNet50
from components.models.resnet.resnet101 import ResNet101
from components.models.resnet.resnet151 import ResNet151



"""**************************************
***** Object Detection (anchor box) *****
**************************************"""
from components.models.ssd.ssd300_vgg16 import SSD300VGG16
from components.models.ssd.ssd300_lite_vgg16 import SSD300LiteVGG16
from components.models.ssd.ssdwide_lite_vgg16 import SSDWideLiteVGG16



"""******************************
***** Semantic Segmentation *****
******************************"""

"""***** FCN *****"""
from components.models.fcn.fcn32s_vgg16 import FCN32sVGG16
from components.models.fcn.fcn16s_vgg16 import FCN16sVGG16
from components.models.fcn.fcn8s_vgg16 import FCN8sVGG16

"""***** UNet *****"""
from components.models.unet.unet import UNet
from components.models.unet.lite_unet import LiteUNet



"""***************
***** Others *****
***************"""
from components.models.ririverce.ririverce_cifar10net9 import RiriverceCifar10Net9

from components.models.word2vec.skip_gram import SkipGram