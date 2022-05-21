import torch.nn as nn
# import torch.nn.functional as F
from torchvision import models
from im2mesh.common import normalize_imagenet
from im2mesh.encoder import batchnet as bnet
from im2mesh.encoder import res34 as res34
from im2mesh.encoder import gaussian_conv
import torch
from im2mesh.encoder import channel_attention_1d
import os
import copy
import numpy as np
import cv2
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from im2mesh.encoder import attention_1d



def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


class CamExtractor():
    """
        Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


def preprocess_image(pil_im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im)
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float().cuda()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.7
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr


def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('../results'):
        os.makedirs('../results')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'jet')
    # Save colored heatmap
    path_to_file = os.path.join('../results', file_name + '_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('../results', file_name + '_Cam_On_Image_hot.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('../results', file_name + '_Cam_Grayscale.png')
    print(path_to_file)
    save_image(activation_map, path_to_file)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ConvEncoderRes18(nn.Module):

    def __init__(self, block, layers, c_dim=128, zero_init_residual=False):
        super(ConvEncoderRes18, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, c_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ConvEncoder(nn.Module):
    r''' Simple convolutional encoder network.

    It consists of 5 convolutional layers, each downsampling the input by a
    factor of 2, and a final fully-connected layer projecting the output to
    c_dim dimenions.

    Args:
        c_dim (int): output dimension of latent embedding
    '''

    def __init__(self, c_dim=128):
        super().__init__()
        self.conv0 = nn.Conv2d(6, 32, 3, stride=2)
        self.conv1 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc_out = nn.Linear(512, c_dim)
        self.actvn = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size = x.size(0)

        net = self.conv0(x)
        net = self.conv1(self.actvn(self.bn1(net)))
        net = self.conv2(self.actvn(self.bn2(net)))
        net = self.conv3(self.actvn(self.bn3(net)))
        net1 = self.conv4(self.actvn(self.bn4(net)))
        net1 = self.avgpool(self.actvn(self.bn5(net1)))
        net = net1.view(batch_size, -1)
        # print('net', net.size())
        # print('net_1', self.actvn(net).size())
        out = self.fc_out(net)

        return out, net1


def gray(tensor):
    # TODO: make efficient
    # print(tensor)
    b, c, h, w = tensor.size()
    R = tensor[:, 0, :, :]
    G = tensor[:, 1, :, :]
    B = tensor[:, 2, :, :]
    tem = torch.add(0.299 * R, 0.587 * G)
    tensor_gray = torch.add(tem, 0.114 * B)
    # print(tensor_gray.size())
    return tensor_gray.view(b, 1, h, w)



class Resnet34(nn.Module):
    r''' ResNet-34 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        # self.features = models.resnet34(pretrained=True)
        self.features = res34.resnet34(pretrained=True)
        self.features.fc = nn.Sequential()
        self.gaussion_conv1 = gaussian_conv.GaussianBlurConv(1, 0.3).cuda()
        self.gaussion_conv2 = gaussian_conv.GaussianBlurConv(1, 0.4).cuda()
        self.gaussion_conv3 = gaussian_conv.GaussianBlurConv(1, 0.5).cuda()
        self.gaussion_conv4 = gaussian_conv.GaussianBlurConv(1, 0.6).cuda()
        self.gaussion_conv5 = gaussian_conv.GaussianBlurConv(1, 0.7).cuda()
        self.gaussion_conv6 = gaussian_conv.GaussianBlurConv(1, 0.8).cuda()
        self.attention_sift = attention_1d.SELayer1(4, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.l_dog_encoder = ConvEncoderRes18(BasicBlock, [2, 2, 2, 2], 256).to('cuda')
        self.m_dog_encoder = ConvEncoderRes18(BasicBlock, [2, 2, 2, 2], 256).to('cuda')
        self.s_dog_encoder = ConvEncoderRes18(BasicBlock, [2, 2, 2, 2], 256).to('cuda')
        self.attention = channel_attention_1d.attention_layer(256)
        self.vis_conv_feature = False
        self.one_hot_vis_feature = False
        # self.l3_branch = self.make_branch_layer(MDCBlock, 256, 512)
        self.conv2_l2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv3x3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False)
        self.avgpool_l2 = nn.AdaptiveAvgPool2d(1)
        self.avgpool_node1 = nn.AdaptiveAvgPool2d(7)
        # self.fc_l2 = nn.Linear(512, c_dim)

        # self.l4_branch = self.make_branch_layer(MDCBlock, 512, 512)
        self.conv3_l3 = nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv3x3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, bias=False)
        self.avgpool_l3 = nn.AdaptiveAvgPool2d(1)
        # self.fc_l3 = nn.Linear(512, c_dim)

        self.conv1x1_whole = nn.Conv2d(1280, 512, kernel_size=1, stride=1, padding=0, bias=False)

        if use_linear:
            self.fc = nn.Linear(512, c_dim)
            self.fc_l2 = nn.Linear(512, c_dim)
            self.fc_l3 = nn.Linear(512, c_dim)
            self.fusion_dog_ori = nn.Linear(1024, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def make_branch_layer(self, block, inplanes, planes):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
        )
        layers = []
        layers.append(block(inplanes, planes, downsample))

        return nn.Sequential(*layers)

    def gaussian(self, x):
        gray_x = gray(x)
        gaussian1 = self.gaussion_conv1(gray_x)
        gaussian2 = self.gaussion_conv2(gray_x)
        gaussian3 = self.gaussion_conv3(gray_x)
        gaussian4 = self.gaussion_conv4(gray_x)
        gaussian5 = self.gaussion_conv5(gray_x)
        gaussian6 = self.gaussion_conv6(gray_x)
        dog1 = torch.sub(gaussian2, gaussian1)
        dog2 = torch.sub(gaussian4, gaussian3)
        dog3 = torch.sub(gaussian6, gaussian5)
        dog_tem = torch.cat((dog1, dog2), dim=1)
        dog = torch.cat((dog_tem, dog3), dim=1)
        return dog

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        x_m = F.interpolate(x, scale_factor=0.8)
        x_s = F.interpolate(x, scale_factor=0.6)
        x0, node1, node2 = self.features(x)
        dog_l = self.gaussian(x)
        # print('dog_l', dog_l.size())
        dog_m = self.gaussian(x_m)
        dog_s = self.gaussian(x_s)
        out_dog = self.l_dog_encoder(dog_l)
        out_dog_m = self.m_dog_encoder(dog_m)
        out_dog_s = self.s_dog_encoder(dog_s)

        out1 = self.conv2_l2(node1)
        out1 = self.avgpool_l2(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc_l2(out1)

        node1_p = self.avgpool_node1(node1)
        out1_2 = torch.cat((node1_p, node2), dim=1)
        out2 = self.conv3_l3(out1_2)
        out2 = self.avgpool_l3(out2)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fc_l3(out2)

        out2_3 = torch.cat((out1_2, node2), dim=1)
        out2_3 = self.conv1x1_whole(out2_3)
        out0 = self.avgpool(out2_3)
        out0 = out0.view(out0.size(0), -1)
        out0 = self.fc(out0)

        out_dog = self.fusion_dog_ori(torch.cat((out_dog, out0, out_dog_m, out_dog_s), dim=1))

        attention_out_dog = self.attention(out_dog.unsqueeze(dim=-1))

        return out0, out1, out2, attention_out_dog.squeeze(1)


class Resnet50(nn.Module):
    r''' ResNet-50 encoder network.

    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out


class Resnet101(nn.Module):
    r''' ResNet-101 encoder network.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, c_dim, normalize=True, use_linear=True):
        super().__init__()
        self.normalize = normalize
        self.use_linear = use_linear
        self.features = models.resnet50(pretrained=True)
        self.features.fc = nn.Sequential()
        if use_linear:
            self.fc = nn.Linear(2048, c_dim)
        elif c_dim == 2048:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 2048 if use_linear is False')

    def forward(self, x):
        if self.normalize:
            x = normalize_imagenet(x)
        net = self.features(x)
        out = self.fc(net)
        return out
