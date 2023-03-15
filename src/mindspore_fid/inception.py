import operator

import mindcv
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import torch
from mindcv import DownLoad

from mindcv.models.inception_v3 import InceptionA, InceptionC, InceptionE

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501


class InceptionV3(nn.Cell):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 use_fid_inception=True):
        """Build pretrained InceptionV3

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        use_fid_inception : bool
            If true, uses the pretrained Inception model used in Tensorflow's
            FID implementation. If false, uses the pretrained Inception model
            available in torchvision. The FID Inception model has different
            weights and a slightly different structure from torchvision's
            Inception model. If you want to compute FID scores, you are
            strongly advised to set this parameter to true to get comparable
            results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.CellList()

        if use_fid_inception:
            inception = fid_inception_v3()
        else:
            inception = _inception_v3(weights='DEFAULT')

        # Block 0: input to maxpool1
        block0 = [
            inception.conv1a,
            inception.conv2a,
            inception.conv2b,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.SequentialCell(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.conv3b,
                inception.conv4a,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.SequentialCell(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.inception5b,
                inception.inception5c,
                inception.inception5d,
                inception.inception6a,
                inception.inception6b,
                inception.inception6c,
                inception.inception6d,
                inception.inception6e,
            ]
            self.blocks.append(nn.SequentialCell(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.inception7a,
                inception.inception7b,
                inception.inception7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.SequentialCell(*block3))

        for param in self.get_parameters():
            param.requires_grad = requires_grad

    def construct(self, inp):
        """Get Inception feature maps

        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)

        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = ops.interpolate(x,
                                size=(299, 299),
                                mode='bilinear',
                                align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def _inception_v3(*args, **kwargs):
    if kwargs['weights'] == 'DEFAULT':
        kwargs['pretrained'] = True
    elif kwargs['weights'] is None:
        kwargs['pretrained'] = False
    else:
        raise ValueError(
            'weights=={} not supported in torchvision {}'
        )
    del kwargs['weights']
    return mindcv.models.inception_v3(*args, **kwargs)


def fid_inception_v3():
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs torchvision's Inception and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(num_classes=1008,
                              weights=None)
    inception.inception5b = FIDInceptionA(192, pool_features=32)
    inception.inception5c = FIDInceptionA(256, pool_features=64)
    inception.inception5d = FIDInceptionA(288, pool_features=64)
    inception.inception6d = FIDInceptionC(768, channels_7x7=128)
    inception.inception6c = FIDInceptionC(768, channels_7x7=160)
    inception.inception6d = FIDInceptionC(768, channels_7x7=160)
    inception.inception6e = FIDInceptionC(768, channels_7x7=192)
    inception.inception7b = FIDInceptionE_1(1280)
    inception.inception7c = FIDInceptionE_2(2048)
    # params_dict = inception.parameters_dict()
    # sorted_params_dict = dict(sorted(params_dict.items(), key=operator.itemgetter(0)))
    # with open("params_dict.txt", "w+") as f:
    #     lines = []
    #     for k, v in sorted_params_dict.items():
    #         line = f"name:{k} , shape:{v.shape}\n"
    #         lines.append(line)
    #     f.writelines(lines)

    model_name = "pt_inception-2015-12-05-6726825d"
    local_path = f'pretrained_models/{model_name}' + ".pth"
    DownLoad().download_url(url=FID_WEIGHTS_URL, path='pretrained_models')
    state_dict = torch.load(local_path, map_location=torch.device('cpu'))
    ms_ckpt = torch_to_mindspore(state_dict)
    # sorted_ms_ckpt = sorted(ms_ckpt, key=lambda i: i['name'])
    # with open("convert_ckpt.txt", "w+") as f:
    #     lines = []
    #     for i in sorted_ms_ckpt:
    #         line = f"name:{i['name']}, shape:{i['data'].shape} \n"
    #         lines.append(line)
    #     f.writelines(lines)
    ms_ckpt_path = local_path.replace('.pth', '.ckpt')
    from mindspore.train.serialization import save_checkpoint
    save_checkpoint(ms_ckpt, ms_ckpt_path)
    state_dict = ms.load_checkpoint(ms_ckpt_path)
    ms.load_param_into_net(inception, state_dict)
    return inception


def torch_to_mindspore(state_dict):
    ms_ckpt = []
    for k, v in state_dict.items():
        if 'fc' in k:
            k = k.replace('fc', 'classifier')
        if 'num_batches_tracked' in k:
            continue
        if 'branch_pool' in k:
            k = k.replace('branch_pool', 'branch_pool.1')
        if 'Conv2d_' in k:
            k = k.replace('Conv2d_', 'conv')
            if '_3x3' in k:
                k = k.replace('_3x3', '')
        if 'running_mean' in k:
            k = k.replace('running_mean', 'moving_mean')
        if 'running_var' in k:
            k = k.replace('running_var', 'moving_variance')
        if 'bn.weight' in k:
            k = k.replace('bn.weight', 'bn.gamma')
        if 'bn.bias' in k:
            k = k.replace('bn.bias', 'bn.beta')
        if 'conv3b_1x1' in k:
            k = k.replace('conv3b_1x1', 'conv3b')
        if 'Mixed_5' in k:
            k = k.replace('Mixed_5', 'inception5')
            if 'branch1x1' in k:
                k = k.replace('branch1x1', 'branch0')
            if 'branch5x5_1' in k:
                k = k.replace('branch5x5_1', 'branch1.0')
            if 'branch5x5_2' in k:
                k = k.replace('branch5x5_2', 'branch1.1')
            if 'branch3x3dbl_1' in k:
                k = k.replace('branch3x3dbl_1', 'branch2.0')
            if 'branch3x3dbl_2' in k:
                k = k.replace('branch3x3dbl_2', 'branch2.1')
            if 'branch3x3dbl_3' in k:
                k = k.replace('branch3x3dbl_3', 'branch2.2')
        if 'Mixed_6a' in k:
            k = k.replace('Mixed_6', 'inception6')
            if 'branch3x3' in k:
                k = k.replace('branch3x3', 'branch0')
            if 'branch3x3dbl_1' in k:
                k = k.replace('branch3x3dbl_1', 'branch1.0')
            if 'branch3x3dbl_2' in k:
                k = k.replace('branch3x3dbl_2', 'branch1.1')
            if 'branch3x3dbl_3' in k:
                k = k.replace('branch3x3dbl_3', 'branch1.2')
            if 'branch0dbl_1' in k:
                k = k.replace('branch0dbl_1', 'branch1.0')
            if 'branch0dbl_2' in k:
                k = k.replace('branch0dbl_2', 'branch1.1')
            if 'branch0dbl_3' in k:
                k = k.replace('branch0dbl_3', 'branch1.2')

        if 'Mixed_6b' in k or 'Mixed_6c' in k or 'Mixed_6d' in k or 'Mixed_6e' in k:
            k = k.replace('Mixed_6', 'inception6')
            if 'branch1x1' in k:
                k = k.replace('branch1x1', 'branch0')
            if 'branch7x7_1' in k:
                k = k.replace('branch7x7_1', 'branch1.0')
            if 'branch7x7_2' in k:
                k = k.replace('branch7x7_2', 'branch1.1')
            if 'branch7x7_3' in k:
                k = k.replace('branch7x7_3', 'branch1.2')
            if 'branch7x7dbl_1' in k:
                k = k.replace('branch7x7dbl_1', 'branch2.0')
            if 'branch7x7dbl_2' in k:
                k = k.replace('branch7x7dbl_2', 'branch2.1')
            if 'branch7x7dbl_3' in k:
                k = k.replace('branch7x7dbl_3', 'branch2.2')
            if 'branch7x7dbl_4' in k:
                k = k.replace('branch7x7dbl_4', 'branch2.3')
            if 'branch7x7dbl_5' in k:
                k = k.replace('branch7x7dbl_5', 'branch2.4')

        if 'Mixed_7a' in k:
            k = k.replace('Mixed_7', 'inception7')
            if 'branch3x3_1' in k:
                k = k.replace('branch3x3_1', 'branch0.0')
            if 'branch3x3_2' in k:
                k = k.replace('branch3x3_2', 'branch0.1')
            if 'branch7x7x3_1' in k:
                k = k.replace('branch7x7x3_1', 'branch1.0')
            if 'branch7x7x3_2' in k:
                k = k.replace('branch7x7x3_2', 'branch1.1')
            if 'branch7x7x3_3' in k:
                k = k.replace('branch7x7x3_3', 'branch1.2')
            if 'branch7x7x3_4' in k:
                k = k.replace('branch7x7x3_4', 'branch1.3')

        if 'Mixed_7b' in k or 'Mixed_7c' in k:
            k = k.replace('Mixed_7', 'inception7')
            if 'branch1x1' in k:
                k = k.replace('branch1x1', 'branch0')
            if 'branch3x3_1' in k:
                k = k.replace('branch3x3_1', 'branch1')
            if 'branch3x3_2a' in k:
                k = k.replace('branch3x3_2a', 'branch1a')
            if 'branch3x3_2b' in k:
                k = k.replace('branch3x3_2b', 'branch1b')
            if 'branch3x3dbl_1' in k:
                k = k.replace('branch3x3dbl_1', 'branch2.0')
            if 'branch3x3dbl_2' in k:
                k = k.replace('branch3x3dbl_2', 'branch2.1')
            if 'branch3x3dbl_3a' in k:
                k = k.replace('branch3x3dbl_3a', 'branch2a')
            if 'branch3x3dbl_3b' in k:
                k = k.replace('branch3x3dbl_3b', 'branch2b')
        ms_ckpt.append({'name': k, 'data': ms.Tensor(v.numpy())})
    return ms_ckpt


class FIDInceptionA(InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = ops.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [x0, x1, x2, branch_pool]
        return ops.cat(outputs, 1)


class FIDInceptionC(InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = ops.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [x0, x1, x2, branch_pool]
        return ops.concat(outputs, 1)


class FIDInceptionE_1(InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = ops.concat((self.branch1a(x1), self.branch1b(x1)), axis=1)
        x2 = self.branch2(x)
        x2 = ops.concat((self.branch2a(x2), self.branch2b(x2)), axis=1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation
        branch_pool = ops.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [x0, x1, x2, branch_pool]
        return ops.concat(outputs, 1)


class FIDInceptionE_2(InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = ops.concat((self.branch1a(x1), self.branch1b(x1)), axis=1)
        x2 = self.branch2(x)
        x2 = ops.concat((self.branch2a(x2), self.branch2b(x2)), axis=1)
        branch_pool = ops.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [x0, x1, x2, branch_pool]
        return ops.concat(outputs, 1)
