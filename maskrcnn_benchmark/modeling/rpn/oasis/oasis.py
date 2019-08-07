import math
import torch
import torch.nn.functional as F
from torch import nn

#from .inference import make_oasis_postprocessor
#from .loss import make_oasis_loss_evaluator

#from maskrcnn_benchmark.layers import Scale

class OASISHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(OASISHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.OASIS.NUM_CLASSES - 1

        semantic_tower = []
        instance_tower = []
        for i in range(cfg.MODEL.OASIS.NUM_CONVS):
            semantic_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            semantic_tower.append(nn.GroupNorm(32, in_channels))
            semantic_tower.append(nn.ReLU())
            instance_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            instance_tower.append(nn.GroupNorm(32, in_channels))
            instance_tower.append(nn.ReLU())

        self.add_module('semantic_tower', nn.Sequential(*semantic_tower))
        self.add_module('instance_tower', nn.Sequential(*instance_tower))
        self.semantic_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.instance_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.semantic_tower, self.instance_tower,
                        self.semantic_logits, self.instance_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        #prior_prob = cfg.MODEL.OASIS.PRIOR_PROB
        #bias_value = -math.log((1 - prior_prob) / prior_prob)
        #torch.nn.init.constant_(self.semantic_logits.bias, bias_value)

        #self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        semantics = []
        instances = []
        for l, feature in enumerate(x):
            semantic_tower = self.semantic_tower(feature)
            semantics.append(self.semantic_logits(semantic_tower))
            instances.append(self.instance_pred(self.instance_tower(feature)))
        return semantics, instances

class OASISModule(torch.nn.Module):
    """
    Module for OASIS computation. Takes feature maps from the backbone and
    OASIS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(OASISModule, self).__init__()

        head = OASISHead(cfg, in_channels)

        #box_selector_test = make_oasis_postprocessor(cfg)

        #loss_evaluator = make_oasis_loss_evaluator(cfg)
        self.head = head
        #self.box_selector_test = box_selector_test
        #self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.OASIS.FPN_STRIDES

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        semantics, instances = self.head(features)
        import pdb
        pdb.set_trace()
        print(len(semantics), semantics[0].shape, instances[0].shape, len(targets), targets[0].shape)

        return semantics, {'se_loss': torch.tensor(0)}

        '''
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations
    '''


def build_oasis(cfg, in_channels):
    return OASISModule(cfg, in_channels)