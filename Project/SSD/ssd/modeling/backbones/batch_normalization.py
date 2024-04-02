import torch
from torch import nn
from typing import Tuple, List

def extractorLayer(in_channels, 
                   out_channels, 
                   num_filters, 
                   output_stride=2, 
                   output_padding=1):
    extractor = nn.Sequential(
        nn.ReLU(), 
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(), 
        nn.Conv2d(
            in_channels=num_filters,
            out_channels=out_channels,
            kernel_size=3,
            stride=output_stride,
            padding=output_padding
        ),
        nn.ReLU(), 
        nn.BatchNorm2d(out_channels),
        )
    return extractor


class NormalizedModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.conv_size = 3
        self.conv_padding = 1
        self.pool_stride = 2
        self.pool_size = 2

        self.first_extractor = nn.Sequential(
            # Layer 1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=self.conv_size,
                stride=1,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_size, 
                         stride=self.pool_stride
            ),
            nn.BatchNorm2d(32),

            # Layer 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self.conv_size,
                stride=1,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=self.pool_size, 
                         stride=self.pool_stride
            ),
            nn.BatchNorm2d(64),

            # Layer 3
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self.conv_size,
                stride=1,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),


            # Layer 4
            nn.Conv2d(
                in_channels=64,
                out_channels=output_channels[0],
                kernel_size=self.conv_size,
                stride=2,
                padding=self.conv_padding
            ),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels[0]),
        )

        self.output2 = extractorLayer(output_channels[0], output_channels[1], 128)

        self.output3 = extractorLayer(output_channels[1], output_channels[2], 256)

        self.output4 = extractorLayer(output_channels[2], output_channels[3], 128)

        self.output5 = extractorLayer(output_channels[3], output_channels[4], 128)

        self.output6 = extractorLayer(output_channels[4], 
                                output_channels[5], 
                                128, 
                                output_stride=1, 
                                output_padding=0)
        


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out_features.append(self.first_extractor(x))
        out_features.append(self.output2(out_features[0]))
        out_features.append(self.output3(out_features[1]))
        out_features.append(self.output4(out_features[2]))
        out_features.append(self.output5(out_features[3]))
        out_features.append(self.output6(out_features[4]))

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        
        return tuple(out_features)

