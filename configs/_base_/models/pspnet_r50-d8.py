# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)  # Segmentation usually uses SyncBN
data_preprocessor = dict(           # The config of data preprocessor, usually includes image normalization and augmentation.
    type='SegDataPreProcessor',     # The type of data preprocessor.
    mean=[123.675, 116.28, 103.53], # Mean values used for normalizing the input images.
    std=[58.395, 57.12, 57.375],    # Standard variance used for normalizing the input images.
    bgr_to_rgb=True,                # Whether to convert image from BGR to RGB.
    pad_val=0,                      # Padding value of image.
    seg_pad_val=255)                # Padding value of segmentation map.
model = dict(
    type='EncoderDecoder',          # Name of segmentor
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',     # The ImageNet pretrained backbone to be loaded
    backbone=dict(
        type='ResNetV1c',           # The type of backbone. Please refer to mmseg/models/backbones/resnet.py for details.
        depth=50,                   # Depth of backbone. Normally 50, 101 are used.
        num_stages=4,               # Number of stages of backbone.
        out_indices=(0, 1, 2, 3),   # The index of output feature maps produced in each stages.
        dilations=(1, 1, 2, 4),     # The dilation rate of each layer.
        strides=(1, 2, 1, 1),       # The stride of each layer.
        norm_cfg=norm_cfg,          # The configuration of norm layer.
        norm_eval=False,            # Whether to freeze the statistics in BN
        style='pytorch',            # The style of backbone, 'pytorch' means that stride 2 layers are in 3x3 conv, 
                                    # 'caffe' means stride 2 layers are in 1x1 convs.
        contract_dilation=True),    # When dilation > 1, whether contract first layer of dilation.
    decode_head=dict(
        type='PSPHead',             # Type of decode head. Please refer to mmseg/models/decode_heads for available options.
        in_channels=2048,           # Input channel of decode head.
        in_index=3,                 # The index of feature map to select.
        channels=512,               # The intermediate channels of decode head.
        pool_scales=(1, 2, 3, 6),   # The avg pooling scales of PSPHead. Please refer to paper for details.
        dropout_ratio=0.1,          # The dropout ratio before final classification layer.
        num_classes=19,             # Number of segmentation class. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
        norm_cfg=norm_cfg,          # The configuration of norm layer.
        align_corners=False,        # The align_corners argument for resize in decoding.
        loss_decode=dict(           # Config of loss function for the decode_head.
            type='CrossEntropyLoss',    # Type of loss used for segmentation.
            use_sigmoid=False,          # Whether use sigmoid activation for segmentation.
            loss_weight=1.0)),          # Loss weight of decode_head.
    auxiliary_head=dict(
        type='FCNHead',             # Type of auxiliary head. Please refer to mmseg/models/decode_heads for available options.
        in_channels=1024,           # Input channel of auxiliary head.
        in_index=2,                 # The index of feature map to select.
        channels=256,               # The intermediate channels of decode head.
        num_convs=1,                # Number of convs in FCNHead. It is usually 1 in auxiliary head.
        concat_input=False,         # Whether concat output of convs with input before classification layer.
        dropout_ratio=0.1,          # The dropout ratio before final classification layer.
        num_classes=19,             # Number of segmentation class. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
        norm_cfg=norm_cfg,          # The configuration of norm layer.
        align_corners=False,        # The align_corners argument for resize in decoding.
        loss_decode=dict(           # Config of loss function for the auxiliary_head.
            type='CrossEntropyLoss',    # Type of loss used for segmentation.
            use_sigmoid=False,          # Whether use sigmoid activation for segmentation.
            loss_weight=0.4)),          # Loss weight of auxiliary_head.
    # model training and testing settings
    train_cfg=dict(),               # train_cfg is just a place holder for now.
    test_cfg=dict(mode='whole'))    # The test mode, options are 'whole' and 'slide'. 
                                    # 'whole': whole image fully-convolutional test. 
                                    # 'slide': sliding crop window on the image.    
