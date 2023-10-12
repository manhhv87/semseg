# dataset settings
dataset_type = 'CityscapesDataset'      # Dataset type, this will be used to define the dataset.
data_root = 'data/cityscapes/'          # Root path of data.
crop_size = (512, 1024)                 # The crop size during training.
train_pipeline = [                      # Training pipeline.
    dict(type='LoadImageFromFile'),     # First pipeline to load images from file path.
    dict(type='LoadAnnotations'),       # Second pipeline to load annotations for current image.
    dict(
        type='RandomResize',            # Augmentation pipeline that resize the images and their annotations.
        scale=(2048, 1024),             # The scale of image.
        ratio_range=(0.5, 2.0),         # The augmented scale range as ratio.
        keep_ratio=True),               # Whether to keep the aspect ratio when resizing the image.
    dict(type='RandomCrop',             # Augmentation pipeline that randomly crop a patch from current image.
         crop_size=crop_size,           # The crop size of patch.
         cat_max_ratio=0.75),           # The max area ratio that could be occupied by single category.
    dict(type='RandomFlip',             # Augmentation pipeline that flip the images and their annotations
         prob=0.5),                     # The ratio or probability to flip
    dict(type='PhotoMetricDistortion'), # Augmentation pipeline that distort current image with several photo metric methods.
    dict(type='PackSegInputs')          # Pack the inputs data for the semantic segmentation.
]
test_pipeline = [
    dict(type='LoadImageFromFile'),     # First pipeline to load images from file path
    dict(type='Resize',                 # Use resize augmentation
         scale=(2048, 1024),            # Images scales for resizing.
         keep_ratio=True),              # Whether to keep the aspect ratio when resizing the image.
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),       # Load annotations for semantic segmentation provided by dataset.
    dict(type='PackSegInputs')          # Pack the inputs data for the semantic segmentation.
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(        # Train dataloader config
    batch_size=2,               # Batch size of a single GPU
    num_workers=2,              # Worker to pre-fetch data for each single GPU
    persistent_workers=True,    # Shut down the worker processes after an epoch end, 
                                # which can accelerate training speed.
    sampler=dict(type='InfiniteSampler', shuffle=True),     # Randomly shuffle during training.
    dataset=dict(               # Train dataset config
        type=dataset_type,      # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=data_root,    # The root of dataset.      
        data_prefix=dict(
            img_path='leftImg8bit/train', seg_map_path='gtFine/train'), # Prefix for training data.
        pipeline=train_pipeline))   # Processing pipeline. This is passed by the train_pipeline created before.
val_dataloader = dict(
    batch_size=1,               # Batch size of a single GPU
    num_workers=4,              # Worker to pre-fetch data for each single GPU
    persistent_workers=True,    # Shut down the worker processes after an epoch end, 
                                # which can accelerate testing speed.      
    sampler=dict(type='DefaultSampler', shuffle=False), # Not shuffle during validation and testing.
    dataset=dict(               # Test dataset config
        type=dataset_type,      # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=data_root,    # The root of dataset.
        data_prefix=dict(
            img_path='leftImg8bit/val', seg_map_path='gtFine/val'), # Prefix for testing data.
        pipeline=test_pipeline))    # Processing pipeline. This is passed by the test_pipeline created before.
test_dataloader = val_dataloader

# The metric to measure the accuracy. Here, we use IoUMetric.
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
