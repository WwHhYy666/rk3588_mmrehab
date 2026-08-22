default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=410,
        switch_pipeline=[
            dict(
                type='LoadImage',
                backend_args=dict(
                    backend='petrel',
                    path_mapping=dict({'data/': 's3://openmmlab/datasets/'}))),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(
                type='RandomBBoxTransform',
                shift_factor=0.0,
                scale_factor=[0.5, 1.5],
                rotate_factor=90),
            dict(type='TopdownAffine', input_size=(288, 384)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(
                type='Albumentation',
                transforms=[
                    dict(type='Blur', p=0.1),
                    dict(type='MedianBlur', p=0.1),
                    dict(
                        type='CoarseDropout',
                        max_holes=1,
                        max_height=0.4,
                        max_width=0.4,
                        min_holes=1,
                        min_height=0.2,
                        min_width=0.2,
                        p=0.5)
                ]),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='SimCCLabel',
                    input_size=(288, 384),
                    sigma=(6.0, 6.93),
                    simcc_split_ratio=2.0,
                    normalize=False,
                    use_dark=False)),
            dict(type='PackPoseInputs')
        ])
]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False
backend_args = dict(
    backend='petrel', path_mapping=dict({'data/': 's3://openmmlab/datasets/'}))
train_cfg = dict(by_epoch=True, max_epochs=420, val_interval=10)
val_cfg = dict()
test_cfg = dict()
max_epochs = 420
stage2_num_epochs = 10
base_lr = 0.004
randomness = dict(seed=21)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    loss_scale='dynamic')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=210,
        end=420,
        T_max=210,
        by_epoch=True,
        convert_to_iter_based=True)
]
auto_scale_lr = dict(base_batch_size=1024)
codec = dict(
    type='SimCCLabel',
    input_size=(288, 384),
    sigma=(6.0, 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.0,
        widen_factor=1.0,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=
            '/mnt/petrelfs/jiangtao/ckpts/cspnext-l-body-cocktail7-udp-pretrain-bs256-384/best_coco/AP_epoch_210.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=1024,
        out_channels=17,
        input_size=(288, 384),
        in_featuremap_size=(9, 12),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.0,
            label_softmax=True),
        decoder=dict(
            type='SimCCLabel',
            input_size=(288, 384),
            sigma=(6.0, 6.93),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    test_cfg=dict(flip_test=True))
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/'
train_pipeline = [
    dict(
        type='LoadImage',
        backend_args=dict(
            backend='petrel',
            path_mapping=dict({'data/': 's3://openmmlab/datasets/'}))),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=90),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='PhotometricDistortion'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0)
        ]),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='SimCCLabel',
            input_size=(288, 384),
            sigma=(6.0, 6.93),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(
        type='LoadImage',
        backend_args=dict(
            backend='petrel',
            path_mapping=dict({'data/': 's3://openmmlab/datasets/'}))),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='PackPoseInputs')
]
train_pipeline_stage2 = [
    dict(
        type='LoadImage',
        backend_args=dict(
            backend='petrel',
            path_mapping=dict({'data/': 's3://openmmlab/datasets/'}))),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.0,
        scale_factor=[0.5, 1.5],
        rotate_factor=90),
    dict(type='TopdownAffine', input_size=(288, 384)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5)
        ]),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='SimCCLabel',
            input_size=(288, 384),
            sigma=(6.0, 6.93),
            simcc_split_ratio=2.0,
            normalize=False,
            use_dark=False)),
    dict(type='PackPoseInputs')
]
aic_coco = [(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12), (7, 14),
            (8, 16), (9, 11), (10, 13), (11, 15)]
crowdpose_coco = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                  (7, 12), (8, 13), (9, 14), (10, 15), (11, 16)]
mpii_coco = [(0, 16), (1, 14), (2, 12), (3, 11), (4, 13), (5, 15), (10, 10),
             (11, 8), (12, 6), (13, 5), (14, 7), (15, 9)]
jhmdb_coco = [(3, 6), (4, 5), (5, 12), (6, 11), (7, 8), (8, 7), (9, 14),
              (10, 13), (11, 10), (12, 9), (13, 16), (14, 15)]
halpe_coco = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
              (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14),
              (15, 15), (16, 16)]
ochuman_coco = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
                (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13),
                (14, 14), (15, 15), (16, 16)]
posetrack_coco = [(0, 0), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
                  (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14),
                  (15, 15), (16, 16)]
dataset_coco = dict(
    type='CocoDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='coco/annotations/person_keypoints_train2017.json',
    data_prefix=dict(img='detection/coco/train2017/'),
    pipeline=[])
dataset_aic = dict(
    type='AicDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='aic/annotations/aic_train.json',
    data_prefix=dict(
        img=
        'pose/ai_challenge/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
    ),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12),
                     (7, 14), (8, 16), (9, 11), (10, 13), (11, 15)])
    ])
dataset_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='crowdpose/annotations/mmpose_crowdpose_trainval.json',
    data_prefix=dict(img='pose/CrowdPose/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                     (7, 12), (8, 13), (9, 14), (10, 15), (11, 16)])
    ])
dataset_mpii = dict(
    type='MpiiDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='mpii/annotations/mpii_train.json',
    data_prefix=dict(img='pose/MPI/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 16), (1, 14), (2, 12), (3, 11), (4, 13), (5, 15),
                     (10, 10), (11, 8), (12, 6), (13, 5), (14, 7), (15, 9)])
    ])
dataset_jhmdb = dict(
    type='JhmdbDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='jhmdb/annotations/Sub1_train.json',
    data_prefix=dict(img='pose/JHMDB/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(3, 6), (4, 5), (5, 12), (6, 11), (7, 8), (8, 7), (9, 14),
                     (10, 13), (11, 10), (12, 9), (13, 16), (14, 15)])
    ])
dataset_halpe = dict(
    type='HalpeDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='halpe/annotations/halpe_train_v1.json',
    data_prefix=dict(img='pose/Halpe/hico_20160224_det/images/train2015'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
                     (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                     (13, 13), (14, 14), (15, 15), (16, 16)])
    ])
dataset_posetrack = dict(
    type='PoseTrack18Dataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='posetrack18/annotations/posetrack18_train.json',
    data_prefix=dict(img='pose/PoseChallenge2018/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 0), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
                     (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14),
                     (15, 15), (16, 16)])
    ])
train_dataloader = dict(
    batch_size=256,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[
            dict(
                type='CocoDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='coco/annotations/person_keypoints_train2017.json',
                data_prefix=dict(img='detection/coco/train2017/'),
                pipeline=[]),
            dict(
                type='AicDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='aic/annotations/aic_train.json',
                data_prefix=dict(
                    img=
                    'pose/ai_challenge/ai_challenger_keypoint_train_20170902/keypoint_train_images_20170902/'
                ),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 6), (1, 8), (2, 10), (3, 5), (4, 7),
                                 (5, 9), (6, 12), (7, 14), (8, 16), (9, 11),
                                 (10, 13), (11, 15)])
                ]),
            dict(
                type='CrowdPoseDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='crowdpose/annotations/mmpose_crowdpose_trainval.json',
                data_prefix=dict(img='pose/CrowdPose/images/'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
                                 (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
                                 (10, 15), (11, 16)])
                ]),
            dict(
                type='MpiiDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='mpii/annotations/mpii_train.json',
                data_prefix=dict(img='pose/MPI/images/'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 16), (1, 14), (2, 12), (3, 11), (4, 13),
                                 (5, 15), (10, 10), (11, 8), (12, 6), (13, 5),
                                 (14, 7), (15, 9)])
                ]),
            dict(
                type='JhmdbDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='jhmdb/annotations/Sub1_train.json',
                data_prefix=dict(img='pose/JHMDB/'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(3, 6), (4, 5), (5, 12), (6, 11), (7, 8),
                                 (8, 7), (9, 14), (10, 13), (11, 10), (12, 9),
                                 (13, 16), (14, 15)])
                ]),
            dict(
                type='HalpeDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='halpe/annotations/halpe_train_v1.json',
                data_prefix=dict(
                    img='pose/Halpe/hico_20160224_det/images/train2015'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                                 (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
                                 (10, 10), (11, 11), (12, 12), (13, 13),
                                 (14, 14), (15, 15), (16, 16)])
                ]),
            dict(
                type='PoseTrack18Dataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='posetrack18/annotations/posetrack18_train.json',
                data_prefix=dict(img='pose/PoseChallenge2018/'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 0), (3, 3), (4, 4), (5, 5), (6, 6),
                                 (7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
                                 (12, 12), (13, 13), (14, 14), (15, 15),
                                 (16, 16)])
                ])
        ],
        pipeline=[
            dict(
                type='LoadImage',
                backend_args=dict(
                    backend='petrel',
                    path_mapping=dict({'data/': 's3://openmmlab/datasets/'}))),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(
                type='RandomBBoxTransform',
                scale_factor=[0.5, 1.5],
                rotate_factor=90),
            dict(type='TopdownAffine', input_size=(288, 384)),
            dict(type='mmdet.YOLOXHSVRandomAug'),
            dict(type='PhotometricDistortion'),
            dict(
                type='Albumentation',
                transforms=[
                    dict(type='Blur', p=0.1),
                    dict(type='MedianBlur', p=0.1),
                    dict(
                        type='CoarseDropout',
                        max_holes=1,
                        max_height=0.4,
                        max_width=0.4,
                        min_holes=1,
                        min_height=0.2,
                        min_width=0.2,
                        p=1.0)
                ]),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='SimCCLabel',
                    input_size=(288, 384),
                    sigma=(6.0, 6.93),
                    simcc_split_ratio=2.0,
                    normalize=False,
                    use_dark=False)),
            dict(type='PackPoseInputs')
        ],
        test_mode=False))
val_coco = dict(
    type='CocoDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='coco/annotations/person_keypoints_val2017.json',
    data_prefix=dict(img='detection/coco/val2017/'),
    pipeline=[])
val_aic = dict(
    type='AicDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='aic/annotations/aic_val.json',
    data_prefix=dict(
        img=
        'pose/ai_challenge/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/'
    ),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 6), (1, 8), (2, 10), (3, 5), (4, 7), (5, 9), (6, 12),
                     (7, 14), (8, 16), (9, 11), (10, 13), (11, 15)])
    ])
val_crowdpose = dict(
    type='CrowdPoseDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='crowdpose/annotations/mmpose_crowdpose_test.json',
    data_prefix=dict(img='pose/CrowdPose/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 5), (1, 6), (2, 7), (3, 8), (4, 9), (5, 10), (6, 11),
                     (7, 12), (8, 13), (9, 14), (10, 15), (11, 16)])
    ])
val_mpii = dict(
    type='MpiiDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='mpii/annotations/mpii_val.json',
    data_prefix=dict(img='pose/MPI/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 16), (1, 14), (2, 12), (3, 11), (4, 13), (5, 15),
                     (10, 10), (11, 8), (12, 6), (13, 5), (14, 7), (15, 9)])
    ])
val_jhmdb = dict(
    type='JhmdbDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='jhmdb/annotations/Sub1_test.json',
    data_prefix=dict(img='pose/JHMDB/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(3, 6), (4, 5), (5, 12), (6, 11), (7, 8), (8, 7), (9, 14),
                     (10, 13), (11, 10), (12, 9), (13, 16), (14, 15)])
    ])
val_halpe = dict(
    type='HalpeDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='halpe/annotations/halpe_val_v1.json',
    data_prefix=dict(img='detection/coco/val2017/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
                     (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                     (13, 13), (14, 14), (15, 15), (16, 16)])
    ])
val_ochuman = dict(
    type='OCHumanDataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='ochuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json',
    data_prefix=dict(img='pose/OCHuman/images/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
                     (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12),
                     (13, 13), (14, 14), (15, 15), (16, 16)])
    ])
val_posetrack = dict(
    type='PoseTrack18Dataset',
    data_root='data/',
    data_mode='topdown',
    ann_file='posetrack18/annotations/posetrack18_val.json',
    data_prefix=dict(img='pose/PoseChallenge2018/'),
    pipeline=[
        dict(
            type='KeypointConverter',
            num_keypoints=17,
            mapping=[(0, 0), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
                     (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14),
                     (15, 15), (16, 16)])
    ])
val_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/',
        data_mode='topdown',
        ann_file='coco/annotations/person_keypoints_val2017.json',
        bbox_file=
        'data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='detection/coco/val2017/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImage',
                backend_args=dict(
                    backend='petrel',
                    path_mapping=dict({'data/': 's3://openmmlab/datasets/'}))),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(288, 384)),
            dict(type='PackPoseInputs')
        ]))
test_dataloader = dict(
    batch_size=64,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        datasets=[
            dict(
                type='CocoDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='coco/annotations/person_keypoints_val2017.json',
                data_prefix=dict(img='detection/coco/val2017/'),
                pipeline=[]),
            dict(
                type='AicDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='aic/annotations/aic_val.json',
                data_prefix=dict(
                    img=
                    'pose/ai_challenge/ai_challenger_keypoint_validation_20170911/keypoint_validation_images_20170911/'
                ),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 6), (1, 8), (2, 10), (3, 5), (4, 7),
                                 (5, 9), (6, 12), (7, 14), (8, 16), (9, 11),
                                 (10, 13), (11, 15)])
                ]),
            dict(
                type='CrowdPoseDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='crowdpose/annotations/mmpose_crowdpose_test.json',
                data_prefix=dict(img='pose/CrowdPose/images/'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
                                 (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
                                 (10, 15), (11, 16)])
                ]),
            dict(
                type='MpiiDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='mpii/annotations/mpii_val.json',
                data_prefix=dict(img='pose/MPI/images/'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 16), (1, 14), (2, 12), (3, 11), (4, 13),
                                 (5, 15), (10, 10), (11, 8), (12, 6), (13, 5),
                                 (14, 7), (15, 9)])
                ]),
            dict(
                type='JhmdbDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='jhmdb/annotations/Sub1_test.json',
                data_prefix=dict(img='pose/JHMDB/'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(3, 6), (4, 5), (5, 12), (6, 11), (7, 8),
                                 (8, 7), (9, 14), (10, 13), (11, 10), (12, 9),
                                 (13, 16), (14, 15)])
                ]),
            dict(
                type='HalpeDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='halpe/annotations/halpe_val_v1.json',
                data_prefix=dict(img='detection/coco/val2017/'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                                 (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
                                 (10, 10), (11, 11), (12, 12), (13, 13),
                                 (14, 14), (15, 15), (16, 16)])
                ]),
            dict(
                type='OCHumanDataset',
                data_root='data/',
                data_mode='topdown',
                ann_file=
                'ochuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json',
                data_prefix=dict(img='pose/OCHuman/images/'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                                 (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
                                 (10, 10), (11, 11), (12, 12), (13, 13),
                                 (14, 14), (15, 15), (16, 16)])
                ]),
            dict(
                type='PoseTrack18Dataset',
                data_root='data/',
                data_mode='topdown',
                ann_file='posetrack18/annotations/posetrack18_val.json',
                data_prefix=dict(img='pose/PoseChallenge2018/'),
                pipeline=[
                    dict(
                        type='KeypointConverter',
                        num_keypoints=17,
                        mapping=[(0, 0), (3, 3), (4, 4), (5, 5), (6, 6),
                                 (7, 7), (8, 8), (9, 9), (10, 10), (11, 11),
                                 (12, 12), (13, 13), (14, 14), (15, 15),
                                 (16, 16)])
                ])
        ],
        pipeline=[
            dict(
                type='LoadImage',
                backend_args=dict(
                    backend='petrel',
                    path_mapping=dict({'data/': 's3://openmmlab/datasets/'}))),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(288, 384)),
            dict(type='PackPoseInputs')
        ],
        test_mode=True))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/person_keypoints_val2017.json')
test_evaluator = [
    dict(type='PCKAccuracy', thr=0.2),
    dict(type='AUC'),
    dict(type='EPE')
]
launcher = 'slurm'
work_dir = '../ckpts/rtmpose-l-body-cocktail7-384'
