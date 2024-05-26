import argparse
import os

# Given dictionary-like structure
args_dict = {
    'root_path': './realtime',
    'store_name': 'model',
    'result_path': 'results',
    'modality': 'RGB',
    'modality_det': 'RGB',
    'modality_clf': 'RGB',
    'pretrain_modality': 'RGB',
    'dataset': 'kinetics',
    'n_classes_det': 2,
    'n_finetune_classes_det': 2,
    'n_classes_clf': 83,
    'n_finetune_classes_clf': 83,
    'n_classes': 83,
    'n_finetune_classes': 83,
    'sample_size': 112,
    'sample_duration_det': 8,
    'sample_duration_clf': 32,
    'sample_duration': 32,
    'initial_scale': 1.0,
    'n_scales': 5,
    'scale_step': 0.84089641525,
    'mean_dataset': 'activitynet',
    'no_mean_norm': False,
    'std_norm': False,
    'n_val_samples': 1,
    'resume_path_det': 'trained_models/Pretrained models/egogesture_resnetl_10_RGB_8.pth',
    'resume_path_clf': 'trained_models/Pretrained models/egogesture_resnext_101_RGB_32.pth',
    'resume_path': 'trained_models/Pretrained models/egogesture_resnext_101_RGB_32.pth',
    'pretrain_path_det': '',
    'pretrain_path_clf': '',
    'pretrain_path': '',
    'ft_begin_index': 0,
    'no_train': True,
    'no_val': True,
    'test': True,
    'test_subset': 'val',
    'scale_in_test': 1.0,
    'crop_position_in_test': 'c',
    'no_softmax_in_test': False,
    'no_cuda': True,
    'n_threads': 16,
    'no_hflip': False,
    'norm_value': 1,
    'model_det': 'resnetl',
    'model_depth_det': 10,
    'resnet_shortcut_det': 'A',
    'wide_resnet_k_det': 2,
    'resnext_cardinality_det': 32,
    'model': 'resnext',
    'model_depth': 101,
    'resnet_shortcut': 'B',
    'wide_resnet_k': 2,
    'resnext_cardinality': 32,
    'model_clf': 'resnext',
    'model_depth_clf': 101,
    'resnet_shortcut_clf': 'B',
    'wide_resnet_k_clf': 2,
    'resnext_cardinality_clf': 32,
    'width_mult': 1.0,
    'width_mult_det': 0.5,
    'width_mult_clf': 1.0,
    'manual_seed': 1,
    'det_strategy': 'median',
    'det_queue_size': 4,
    'det_counter': 2.0,
    'clf_strategy': 'median',
    'clf_queue_size': 32,
    'clf_threshold_pre': 1.0,
    'clf_threshold_final': 0.15,
    'stride_len': 1,
    'ft_portion': 'complete',
    'groups': 3,
    'downsample': 1,
    'scales': [1.0, 0.84089641525, 0.7071067811803005, 0.5946035574934808, 0.4999999999911653],
    'arch': 'resnext',
    'mean': [114.7748, 107.7354, 99.475],
    'std': [38.7568578, 37.88248729, 40.02898126]
}

# Create a Namespace object
args = argparse.Namespace()

# Populate the Namespace object with the given key-value pairs
for key, value in args_dict.items():
    setattr(args, key, value)

if args.root_path != '':
    args.result_path = os.path.join(args.root_path, args.result_path)
    if args.resume_path:
        args.resume_path = os.path.join(args.root_path, args.resume_path)
    if args.pretrain_path:
        args.pretrain_path = os.path.join(args.root_path, args.pretrain_path)
    if args.resume_path_det:
        args.resume_path_det = os.path.join(args.root_path, args.resume_path_det)
    if args.pretrain_path_det:
        args.pretrain_path_det = os.path.join(args.root_path, args.pretrain_path_det)
    if args.resume_path_clf:
        args.resume_path_clf = os.path.join(args.root_path, args.resume_path_clf)
    if args.pretrain_path_clf:
        args.pretrain_path_clf = os.path.join(args.root_path, args.pretrain_path_clf)