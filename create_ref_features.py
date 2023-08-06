import os
import timm
import pickle
import argparse
from collections import OrderedDict

from tqdm import tqdm

import torch
from datasets.mvtec import MVTecDataset, MVTEC_CLASS_NAMES
from datasets.btad import BTADDataset, BTAD_CLASS_NAMES
from datasets.mvtec_3d import MVTec3DDataset, MVTEC3D_CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser('FOD')

    parser.add_argument('--save_path', type=str, default='./rfeatures_w50')
    
    parser.add_argument('--dataset', default='mvtec', type=str, metavar='D',
                        help='dataset name: mvtec/btad/mvtec3d (default: mvtec)')
    parser.add_argument('--data_path', default='/data2/yxc/datasets/mvtec_anomaly_detection', type=str)
    parser.add_argument('--class_name', default='none', type=str, metavar='C',
                        help='class name for MVTecAD (default: none)')
    parser.add_argument('--inp_size', default=256, type=int, metavar='C',
                        help='image resize dimensions (default: 256)')
    parser.add_argument('--batch_size', default=32, type=int, metavar='B',
                        help='train batch size (default: 32)')
    parser.add_argument('--num_workers', default=4, type=int, metavar='G',
                        help='number of data loading workers (default: 4)')
    
    parser.add_argument('--backbone_arch', default='wide_resnet50_2', type=str, metavar='A',
                        help='feature extractor: (default: efficientnet_b6)')
    parser.add_argument('--feature_levels', default=3, type=int, metavar='L',
                        help='number of feature layers (default: 3)')

    return parser.parse_args()


def main():
    # device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    args = parse_args()
    args.img_size = (args.inp_size, args.inp_size)  
    args.crop_size = (args.inp_size, args.inp_size)  
    args.norm_mean, args.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    os.makedirs(args.save_path, exist_ok=True)

    # load model
    encoder = timm.create_model(args.backbone_arch, features_only=True, 
                out_indices=[i + 1 for i in range(args.feature_levels)], pretrained=True)
    encoder = encoder.to(device).eval()
    print("Feature Dimensions", encoder.feature_info.channels())
    
    if args.dataset == 'mvtec':
        CLASS_NAMES = MVTEC_CLASS_NAMES
    elif args.dataset == 'btad':
        CLASS_NAMES = BTAD_CLASS_NAMES
    elif args.dataset == 'mvtec3d':
        CLASS_NAMES = MVTEC3D_CLASS_NAMES
    else:
        raise ValueError("Not recognized dataset: {}!".format(args.dataset))
    
    for class_name in CLASS_NAMES:
        args.class_name = class_name
        if args.class_name in MVTEC_CLASS_NAMES:
            dataset = MVTecDataset(args, is_train=True)
        elif args.class_name in BTAD_CLASS_NAMES:
            dataset = BTADDataset(args.data_path, classname=args.class_name, resize=256, cropsize=256, is_train=True)
        elif args.class_name in MVTEC3D_CLASS_NAMES:
            dataset = MVTec3DDataset(args.data_path, classname=args.class_name, resize=256, cropsize=256, is_train=True)  
        else:
            raise ValueError('Invalid Class Name: {}'.format(args.class_name))

        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)

        train_outputs = OrderedDict([('layer0', []), ('layer1', []), ('layer2', [])])

        # extract train set features
        train_feature_filepath = os.path.join(args.save_path, '%s.pkl' % class_name)
        if not os.path.exists(train_feature_filepath):
            for (images, _, _, _, _) in tqdm(loader, '| feature extraction | train | %s |' % class_name):
                # model prediction
                with torch.no_grad():
                    outputs = encoder(images.to(device))  
                # get intermediate layer outputs
                for k, v in zip(train_outputs.keys(), outputs):
                    train_outputs[k].append(v.cpu().detach())
            # every single feature level, calculate mean and cov statistics.
            for k, v in train_outputs.items():
                embedding_vectors = torch.cat(v, 0)
                m = torch.nn.AvgPool2d(3, 1, 1)
                embedding_vectors = m(embedding_vectors)
                    
                B, C, H, W = embedding_vectors.size()  # (32, 256, 56, 56)
                embedding_vectors = embedding_vectors.view(B, C, H * W)

                mean = torch.mean(embedding_vectors, dim=0).numpy()  # (C, H*W)
    
                train_outputs[k] = mean
            with open(train_feature_filepath, 'wb') as f:
                pickle.dump(train_outputs, f)
        else:
            print('load train set feature from: %s' % train_feature_filepath)
            with open(train_feature_filepath, 'rb') as f:
                train_outputs = pickle.load(f)


if __name__ == '__main__':
    main()