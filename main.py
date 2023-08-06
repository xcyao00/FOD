import argparse

from torch.backends import cudnn
from utils.utils import *

from trainer import Trainer
from datasets.mvtec import MVTEC_CLASS_NAMES
from datasets.btad import BTAD_CLASS_NAMES
from datasets.mvtec_3d import MVTEC3D_CLASS_NAMES


def main(args):
    cudnn.benchmark = True
    init_seeds(3407)
        
    trainer = Trainer(args)

    if args.mode == 'train':
        img_auc, pix_auc = trainer.train()
    elif args.mode == 'test':
        img_auc, pix_auc = trainer.test(vis=args.vis, checkpoint_path=args.checkpoint)
    print("Class Name: {}".format(args.class_name))
    print('Image AUC: {}'.format(img_auc))
    print('Pixel AUC: {}'.format(pix_auc))

    return img_auc, pix_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    # dataset config
    parser.add_argument('--dataset', default='mvtec', type=str, metavar='D',
                        help='dataset name: mvtec/btad/mvtec3d (default: mvtec)')
    parser.add_argument('--data_path', default='/data/to/your/path', type=str)
    parser.add_argument('--class_name', default='none', type=str, metavar='C',
                        help='class name for MVTecAD (default: none)')
    parser.add_argument('--inp_size', default=256, type=int, metavar='C',
                        help='image resize dimensions (default: 256)')
    parser.add_argument('--batch_size', default=1, type=int, metavar='B',
                        help='train batch size (default: 32)')
    parser.add_argument('--num_workers', default=4, type=int, metavar='G',
                        help='number of data loading workers (default: 4)')
    # model config
    parser.add_argument('--backbone_arch', default='wide_resnet50_2', type=str, metavar='A',
                        help='feature extractor: (default: wide_resnet50_2)')
    parser.add_argument('--feature_levels', default=2, type=int, metavar='L',
                        help='number of feature layers (default: 2)')
    parser.add_argument('--rfeatures_path', default='rfeatures_w50', type=str, metavar='A',
                        help='path to reference features (default: rfeatures_w50)')
    parser.add_argument('--with_intra', action='store_true', default=True,
                        help='Learning intra correlations (default: True)')
    parser.add_argument('--with_inter', action='store_true', default=True,
                        help='Learning inter correlations (default: True)')
    parser.add_argument('--lambda1', type=int, default=0.5)
    parser.add_argument('--lambda2', type=int, default=0.5)
    # misc
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--save_prefix', type=str, default='mvtec')
    parser.add_argument('--checkpoint', default='', type=str, metavar='D',
                        help='used in test phase, set same with the save_path/save_prefix')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='Visualize localization map (default: False)')

    args = parser.parse_args()
    
    args.device = torch.device("cuda")
    args.img_size = (args.inp_size, args.inp_size)  
    args.crop_size = (args.inp_size, args.inp_size)  
    args.norm_mean, args.norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    args_dict = vars(args)
    print('------------ Options -------------')
    for k, v in sorted(args_dict.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    if args.dataset == 'mvtec':
        CLASS_NAMES = MVTEC_CLASS_NAMES
    elif args.dataset == 'btad':
        CLASS_NAMES = BTAD_CLASS_NAMES
    elif args.dataset == 'mvtec3d':
        CLASS_NAMES = MVTEC3D_CLASS_NAMES
    else:
        raise ValueError("Not recognized dataset: {}!".format(args.dataset))
    
    img_aucs, pix_aucs = [], []
    for class_name in CLASS_NAMES:
        args.class_name = class_name
        img_auc, pix_auc = main(args)
        img_aucs.append(img_auc)
        pix_aucs.append(pix_auc)
    for i, class_name in enumerate(CLASS_NAMES):
        print(f'{class_name}: Image-AUC: {img_aucs[i]}, Pixel-AUC: {pix_aucs[i]}')
    print('Mean Image-AUC: {}'.format(np.mean(img_aucs)))
    print('Mean Pixel-AUC: {}'.format(np.mean(pix_aucs)))