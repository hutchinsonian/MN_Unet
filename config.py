import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--upper', type=int, default=500, help='up threshold')
parser.add_argument('--lower', type=int, default=-1000, help='low threshold')
parser.add_argument('--expand_slice', type=int, default=2, help='expend slice')
parser.add_argument('--min_slices', type=int, default=100, help='min slice')
parser.add_argument('--slice_down_scale', type=float, default=1)
parser.add_argument('--xy_down_scale', type=float, default=1)
parser.add_argument('--valid_rate', type=float, default=0)
parser.add_argument('--crop_size', type=int, default=100)
parser.add_argument('--processed_dataset_path', type=str, default='./processed_datasets_lesion')
parser.add_argument('--norm_factor', type=float, default='1000')
parser.add_argument('--early_stop', default=30, type=int, help='early stopping (default: 80)')
parser.add_argument('--local_rank', default=-1, type=int)

parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--output_dir', default='./lesion', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=32, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_shape', type=list,
                    default=[512, 512], help='input shape of network input')
parser.add_argument('--seed', type=int,
                    default=42, help='random seed')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

args = parser.parse_args()
