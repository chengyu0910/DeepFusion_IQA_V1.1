from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--dataroot', required=True, help='path to images')
        self.parser.add_argument('--restore_train', action='store_true',default=False, help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=int, default=0, help='which epoch to load? set to latest to use latest cached model')
        # deriveds type and paramters is fixed now, maybe used in future for better results.
        # self.parser.add_argument('--deriveds_type', type=str, default='clahe,lightch,logarithm', help='the deriveds types')
        # self.parser.add_argument('--deriveds_param', type=str, default='clahe,lightch,logarithm',
        #                          help='the deriveds parameters')
        self.parser.add_argument('--input_nc', type=int, default=12, help='# of input image channels')
        #data augment
        self.parser.add_argument('--resize', action='store_true',default=True, help='data augmentation in training')
        self.parser.add_argument('--crop', action='store_true', default=True,help='data augmentation in training')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--rotate', action='store_true', default=True,help='data augmentation in training')
        self.parser.add_argument('--flip', action='store_true', default=True,help='data augmentation in training')

        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--checkpoints_freq', type=int, default=1,help='epoches/checkpointer')

        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen, iters/checkpointer')
        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console, iters/checkpointer')
        self.parser.add_argument('--log_freq', type=int, default=100,help='frequency of recording the train loss to log text file')
        self.parser.add_argument('--plot_freq', type=int, default=20, help='frequency of showing ploting results on visdom, iters/checkpointer')

        # self.parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen, iters/checkpointer')
        # self.parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console, iters/checkpointer')
        # self.parser.add_argument('--log_freq', type=int, default=1,help='frequency of recording the train loss to log text file')
        # self.parser.add_argument('--plot_freq', type=int, default=1, help='frequency of showing ploting results on visdom, iters/checkpointer')


        self.parser.add_argument('--grad_clip', type=float, default=0.08, help='gradient clipping, if not use it ,set to -1')

        self.parser.add_argument('--batchSize', type=int, default=10, help='input batch size')
        self.parser.add_argument('--batchloader_shuffle', action='store_true', default=True,help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--batchloader_nThreads', default=1, type=int, help='# threads for loading data in dataloader')
        self.parser.add_argument('--init_lr', type=float, default=0.00001, help='initial learning rate in adam optimizer')
        self.parser.add_argument('--lr_decay_nodes', type=str, default='[50,75,100,120]',help='the epoch node to decay learning rate')
        self.parser.add_argument('--lr_decay_mode', type=str, default='linear',help='the learning rate decay mode,e.g linear, exp')
        self.parser.add_argument('--lr_decay_param', type=float, default=0.25,help='the learning rate decay parameters')
        self.parser.add_argument('--max_epoch', type=int, default=150,help='which epoch to load? set to latest to use latest cached model')

        #loss weights
        self.parser.add_argument('--loss_weights', type=str,
                                 default="{'mse': 5, 'l1': 2, 'ssim': 1,'iqa_c1': 100, 'iqa_c2': 100, 'iqa_c3': 100,'iqa_s1': 10, 'iqa_s2': 0.01}",
                                 help='weights of different types loss')


        # self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        # self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        # self.parser.add_argument('--niter', type=int, default=245, help='# of iter at starting learning rate')
        # self.parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay earning rate to zero')
        # self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        # self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        # self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        # self.parser.add_argument('--lambda_Content', type=float, default=0.1, help='weight for content loss')
        # self.parser.add_argument('--lambda_perceptual', type=float, default=0.1, help='weight for perceptual loss')
        # self.parser.add_argument('--lambda_TV', type=float, default=0.1, help='weight for total variance loss')
        # self.parser.add_argument('--lambda_extra', type=float, default=0.1, help='weight for loss from extra input')
        # self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        # self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        self.parser.add_argument('--iqa_param_path', type=str, default='./Trancated.mat', help='the path of iqa net parameter for iqa loss supervision')
        # self.isTrain = True
