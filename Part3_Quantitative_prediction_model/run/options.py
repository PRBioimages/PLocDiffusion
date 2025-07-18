import argparse
import os


workdirec = './'
loss_names = ['FocalSymmetricLovaszHardLogLoss']


class BaseOptions():
    def __init__(self):
        self.namestr = '_cynp'
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--out_dir', type=str, default=self.namestr, help='current work directory')
        self.parser.add_argument('--workers', default=3, type=int, help='number of data loading workers (default: 3)')
        self.parser.add_argument('--learning_rate', type=float, default=1e-5, help='initial learning rate for adam')
        self.parser.add_argument('--d_learning_rate', type=float, default=4e-5, help='initial learning rate for adam')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr_decay_iter', type=int, default=150000,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--d_lr_decay_iter', type=int, default=100000,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--lr_min_iter', type=int, default=500000,
                                 help='the iter set for min learning rate(1e-4) ')
        self.parser.add_argument('--num_heads', type=int, default=2,
                                 help='number of attention heads')

        self.parser.add_argument('--working_directory', type=str,
                                 default='/home/yywang/Classfication/slocdmganBestModels/',
                                 help='current work directory')
        self.parser.add_argument('--num_class', type=int, default=2, help='number of classes')
        self.parser.add_argument('--height', type=int, default=256, help='height of image')
        self.parser.add_argument('--width', type=int, default=256, help='width of image')
        self.parser.add_argument('--gan_noise', type=float, default=0.01, help='injection noise for the GAN')

        self.parser.add_argument('--batch_size', type=int, default=5,
                                 help='input batch size')
        self.parser.add_argument('--hiddenz_size', type=int, default=16, help='size of the hidden z')
        self.parser.add_argument('--hiddenr_size', type=int, default=8, help='size of the hidden r')

        self.parser.add_argument('--noise_bool', action='store_true', default=False,
                                 help='add noise on all GAN layers or not')

        ##(******)
        self.parser.add_argument('--not_use_tanh', action='store_true', default=False)
        self.parser.add_argument('--no_use_freq', action='store_true', default=True)
        self.parser.add_argument('--z_emb_dim', type=int, default=256)
        self.parser.add_argument('--patch_size', type=int, default=1, help='Patchify image into non-overlapped patches')
        self.parser.add_argument('--image_size', type=int, default=256, help='height of image')
        self.parser.add_argument('--num_channels_dae', type=int, default=128,
                                 help='number of initial channels in denosing model')
        self.parser.add_argument('--ch_mult', nargs='+', type=int, default=[1, 2, 2, 2, 4], help='channel multiplier')
        self.parser.add_argument('--num_res_blocks', type=int, default=2, help='number of resnet blocks per scale')
        self.parser.add_argument('--attn_resolutions', default=(16,), nargs='+', type=int,
                                 help='resolution of applying attention')
        self.parser.add_argument('--dropout', type=float, default=0., help='drop-out rate')
        self.parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                                 help='always up/down sampling with conv')
        self.parser.add_argument('--conditional', action='store_false', default=True, help='noise conditional')
        self.parser.add_argument('--fir', action='store_false', default=False, help='FIR')
        self.parser.add_argument('--fir_kernel', default=[1, 3, 3, 1], help='FIR kernel')
        self.parser.add_argument('--skip_rescale', action='store_false', default=True, help='skip rescale')
        self.parser.add_argument('--resblock_type', default='ddpm',
                                 help='style of resnet block, choice in biggan and ddpm')
        self.parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                                 help='progressive type for output')
        self.parser.add_argument('--progressive_input', type=str, default='residual',
                                 choices=['none', 'input_skip', 'residual'], help='progressive type for input')
        self.parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                                 help='progressive combine method.')
        self.parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                                 help='type of time embedding')
        self.parser.add_argument('--fourier_scale', type=float, default=16., help='scale of fourier transform')
        self.parser.add_argument('--num_channels', type=int, default=4, help='channel of wavelet subbands')
        self.parser.add_argument('--num_channels_ref', type=int, default=8, help='channel of wavelet subbands')
        self.parser.add_argument('--nz', type=int, default=100)
        self.parser.add_argument('--n_mlp', type=int, default=3, help='number of mlp layers for z')
        self.parser.add_argument("--use_pytorch_wavelet", action="store_true")
        self.parser.add_argument("--use_fp16", action='store_true', default=True, help='use half precision')
        self.parser.add_argument("--current_resolution", type=int, default=128)
        self.parser.add_argument("--num_disc_layers", default=5, type=int)
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--t_emb_dim', type=int, default=256)
        self.parser.add_argument('--num_timesteps', type=int, default=50)
        self.parser.add_argument('--beta_min', type=float, default=0.1, help='beta_min for diffusion')
        self.parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
        self.parser.add_argument('--use_geometric', action='store_true', default=False)
        self.parser.add_argument('--centered', action='store_false', default=True, help='-1,1 scale')
        self.parser.add_argument('--lazy_reg', type=int, default=10, help='lazy regulariation.')
        self.parser.add_argument('--r1_gamma', type=float, default=2., help='coef for r1 reg')
        self.parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
        self.parser.add_argument('--in_channels', type=int, default=3, help='in channels')
        self.parser.add_argument('--arch', type=str, default='class_densenet121_dropout',
                                 help='model architecture(default:class_densenet121_dropout)')
        self.parser.add_argument('--loss', default='FocalSymmetricLovaszHardLogLoss', choices=loss_names, type=str,
                                 help='loss function: ' + ' | '.join(
                                     loss_names) + ' (deafault: FocalSymmetricLovaszHardLogLoss)')
        self.parser.add_argument('--clipnorm', default=1, type=int, help='clip grad norm')
        self.parser.add_argument('--scheduler', default='Adam45', type=str, help='scheduler name')


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
        self.parser.add_argument('--no_display_img', action='store_true', help='specified if no dispaly')
        self.parser.add_argument('--max_epoch', type=int, default=61,
                                 help='max epoch for total training')
        self.parser.add_argument('--save_step', type=int, default=3,
                                 help='save model per #save_step epoch')
        self.parser.add_argument('--distill', type=bool, default=False, help='use knowledge distillation or not')
        # DivCo related
        self.parser.add_argument('--featnorm', type=str, default=True, help='whether featnorm')
        self.parser.add_argument('--radius', type=float, default=0.01, help='positive sample - distance threshold')
        self.parser.add_argument('--tau', type=float, default=1.0, help='temperature')
        self.parser.add_argument('--num_negative', type=int, default=1,
                                 help='number of latent negative samples')
        # gamma parameters
        self.parser.add_argument('--gamma_genMSE', type=float, default=1, help='Content Loss for Generator')
        self.parser.add_argument('--gamma_genL1', type=float, default=1, help='Adversarial Loss for Generator')
        self.parser.add_argument('--gamma_genLabel', type=float, default=1, help='Label Loss for Generator')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt


class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='specified the dir of saved models for resume the training')
        self.parser.add_argument('--comb', type=str, default='CyNp', help='phase for dataloading')
        self.parser.add_argument('--suffix', type=str, default='U2La_', help='phase for dataloading')
        self.parser.add_argument('--num', type=int, default=6, help='number of outputs per image')
        self.parser.add_argument('--sampledir', type=str, default=workdirec + 'sampledir' + self.namestr, \
                                 help='Stored evaluated txt for Testing')

        self.parser.add_argument('--test_MetricSimp', type=str,
                                 default=os.path.join(workdirec + 'sampledir' + self.namestr, \
                                                      'test_metricSimp' + self.namestr + '.csv'), \
                                 help='Stored evaluated txt for Testing')
        self.parser.add_argument('--test_Metric', type=str,
                                 default=os.path.join(workdirec + 'sampledir' + self.namestr, \
                                                      'test_metric' + self.namestr + '.csv'), \
                                 help='Stored evaluated txt for Testing')
        self.parser.add_argument('--Mul_Label_GenerateImagedir', type=str,
                                 default=workdirec + 'save_image_MulLabel' + self.namestr, help='saves results here.')
        self.parser.add_argument('--Mul_Samer_GenerateImagedir', type=str,
                                 default=workdirec + 'save_image_SamerGen' + self.namestr, help='saves results here.')
        self.parser.add_argument('--gifdir', type=str,
                                 default=workdirec + 'save_image_gif' + self.namestr,
                                 help='saves results here.')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        # set irrelevant options
        self.opt.dis_scale = 3
        self.opt.dis_norm = 'None'
        self.opt.dis_spectral_norm = False
        return self.opt