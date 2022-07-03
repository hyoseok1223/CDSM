from options.test_options import TestOptions


class crossTestOptions(TestOptions):

    def __init__(self):
        super(crossTestOptions, self).__init__()

    def initialize(self):
        super(crossTestOptions, self).initialize()
        self.parser.add_argument('--source_checkpoint_path', default='/home/work/NaverWebtoonSide/restyle-encoder/pretrained_models/restyle_psp_ffhq_encode.pt', type=str,
                                 help='source domain trained checkpoint path')
        self.parser.add_argument('--source_data_path', default='/home/work/NaverWebtoonSide/restyle-encoder/data/source', type=str,
                                 help='source domain trained checkpoint path')
        self.parser.add_argument('--target_checkpoint_path', default='./experiment/restyle_psp_ffhq_encode/checkpoints/iteration_10000.pt', type=str,
                                 help='target domain trained checkpoint path')
        self.parser.add_argument('--target_data_path', default='/home/work/NaverWebtoonSide/restyle-encoder/data/target', type=str,
                                 help='source domain trained checkpoint path')
        self.parser.add_argument('--load_numpy', action='store_true',
                                 help='load latent codes from npy files')
        self.parser.set_defaults(load_numpy=False)
        self.parser.add_argument('--m', default=6, type=int,
                                 help='Style mixing level which is W+ layer index')
        self.parser.add_argument('--k_sampling', default=50, type=int,
                                 help='Random k( default : 50) sampling from specific character ID cartoon dataset')
        self.parser.add_argument('--layer_swap_resolution', default=32, type=int,
                                 help='resolution level which is swapped to fine-tunned generator')
        self.parser.add_argument('--out_path', default='../result.jpg', type=str,
                                 help='out image path')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
