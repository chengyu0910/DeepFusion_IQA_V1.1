
def create_model(opt):
    model = None
    print('模型:')
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'refinenet':
        from .refine_net import RefineNetModel
        model = RefineNetModel()

    elif opt.model == 'multifusion':
        from .gatednet import DeepFusionNet
        model = DeepFusionNet()

    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'recon':
        assert(opt.dataset_mode == 'depth')
        from .recon_model import ReconModel
        model = ReconModel()
    elif opt.model == 'recon_cont':
        assert(opt.dataset_mode == 'depth')
        from .recon_content_model import ReconContModel
        model = ReconContModel()      
    elif opt.model == 'gannormal':
        assert(opt.dataset_mode == 'unaligned')
        from .gan_normal import gannoromalModel
        model = gannoromalModel()                
    elif opt.model == 'disentangled':
        assert(opt.dataset_mode == 'depth')
        from .disentangled_model import DisentangledModel
        model = DisentangledModel()
    elif opt.model == 'disentangled2':
        assert(opt.dataset_mode == 'depth')
        from .disentangled_model2 import DisentangledModel
        model = DisentangledModel()
    elif opt.model == 'disentangled_LB':
        assert(opt.dataset_mode == 'depth')
        from .disentangled_LB import DisentangledLBModel
        model = DisentangledLBModel()
    elif opt.model == 'disentangled_LB_old':
        assert(opt.dataset_mode == 'depth')
        from .disentangled_LB_old import DisentangledLBModel
        model = DisentangledLBModel()
    elif opt.model == 'disentangled_extra':
        assert(opt.dataset_mode == 'depth')
        from .disentangled_extra import DisentangledExtraModel
        model = DisentangledExtraModel()
    elif opt.model == 'disentangled_multi':
        assert(opt.dataset_mode == 'depth')
        from .disentangled_multi import DisentangledMultiModel
        model = DisentangledMultiModel()
    elif opt.model == 'disentangled_final':
        assert(opt.dataset_mode == 'depth')
        from .disentangled_final import DisentangledModel
        model = DisentangledModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'debug':
        from .debug import DebugModel
        model = DebugModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
