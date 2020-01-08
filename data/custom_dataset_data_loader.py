import torch.utils.data
from data.base_data_loader import BaseDataLoader
from data.custom_dataset import GatedNetDataset

def CreateDataset(opt):
    # dataset = None
    # if opt.dataset_mode == 'aligned':
    #     from data.aligned_dataset import AlignedDataset
    #     dataset = AlignedDataset()
    # elif opt.dataset_mode == 'refine':
    #     from data.custom_dataset import refineDataset
    #     dataset = refineDataset()
    #
    # elif opt.dataset_mode == 'multifusion':
    #     from data.custom_dataset import GatedNetDataset
    #     trainset = GatedNetDataset()
    #     testset = GatedNetDataset()
    #
    # elif opt.dataset_mode == 'unaligned':
    #     from data.unaligned_dataset import UnalignedDataset
    #     dataset = UnalignedDataset()
    # elif opt.dataset_mode == 'single':
    #     from data.single_dataset import SingleDataset
    #     dataset = SingleDataset()
    # elif opt.dataset_mode == 'depth':
    #     from data.depth_dataset import DepthDataset
    #     dataset = DepthDataset()
    # else:
    #     raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    trainset = GatedNetDataset()
    testset = GatedNetDataset()
    trainset.initialize(opt, mode='train')  # 读入图片 构成数据集
    testset.initialize(opt, mode='test')  # 读入图片 构成数据集
    print("trainset [%s] was created" % (trainset.name()))
    print("testset [%s] was created" % (testset.name()))
    return trainset, testset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.trainset, self.testset = CreateDataset(opt)#读入的数据集
        self.dataloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=opt.batchSize,
            shuffle= opt.batchloader_shuffle,
            num_workers=int(opt.batchloader_nThreads))
        self.testloader = torch.utils.data.DataLoader(
            self.testset,
            batch_size=opt.batchSize,
            shuffle= opt.batchloader_shuffle,
            num_workers=int(opt.batchloader_nThreads))

    def load_traindata(self):
        return self.dataloader
    def load_testdata(self):
        return self.testloader

    def __len__(self):
        return min(len(self.trainset), self.opt.max_dataset_size)
