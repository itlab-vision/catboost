import torch
import torchvision
from datasets.pascal_voc import VocDataset

def collate(batch):
    '''
    :batch:
    :return:
    images : (tensor)
    targets : (list) [(tensor), (tensor)]
    '''
    targets = []
    images = []

    for x in batch:
        targets.append(torch.from_numpy(x[1]))
        images.append(x[0])

    return torch.stack(images, 0), torch.stack(targets, 0)


class VOC(object):
    def __init__(self,
                 batch_size,
                 year='2007'):

        self.classes = 20
        self.batch_size = batch_size

        self.img_path = './datasets/voc/trainval/VOC{}/JPEGImages'.format(year)
        self.ann_path = './datasets/voc/trainval/VOC{}/Annotations'.format(year)
        self.spl_path = './datasets/voc/trainval/VOC{}/ImageSets/Main'.format(year)

        self.train_path = './datasets/voc/trainval/VOC{}'.format(year)
        self.test_path = './datasets/voc/test/VOC{}'.format(year)

    def get_loader(self, transformer, datatype):
        if datatype == 'train' or datatype == 'val' or datatype == 'trainval':
            path = self.train_path
        elif datatype == 'test':
            path = self.test_path
        else:
            AssertionError("[ERROR] Invalid path")

        custom_voc = VocDataset(path,
                                dataType=datatype,
                                transformer=transformer)

        custom_loader = torch.utils.data.DataLoader(
            dataset=custom_voc,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate)

        return custom_loader
