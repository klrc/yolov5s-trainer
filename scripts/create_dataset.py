import random
import os
import shutil
from pycocotools.coco import COCO
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


class Bbox():
    def __init__(self, cid, name, supercat, x, y, w, h,):
        self.cid = cid
        self.name = name
        self.supercat = supercat
        self.x = x
        self.y = y
        self.w = w
        self.h = h


class Metadata():
    def __init__(self, id, filename, width, height, annotations):
        self.id = id
        self.filename = filename
        self.width = width
        self.height = height
        self.annotations = annotations


class COCOX():
    target_classes = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck')

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.data = {
            'train2017': self.load_data('train2017'),
            'val2017': self.load_data('val2017'),
        }

    def load_data(self, dataset):
        coco = COCO(f'{self.root}/annotations/instances_{dataset}.json')
        categories = coco.dataset['categories']
        self.__cid2name = {x['id']: x['name'] for x in categories}
        self.__cid2supercat = {x['id']: x['supercategory'] for x in categories}
        for cid in (5, 7, 9):
            self.__cid2supercat[cid] = 'fake_vehicle'

        iids = []
        for cid in [x['id'] for x in categories]:
            for iid in tqdm(coco.getImgIds(catIds=[cid])):
                iids.append(iid)

        return [self.read_meta(coco, iid) for iid in tqdm(set(iids))]

    def read_meta(self, coco, iid):
        img_meta = coco.loadImgs(iid)[0]
        filename = img_meta['file_name']
        width = img_meta['width']
        height = img_meta['height']
        annotations = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=None)):
            bbox = ann['bbox']
            cid = ann['category_id']
            name = self.__cid2name[cid]
            supercat = self.__cid2supercat[cid]
            x = (bbox[0] + bbox[2]/2.0 - 1)/width
            y = (bbox[1] + bbox[3]/2.0 - 1)/height
            w = bbox[2]/width
            h = bbox[3]/height
            annotations.append(Bbox(cid, name, supercat, x, y, w, h))
        return Metadata(iid, filename, width, height, annotations)


def export(cocox, root):
    print('exporting...')
    # clean target root
    print(f'clean dir at {root}...')
    if os.path.exists(root):
        shutil.rmtree(root)

    # # load true cid mapping
    # true_cid = dict()
    # with open('data/coco-ref/coco-labels.txt', 'r') as f:
    #     for i, line in enumerate(f.readlines()):
    #         line = line.strip()
    #         true_cid[line] = i

    # build new dataset
    for dataset in cocox.data.keys():
        os.makedirs(f'{root}/images/{dataset}')
        os.makedirs(f'{root}/labels/{dataset}')
        for m in tqdm(cocox.data[dataset]):
            # save image
            src_path = f'{cocox.root}/{dataset}/{m.filename}'
            target_path = f'{root}/images/{dataset}/{m.filename}'
            shutil.copy(src_path, target_path)
            with open(f'{root}/{dataset}.txt', 'a') as f:
                f.write(f'./images/{dataset}/{m.filename}\n')
            # save annotations
            target_path = f'{root}/labels/{dataset}/{m.filename.replace(".jpg",".txt")}'
            with open(target_path, 'w') as f:
                for bbox in m.annotations:
                    # line = (true_cid[bbox.name], bbox.x, bbox.y, bbox.w, bbox.h)
                    line = (bbox.cid, bbox.x, bbox.y, bbox.w, bbox.h)
                    f.write(" ".join([str(x) for x in line]) + '\n')

    # finish exporting, return with root path
    return root


def get_class_hist(cocox, dataset='train2017'):
    x = []
    for m in cocox.data[dataset]:
        for b in m.annotations:
            x.append(b.cid)
    sns.histplot(x, kde=True)
    plt.show()


def get_instance_dist(cocox, dataset='train2017'):
    ret = {}
    for dataset in cocox.data.keys():
        for m in cocox.data[dataset]:
            for b in m.annotations:
                if b.supercat not in ret:
                    ret[b.supercat] = 1
                else:
                    ret[b.supercat] += 1
    return ret


def prune_func(cocox, __dist, dataset='train2017'):

    print('pruning coco dataset..')

    ark = []
    # bg_ark = []
    coco_p1_cid = {name: i for i, name in enumerate(['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck'])}

    for m in tqdm(cocox.data[dataset]):
        anno_ark = []
        num_human = 0
        for b in m.annotations:
            if b.supercat in ('person', 'vehicle'):
                b.cid = coco_p1_cid[b.name]
                anno_ark.append(b)
            if b.supercat == 'person':
                num_human += 1
        m.annotations = anno_ark

        # if len(anno_ark) == 0:
        #     bg_ark.append(m)
        if len(anno_ark) > 0 and num_human/len(anno_ark) > 0.8:
            continue
        else:
            ark.append(m)

    # print(f'{len(ark)} target images, {len(bg_ark)} background images..')
    # bg_rate = 0.8
    # bg_size = len(ark)/(1-bg_rate)*bg_rate
    # bg_ark = random.choices(bg_ark, k=int(bg_size))
    # ark.extend(bg_ark)

    cocox.data[dataset] = ark


if __name__ == "__main__":
    dataset = COCOX('../datasets/coco')
    # get_class_hist(dataset)
    __rawsize = sum([len(x) for x in dataset.data.values()])

    __dist = get_instance_dist(dataset)
    prune_func(dataset, __dist, 'train2017')
    prune_func(dataset, __dist, 'val2017')
    get_class_hist(dataset)
    __newsize = sum([len(x) for x in dataset.data.values()])
    print(f"{__newsize/__rawsize:.2%}({__newsize}/{__rawsize}) data preserved")

    export(dataset, '../datasets/coco-l')
