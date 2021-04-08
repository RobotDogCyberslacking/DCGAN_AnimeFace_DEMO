from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path
import numpy as np
import torch
import codecs
#from  skimage import io
from PIL import Image

'''
def create_collate_fn(pad_index, max_token_len,max_seq_len):
    #dataset: fn, caps, encoder_outputs, caption_label
    #outputs: fn,encoder_outputs,encoder_masks,caps,label_length
    def collate_fn(dataset):
        ground_truth = {}
        tmp = []
        #grouped by fn and extend by cap
        for fn, imgs in dataset:
            feature = torch.tensor(encoder_outputs[0:max_token_len,:])
            len2patch = max_token_len - feature.shape[0]
            p2d = (0, 0, 0, len2patch)
            p1d = (0,len2patch)
            att = feature.new_ones((feature.shape[0]),dtype = torch.long)#token size
            att = F.pad(att, p1d, "constant", 0)
            feature = F.pad(feature, p2d, "constant", 0)
            #ground_truth[fn] = [c[:max_seq_len] for c in caption_label]
            for i in range(0,len(caps)):
                sent = torch.tensor(caps[i])
                seq_sent = torch.tensor(caption_label[i])
                seq_mask = seq_sent.new_ones(seq_sent.shape)

                p1d = (0,max_seq_len-seq_sent.shape[0])
                pad_mask = F.pad(seq_mask, p1d, "constant", 0)
                seq_sent = F.pad(seq_sent, p1d, "constant", 0)
                p1d = (0,max_token_len-sent.shape[0])
                pad_cap= F.pad(sent, p1d, "constant", 0)
                tmp.append([fn,feature.numpy(),att.numpy(),pad_cap.numpy(),seq_sent.numpy(),pad_mask.numpy()])
        dataset = np.asarray(tmp)
        #dataset.sort(key=lambda p: len(p[1]), reverse=True)
        fns, imgs= zip(*dataset)
        imgs= torch.LongTensor(imgs)

        return fns, imgs

    return collate_fn
'''
class Anime_faces(VisionDataset):
    def __init__(self,train, root, transform=None, target_transform=None,
            download=False):
        super(Anime_faces, self).__init__(root, transform=transform,
                            target_transform=target_transform)
        self.train = train  # training set or test set
        self.root = root
        self.file_list = os.listdir(root)

    def __getitem__(self, index):
        fn= self.file_list[index]
        img = Image.open(self.root + "/" + fn)
        #print(img)
        if self.transform is not None:
            img = self.transform(img)
        #print(img)
        return fn, img

    def __len__(self):
        return len(self.file_list)
