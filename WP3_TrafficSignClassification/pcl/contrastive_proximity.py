import torch
import torch.nn as nn
PCL_custom_labels_gtsrb={
    0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0,
    9:1, 10:1, 15:1, 16:1,
    11:2, 18:2, 19:2, 20:2, 21:2, 22:2, 23:2, 24:2, 25:2, 26:2, 27:2, 28:2, 29:2, 30:2, 31:2,
    32:3, 41:3, 42:3,
    33:4, 34:4, 35:4, 36:4, 37:4, 38:4, 39:4, 40:4,
    12:5, 13:6, 14:7, 17:8
}

PCL_custom_labels_imagenet={
    0:0, 1:0, 2:0,
    3:1, 4:1, 5:1, 6:1,
    7:2, 8:2, 9:2, 10:2,
    11:3, 12:3, 13:3, 14:3
}
PCL_custom_labels_cure_tsd={
    1:0, 2:0,
    3:1, 4:1,
    8:2, 9:2, 10:2
}

single_classes_gtsrb=[12, 13, 14, 17]
single_classes_imagenet=[]
single_classes_cure_tsd=[0, 5, 6, 7, 11, 12, 13]

cluster_prox_gtsrb={
        0:[0, 1, 2, 3, 4, 5, 6, 7, 8],
        1:[9,10,15,16],
        2:[11,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
        3:[32,41,42],
        4:[33,34,35,36,37,38,39,40]
    }
cluster_prox_imagenet={
        0:[0, 1, 2],
        1:[3, 4, 5, 6],
        2:[7, 8, 9, 10],
        3:[11, 12, 13, 14]
    }

cluster_prox_cure_tsd={
        0:[1, 2],
        1:[3, 4],
        2:[8, 9, 10]
    }



class Con_Proximity(nn.Module):

    def __init__(self, num_classes=100, feat_dim=1024, use_gpu=True, dataset='gtsrb'):
        super(Con_Proximity, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())#100 x feats- for 100 centers
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        if dataset=='gtsrb':
            self.PCL_custom_labels=PCL_custom_labels_gtsrb
            self.single_classes=single_classes_gtsrb
            self.cluster_prox=cluster_prox_gtsrb
        elif dataset=='imagenet':
            self.PCL_custom_labels = PCL_custom_labels_imagenet
            self.single_classes = single_classes_imagenet
            self.cluster_prox = cluster_prox_imagenet
        elif dataset=='cure_tsd':
            self.PCL_custom_labels = PCL_custom_labels_cure_tsd
            self.single_classes = single_classes_cure_tsd
            self.cluster_prox = cluster_prox_cure_tsd

    def forward(self, x, labels, has_custom_classes=False, custom_classes=[], custom_distance=False, custom_conprox_center=0, custom_dist_rate=10):

        batch_size = x.size(0)
        if has_custom_classes:
            centers=self.centers-custom_conprox_center
        else:
            centers=self.centers
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, centers.t(),beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if has_custom_classes:
            classes=torch.from_numpy(custom_classes)
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):

            k= mask[i].clone().to(dtype=torch.int8)

            k= -1* k +1

            kk= k.clone().to(dtype=torch.bool)
            if custom_distance and not labels[i][0].item() in self.single_classes:
                ones=torch.ones(self.num_classes).cuda()
                cluster_id=self.PCL_custom_labels[labels[i][0].item()]
                for _class in classes:
                    if _class.item() in self.cluster_prox[cluster_id]:
                        ones[_class]*=custom_dist_rate
                value=distmat[i]
                value*=ones
                value=value[kk]
            else:
                value = distmat[i][kk]

            value = value.clamp(min=1e-12, max=1e+12) # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()

        return loss