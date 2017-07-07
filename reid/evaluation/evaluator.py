from __future__ import print_function, absolute_import
import time
import torch
from torch.autograd import Variable

from utils.meters import AverageMeter
from utils import to_numpy
from reid.evaluation import cmc, mean_ap, avepool_dismat


def pairwise_distance_tensor(x, metric=None):
    n = x.size(0)
    if metric is not None:
        x = metric.transform(x)
    dist = torch.pow(x, 2).sum(1) * 2
    dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
    return dist




def extract_seqfeature(cnn_model, rnn_model, data_loader, mode):
    print_freq = 50
    cnn_model.eval()
    rnn_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    allfeatures = 0
    allpids = 0
    allcamid = 0
    for i, (imgs, flows, pids, camid) in enumerate(data_loader):
        data_time.update(time.time() - end)
        #####
        imgs = Variable(imgs, volatile=True)
        flows = Variable(flows, volatile=True)
        out_feat = cnn_model(imgs, flows, mode)

        if mode == 'cnn_rnn' or mode == 'fixcnn_rnn':
            out_feat = rnn_model(out_feat)

        if i == 0:
            allfeatures = out_feat.data
            allpids = pids
            allcamid = camid
        else:
            allfeatures = torch.cat((allfeatures, out_feat.data), 0)
            allpids = torch.cat((allpids, pids), 0)
            allcamid = torch.cat((allcamid, camid), 0)

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return allfeatures, allpids, allcamid



def evaluate_seq(distmat, pids, camids, cmc_topk=(1, 5, 10), score_pool='ave_pool'):
    query_ids   =  to_numpy(pids)
    gallery_ids =  query_ids
    query_cams = to_numpy(camids)
    gallery_cams = query_cams
    distmat = distmat.cpu()

    if score_pool == 'ave_pool':
        distmat, query_ids, gallery_ids, query_cams, gallery_cams = \
        avepool_dismat(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    else:
        raise RuntimeError

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['allshots'][0]


class Evaluator(object):
    def __init__(self, cnn_model, rnn_model):
        super(Evaluator, self).__init__()
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model



    def evaluate(self, data_loader, mode):

        features, pids, camids = extract_seqfeature(self.cnn_model, self.rnn_model, data_loader, mode)
        distmat = pairwise_distance_tensor(features)
        return evaluate_seq(distmat, pids, camids)




