import os
from lamr_eval.eval_MR_multisetup import COCOeval
from lamr_eval.coco import COCO


def validate(dt_path):
    annFile = '/path/to/output_dir/val_gt.json'

    mean_MR = []
    my_id_setup = []
    for id_setup in range(0, 4):
        cocoGt = COCO(annFile)
        cocoDt = cocoGt.loadRes(dt_path)
        imgIds = sorted(cocoGt.getImgIds())
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate(id_setup)
        cocoEval.accumulate()
        mean_MR.append(cocoEval.summarize_nofile(id_setup))
        my_id_setup.append(id_setup)
    return mean_MR


if __name__ == '__main__':
    input_path = 'path/to/output_dir/results_json/val_dt.json'
    print('hi')
    MRs = validate('path/to/output_dir/val_gt.json', input_path)
    print('[Reasonable: %.2f%%]\n[Reasonable_Small: %.2f%%]\n[Heavy: %.2f%%]\n[All: %.2f%%]'
          % (MRs[0] * 100, MRs[1] * 100, MRs[2] * 100, MRs[3] * 100))

    """
    Metrics
    We use the same protocol as in [1] for evaluation. As a numerical measure of the performance, log-average miss rate (MR) is computed by averaging over the precision range of [10e-2; 10e0] FPPI (false positives per image). 
    For detailed evaluation, we consider the following 4 subsets:
    1. 'Reasonable': height [50, inf]; visibility [0.65, inf]
    2. 'Reasonable_small': height [50, 75]; visibility [0.65, inf]
    3. 'Reasonable_occ=heavy': height [50, inf]; visibility [0.2, 0.65]
    4. 'All': height [20, inf]; visibility [0.2, inf]
    
    
    Reference
    [1] P. Dollar, C. Wojek, B. Schiele and P. Perona. Pedestrian Detection: An Evaluation of the State of the Art. TPAMI, 2012. 
    
    """
