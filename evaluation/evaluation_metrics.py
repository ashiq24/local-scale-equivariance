import torch
import numpy as np
import evaluate
from functools import partial
def get_metric(params):
    '''
    return the metric function
    '''
    metric = None
    if getattr(params, 'task', 'classification') == 'classification':
        metric = acc_metric_classification
    elif getattr(params, 'task', 'classification') == 'segmentation':
        '''
        num_lables: number of classes (inlcuding background)
        ignore_index: the index to ignore in the evaluation (ingore the background class)
        '''
        metric = segmentation_metric(num_classes=params.num_classes, ignore_index=params.ignore_index)
    else:
        raise ValueError('Task not supported')
    
    return metric
def acc_metric_classification(output, target):
    '''
    return the number of correct predictions
    '''
    _, pred = torch.max(output, dim=1)
    correct = torch.sum(pred == target)
    return correct 

class segmentation_metric():
    def __init__(self, num_classes, ignore_index):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.metric = evaluate.load('mean_iou')
    
    def __call__(self, output, target):
        ## apply softmax to the output
        output = torch.softmax(output, dim=1)
        ## get segmentation predictions
        _, pred = torch.max(output, dim=1)
        return self.metric.compute(predictions=pred,references=target, num_labels=self.num_classes,\
                                    ignore_index=self.ignore_index)['mean_iou'] * target.size(0)
