from .surrogate_model import Adapter, DEMAdapter
from .ada_swin import AdaSwinForImageClassification, AdaSwinForDensePrediction
from .ada_resnet import AdaResNetForImageClassification, AdaResNetForDensePrediction
from .ada_vit import AdaViTForImageClassification, AdaViTForDensePrediction
from .ada_dino import AdaDinov2ForImageClassification, AdaDinov2ForDensePrediction
from .ada_beit import AdaBeitForImageClassification, AdaBeitForDensePrediction
from .ada_deit import AdaDeiTForImageClassification, AdaDeiTForDensePrediction
from .canonicalizer_wrapper import CanonicalizeWrapper

def get_model(params):
    model_mapping = {
        'swin_transformer': (AdaSwinForImageClassification, AdaSwinForDensePrediction),
        'resnet': (AdaResNetForImageClassification, AdaResNetForDensePrediction),
        'vit': (AdaViTForImageClassification, AdaViTForDensePrediction),
        'dino': (AdaDinov2ForImageClassification, AdaDinov2ForDensePrediction),
        'beit': (AdaBeitForImageClassification, AdaBeitForDensePrediction),
        'deit': (AdaDeiTForImageClassification, AdaDeiTForDensePrediction),
    }
    
    if params.model not in model_mapping:
        raise ValueError(f'Model {params.model} not found')
    
    model_cls = model_mapping[params.model]
    task = getattr(params, 'task', 'classification')
    
    if params.model in ['cnn', 'adaptive_cnn']:
        model = model_cls(
            num_layers=params.num_layers,
            num_channels=params.num_channels,
            kernel_sizes=params.kernel_sizes,
            num_classes=params.num_classes
        )
    else:
        model_classification, model_segmentation = model_cls
        model = model_segmentation(params=params) if task == 'segmentation' else model_classification(params=params)

    if getattr(params, "do_cannonicalization", False) and not params.do_adaptation:
        model = CanonicalizeWrapper(
            model=model,
            num_layers=params.cannon_num_layers,
            num_channels=params.cannon_num_channels,
            kernel_sizes=params.kernel_sizes,
            task=task,
            unique_params_limit=params.unique_params_limit
        )
    
    return model

def get_surrogate_model(params):
    print("*********Initializing surrogate model*********")
    return DEMAdapter(params) if params.deep_equlibrium_mode else Adapter(params)

def get_dem_model(params):
    return DEMAdapter(params)