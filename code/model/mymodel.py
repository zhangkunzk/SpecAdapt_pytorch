import timm
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models._registry import register_model
from transformers import BertModel

cache_d = './cache'

@register_model
def bert_base_uncased(
        pretrained=False,
        cache_dir=cache_d,
        **kwargs
):
    # 过滤掉不兼容的参数
    timm_kwargs = kwargs.copy()
    for key in ['pretrained_cfg', 'features_only', 'out_indices', 'out_features', 'global_pool', 'pretrained_cfg_overlay']:
        timm_kwargs.pop(key, None)
    if pretrained:
        model = BertModel.from_pretrained('bert-base-uncased', cache_dir=cache_dir, **timm_kwargs)
    else:
        model = BertModel(config=BertModel.config_class(), **timm_kwargs)
    return model
