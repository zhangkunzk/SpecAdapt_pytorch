import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import types


class CheckpointModule(nn.Module):
    '''
    将将给定的tar_module包裹一个检查点模块返回。V1版本，不支持调用tar_module的其他成员
    '''

    def __init__(self, tar_module):
        super(CheckpointModule, self).__init__()
        self.tar_module = tar_module

    def forward(self, *args, **kwargs):
        # 使用checkpoint对模型的forward计算进行操作
        x = checkpoint.checkpoint(self.tar_module, *args, **kwargs)
        return x


class CheckPointModulePlus(nn.Module):
    '''
    将将给定的tar_module包裹一个检查点模块返回。V2版本，可以调用tar_module的其他成员
    '''

    def __init__(self, tar_module):
        super().__init__()
        self.tar_module = tar_module

    def forward(self, *args, **kwargs):
        # 使用checkpoint对模型的forward计算进行操作
        x = checkpoint.checkpoint(self.tar_module, *args, **kwargs)
        return x

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.tar_module, name):
                return getattr(self.tar_module, name)
            else:
                raise AttributeError(f"'A' object has no attribute '{name}'")


def module_add_checkpoints(model, target_module_name: str):
    """
    遍历模型，若发现目标模块，将其包裹检查点模块。原地操作，无返回值。
    args:
        model: 模型实例, nn.Module
        target_module_name: 想包裹检查点的类的名字，str
    example:
        module_add_checkpoints(model,SwinTransformerBlock)
    """
    for name, child_module in model.named_children():
        #if isinstance(child_module, target_module_class):
        if child_module.__class__.__name__ == target_module_name:
            setattr(model, name, CheckPointModulePlus(child_module))


        elif isinstance(child_module, nn.Module):
            module_add_checkpoints(child_module, target_module_name)


def traverse_model(model):
    """
    调试用，输出模型各层结构
    """
    for name, module in model.named_children():
        print(name, module)
        if isinstance(module, nn.Module):
            traverse_model(module)
