#!/usr/bin/env python3

import model.mymodel as mymodel
import timm

models = timm.list_models()
for item in models:
    if 'bert' in item:
        print(item)
        model = timm.create_model(item, pretrained=True)
        print(model)
        print('----------------------------------')
        for name, param in model.named_parameters():
            print(f"参数名: {name}")
            print(f"参数形状: {param.shape}")
            print(f"是否可训练: {param.requires_grad}")
            print("-" * 40)
                # 这里可以添加更多的测试代码
                # 例如，测试模型的前向传播等
                # inputs = torch.randn(1, 3, 224, 224)  # 示例输入
                # outputs = model(inputs)
                # print(outputs.shape)
        # 这里可以添加更多的测试代码
        # 例如，测试模型的前向传播等
        # inputs = torch.randn(1, 3, 224, 224)  # 示例输入
        # outputs = model(inputs)
        # print(outputs.shape)