import torch
import torchvision

torchvision_list = ['inception_v3']
timm_list = ['dm_nfnet_f0', 'repvgg_b3g4', 'resnetv2_101x1_bitm']


def get_model(model_name: str, num_classes: int, pretrained: bool = True, **kwargs):
    if model_name in torchvision_list:
        model = getattr(torchvision.models, model_name)(pretrained=pretrained, **kwargs)
        if isinstance(model, torchvision.models.Inception3):
            if model.AuxLogits is not None:
                assert isinstance(model.AuxLogits, torchvision.models.inception.InceptionAux)
                in_features = model.AuxLogits.fc.in_features
                model.AuxLogits.fc = torch.nn.Linear(in_features, num_classes)
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, num_classes)
        else:
            raise NotImplementedError('{} {}'.format(type(model), model_name))
    elif model_name in timm_list:
        import timm
        model = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        if isinstance(model, timm.models.resnetv2.ResNetV2):
            if isinstance(model.head.fc, torch.nn.Conv2d):
                num_features = model.head.fc.in_channels
                model.head.fc = timm.models.layers.classifier._create_fc(num_features, num_classes, use_conv=True)
            else:
                raise NotImplementedError('{} {}'.format(type(model), model_name))
        elif isinstance(model, timm.models.nfnet.NormFreeNet):
            if isinstance(model.head.fc, torch.nn.Linear):
                num_features = model.head.fc.in_features
                model.head.fc = timm.models.layers.classifier._create_fc(num_features, num_classes, use_conv=False)
            else:
                raise NotImplementedError('{} {}'.format(type(model), model_name))
        elif isinstance(model, timm.models.byobnet.ByobNet):
            model.reset_classifier(num_classes)
        else:
            raise NotImplementedError('{} {}'.format(type(model), model_name))
    else:
        raise ValueError(model_name)
    return model
