import torchvision

def get_model(model_name):
    if model_name.lower() == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        model_id2name = weights.meta["categories"]
        gradcam_target = [model.layer4[-1]]
    elif model_name.lower() == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        model_id2name = weights.meta["categories"]
        gradcam_target = [model.layer4[-1]]
    elif model_name.lower() == "vit":
        weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
        model = torchvision.models.vit_b_16(weights=weights)
        model_id2name = weights.meta["categories"]
        # gradcam_target = [model.encoder.layers[-1].ln_1]
        # gradcam_target = [model.encoder.layers[-2]]
        gradcam_target = [model.encoder.layers[-1].ln_1]
    elif model_name.lower() == "swin":
        weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1
        model = torchvision.models.swin_t(weights=weights)
        model_id2name = weights.meta["categories"]
        # gradcam_target = [model.encoder.layers[-1].ln_1]
        # gradcam_target = [model.encoder.layers[-2]]
        gradcam_target = [model.features[-1][-1].norm1]
        # print(model)
        # exit()
    elif model_name.lower() == "vgg":
        weights = torchvision.models.Swin_T_Weights.IMAGENET1K_V1
        model = torchvision.models.swin_t(weights=weights)
        model_id2name = weights.meta["categories"]
        # gradcam_target = [model.encoder.layers[-1].ln_1]
        # gradcam_target = [model.encoder.layers[-2]]
        gradcam_target = [model.features[-1][-1].norm1]
        # print(model)
        # exit()
    
    model.eval()
    return model, model_id2name, gradcam_target

