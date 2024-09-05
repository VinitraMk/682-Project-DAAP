import torchvision

def get_model(model_name = "resnet18"):
    if model_name.lower() == "resnet50":
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        model = torchvision.models.resnet50(weights=weights)
        model_id2name = weights.meta["categories"]
    elif model_name.lower() == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        model = torchvision.models.resnet18(weights=weights)
        model_id2name = weights.meta["categories"]
    elif model_name.lower() == "inception":
        weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
        model = torchvision.models.inception_v3(weights=weights)
        model_id2name = weights.meta["categories"]
    
    #model.eval()
    return model, model_id2name