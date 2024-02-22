import os
import torch


def save_model(save_directory, model, mode='default'):
    os.makedirs(save_directory, exist_ok=True)
    model_name = model.__class__.__name__
    if mode == "feature_extractor":
        torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}_model_feature_extractor.pth"))
    elif mode == "fine_tuning":
        torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}_model_fine_tuned.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}_model.pth"))


def load_model(load_directory, model, mode='default'):
    model_name = model.__class__.__name__
    if mode == "feature_extractor":
        model.load_state_dict(torch.load(os.path.join(load_directory, f"{model_name}__model_feature_extractor_epochs_25_aug.pth"), map_location=torch.device('cpu')))
    elif mode == "fine_tuning":
        model.load_state_dict(torch.load(os.path.join(load_directory, f"{model_name}_large_model_fine_tuned_epochs_25_aug.pth"), map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(os.path.join(load_directory, f"{model_name}_model_epochs_25_aug.pth"), map_location=torch.device('cpu')))