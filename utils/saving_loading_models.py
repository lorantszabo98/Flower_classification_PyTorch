import os
import torch


def save_model(save_directory, model,  number_of_epochs, mode='default',  additional_text='', augmentation=''):
    os.makedirs(save_directory, exist_ok=True)
    model_name = model.__class__.__name__
    if mode == "feature_extractor":
        torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}{additional_text}_model_feature_extractor_epochs_{number_of_epochs}{augmentation}.pth"))
    elif mode == "fine_tuning":
        torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}{additional_text}_model_fine_tuned_epochs_{number_of_epochs}{augmentation}.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_directory, f"{model_name}{additional_text}_model_epochs_{number_of_epochs}{augmentation}.pth"))


def load_model(load_directory, model, number_of_epochs, mode='default',  additional_text='', augmentation=''):
    model_name = model.__class__.__name__
    if mode == "feature_extractor":
        model.load_state_dict(torch.load(os.path.join(load_directory, f"{model_name}{additional_text}_model_feature_extractor_epochs_{number_of_epochs}{augmentation}.pth"), map_location=torch.device('cpu')))
    elif mode == "fine_tuning":
        model.load_state_dict(torch.load(os.path.join(load_directory, f"{model_name}{additional_text}_model_fine_tuned_epochs_{number_of_epochs}{augmentation}.pth"), map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(os.path.join(load_directory, f"{model_name}{additional_text}_model_epochs_{number_of_epochs}{augmentation}.pth"), map_location=torch.device('cpu')))