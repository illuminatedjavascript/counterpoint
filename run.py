import argparse
import torch
from models import model, dataset, train, sample


def main(argp):
    if argp.function == 'train':
        train_model(argp.model)
    elif argp.function == 'sample':
        sample_model(argp.model, argp.save)


def train_model(model_save_path: str, checkpoint=False):
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        print(f'Using cuda device {torch.cuda.get_device_name()}')
    else:
        print(f"Using {device} device")

    # Load
    chorale_dataset = dataset.ChoraleDataset('./data/processed/jsb32slide.json', device=device) # Change manually 
    model_config = model.ChoraleBertConfig(chorale_dataset)
    chorale_model = model.ChoraleBertModel(model_config).to(device)

    if checkpoint == True:
        chorale_model.load_state_dict(torch.load(model_save_path))

    print(f'Initialised model with {sum(p.numel() for p in chorale_model.parameters() if p.requires_grad)} parameters.')
    
    # Train
    trainer = train.Trainer(chorale_model, chorale_dataset, 1e-3)
    trainer.train(300, 64)
    
    # Save
    torch.save(chorale_model.state_dict(), model_save_path)


def sample_model(model_load_path: str, sample_save_path: str):
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        print(f'Using cuda device {torch.cuda.get_device_name()}')
    else:
        print(f"Using {device} device")
    
    # Load trained model
    chorale_dataset = dataset.ChoraleDataset('./data/processed/jsb32seq.json', device=device) # Change manually 
    model_config = model.ChoraleBertConfig(chorale_dataset)
    chorale_model = model.ChoraleBertModel(model_config).to(device)
    chorale_model.load_state_dict(torch.load(model_load_path))
    
    sampler = sample.Sampler(chorale_model, chorale_dataset, sample_save_path)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('function',
        choices=['train', 'sample'])
    argp.add_argument('-p', '--model',
        help='save/load path for model params',
        default="model_params_32.txt")
    argp.add_argument('-s', '--save',
        help='save path for midi samples',
        default="./data/output/")
    argp = argp.parse_args()
    
    main(argp)