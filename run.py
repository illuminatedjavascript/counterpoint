import argparse
import torch
from models import model, dataset, train, sample


def main(argp):
    if argp.function == 'train':
        train_model(argp.model)
    elif argp.function == 'sample':
        sample_model(argp.model, argp.save)

def train_model(model_save_path: str, checkpoint: bool = False):
    """Entry point for training the model.
    Args:
        model_save_path: path where model parameters should be saved/loaded.
        checkpoint: If true, loads checkpoint located at model_save_path.
    """
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        print(f'Using cuda device {torch.cuda.get_device_name()}')
    else:
        print(f"Using {device} device")

    # Load
    chorale_dataset = dataset.ChoraleDataset('./data/processed/chorale32slide.json', device=device) # Change manually 
    fugue_dataset = dataset.ChoraleDataset('./data/processed/fugue16div32slide.json', device=device) # Change manually 
    
    model_config = model.ChoraleBertConfig(chorale_dataset) # Results in same config as fugue dataset
    bach_model = model.ChoraleBertModel(model_config).to(device)

    if checkpoint == True:
        bach_model.load_state_dict(torch.load(model_save_path))

    print(f'Initialised model with {sum(p.numel() for p in bach_model.parameters() if p.requires_grad)} parameters.')
    
    # Pre-train on chorales 
    trainer = train.Trainer(bach_model, chorale_dataset, 1e-3)
    trainer.train(200, 32)
    torch.save(bach_model.state_dict(), f'pretrain_{model_save_path}')
    
    # Train on fugues
    trainer = train.Trainer(bach_model, fugue_dataset, 1e-3)
    trainer.train(200, 32)
    torch.save(bach_model.state_dict(), f'finetune_{model_save_path}')


def sample_model(model_load_path: str, sample_save_path: str):
    """Produces samples of the model.
    Args:
        model_load_path: path to saved model parameters.
        sample_save_path: path to the directory where samples are saved.
    """
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        print(f'Using cuda device {torch.cuda.get_device_name()}')
    else:
        print(f"Using {device} device")
    
    # Load trained model
    chorale_dataset = dataset.ChoraleDataset('./data/processed/fugue16sep48slide.json', device=device) # Change manually 
    model_config = model.ChoraleBertConfig(chorale_dataset)
    chorale_model = model.ChoraleBertModel(model_config).to(device)
    chorale_model.load_state_dict(torch.load(model_load_path))
    
    # Sample
    sampler = sample.Sampler(chorale_model, chorale_dataset, sample_save_path)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('function',
        choices=['train', 'sample'])
    argp.add_argument('-p', '--model',
        help='save/load path for model params',
        default="16div32sep.txt")
    argp.add_argument('-s', '--save',
        help='save path for midi samples',
        default="./data/output/")
    argp = argp.parse_args()
    
    main(argp)