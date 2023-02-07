import torch
from models import model, dataset, train, sample


def test():
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cuda':
        print(f'Using cuda device {torch.cuda.get_device_name()}')
    else:
        print(f"Using {device} device")

    # Load
    chorale_dataset = dataset.ChoraleDataset('./data/processed/jsb16seq.json')
    model_config = model.ChoraleBertConfig(chorale_dataset)
    chorale_model = model.ChoraleBertModel(model_config)
    
    # Train
    trainer = train.Trainer(chorale_model, chorale_dataset, 1e-4)
    trainer.train(200, 64)
    
    # Test
    chorale_model.eval()
    test_seq = ["<S>", '<M>', 60, '<M>', '<M>',
                "<T>", '<M>', 60, '<M>', '<M>', 
                "<T>", '<M>', 62, '<M>', '<M>', 
                "<T>", '<M>', 62, '<M>', '<M>', 
                "<T>", '<M>', 64, '<M>', '<M>', 
                "<T>", '<M>', 64, '<M>', '<M>', 
                "<T>", '<M>', 65, '<M>', '<M>', 
                "<T>", '<M>', 65, '<M>', '<M>', 
                "<T>", '<M>', 60, '<M>', '<M>', 
                "<T>", '<M>', 60, '<M>', '<M>', 
                "<T>", '<M>', 60, '<M>', '<M>', 
                "<T>", '<M>', 60, '<M>', '<M>', 
                "<T>", '<M>', 67, '<M>', '<M>', 
                "<T>", '<M>', 67, '<M>', '<M>', 
                "<T>", '<M>', 60, '<M>', '<M>', 
                "<T>", '<M>', 60, '<M>', '<M>', "<E>"]

    test_seq_enc = chorale_dataset._encode(test_seq).reshape(1, -1)
    res = chorale_model.forward(test_seq_enc)
    
    res_argmax = torch.argmax(res, dim=2)
    res = chorale_dataset._decode(res_argmax)

    # REMOVE
    utils.to_midi(res, chorale_dataset)
    # END

if __name__ == '__main__':
    test()