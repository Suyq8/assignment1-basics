
import argparse

import numpy as np
import torch

from cs336_basics.bpe import Tokenizer
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.model import Transformer
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import cosine_learning_rate_schedule, data_loading, get_perplexity, gradient_clipping, save_checkpoint


def train(config, model, optimizer, train_data, valid_data, device):
    for epoch in range(config.num_epochs):
        model.train()

        lr = cosine_learning_rate_schedule(epoch, config.lr_min, config.lr_max, config.warmup_epoch, config.cosine_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        total_loss = 0.0
        for _ in range(config.steps_per_epoch):
            x_train, y_train = data_loading(train_data, config.batch_size, config.context_length, device)
            
            preds = model(x_train)
            loss = cross_entropy_loss(preds, y_train)
            total_loss += loss.item()
            loss.backward()
        gradient_clipping(model.parameters(), config.max_l2_norm)
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {total_loss / config.steps_per_epoch:.4f}, Perplexity: {get_perplexity(total_loss / config.steps_per_epoch):.4f}")

        # Validate the model
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            x_valid, y_valid = data_loading(valid_data, config.batch_size, config.context_length, device)

            preds = model(x_valid)
            loss = cross_entropy_loss(preds, y_valid)
            total_loss += loss.item()

        print(f"Validation Loss: {total_loss / config.steps_per_epoch:.4f}, Perplexity: {get_perplexity(total_loss / config.steps_per_epoch):.4f}")

        # Save checkpoint at specified intervals
        if (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, epoch, f"{config.checkpoint_dir}/model_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='Number of steps per epoch')
    parser.add_argument('--warmup_epoch', type=int, default=10, help='Number of warmup epochs')
    parser.add_argument('--cosine_epoch', type=int, default=10, help='Number of cosine epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr_max', type=float, default=0.01, help='Max learning rate')
    parser.add_argument('--lr_min', type=float, default=0.0001, help='Min learning rate')
    parser.add_argument('--max_l2_norm', type=float, default=1.0, help='Max L2 norm for gradient clipping')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Interval for saving checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--vocab_filepath', type=str, help='Path to the vocabulary file')
    parser.add_argument('--merge_filepath', type=str, help='Path to the merge file')
    parser.add_argument('--train_data_path', type=str, help='Path to the training data file')
    parser.add_argument('--valid_data_path', type=str, help='Path to the validation data file')

    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--context_length', type=int, default=512, help='Context length for transformer')

    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.95), help='Betas for AdamW optimizer')
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_filepath, args.merge_filepath, ["<|endoftext|>"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_model * 4,
        num_layers=args.num_layers,
        vocab_size=tokenizer.get_vocab_size(),
        context_length=args.context_length,
        device=device,
    )
    model.to(device)

    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    valid_data = np.memmap(args.valid_data_path, dtype=np.uint16, mode='r')

    optimizer = AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay, betas=args.betas)

    train(
        config=args,
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        valid_data=valid_data,
        device=device,
    )