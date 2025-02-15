

import os
import numpy as np
import torch
import wandb

from ece496b_basics.gelu import Transformer_LM
from ece496b_basics.loop import data_loader
from ece496b_basics.optimize import AdamW, cross_entropy


def wand():
    wandb.init(project="Transformer-LM, HW1")

    # Configuration
    config = {
        "batch_size": 32,
        "learning_rate": 3e-4,
        "num_epochs": 10,
        "seq_length": 128,
        "gradient_accumulation_steps": 4,
        "checkpoint_path": "./checkpoints",
        "log_interval": 10
    }

    wandb.config.update(config)

    # Ensure checkpoint path exists
    os.makedirs(config["checkpoint_path"], exist_ok=True)

# Instantiate model
weights = dict[str, torch.FloatTensor]
model = Transformer_LM(vocab_size=256, context_length=10, d_model=2, num_layers=5,
                       num_heads=6, d_ff=7, attn_pdrop=0.5, residual_pdrop=0.6, weights=weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and Optimizer
criterion = cross_entropy()
optimizer = AdamW

# Load training and validation data
arr = np.array()
train_loader = data_loader(arr, 4, 4, "cpu")
val_loader = data_loader(arr, 2,2,"cpu")

# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    model.train()
    
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)  
            loss = criterion(output.view(-1, output.shape[-1]), batch.view(-1))

            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # Log metrics
            if i % config["log_interval"] == 0:
                wandb.log({"epoch": epoch, "step_loss": loss.item()})
                print(f"Epoch [{epoch+1}/{config['num_epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        # Validation Step
        val_loss = validate_model(model, val_loader, criterion)
        wandb.log({"epoch": epoch, "train_loss": epoch_loss / len(train_loader), "val_loss": val_loss})
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

        # Save Checkpoint
        checkpoint_path = os.path.join(config["checkpoint_path"], f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# Validation Loop
def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output.view(-1, output.shape[-1]), batch.view(-1))
            val_loss += loss.item()
    model.train()
    return val_loss / len(val_loader)

# Run Training
train_model(model, train_loader, val_loader, criterion, optimizer)