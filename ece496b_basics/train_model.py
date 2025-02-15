import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import os

from ece496b_basics.gelu import Transformer_LM
from ece496b_basics.loop import data_loader

# Ensure checkpoint directory exists
os.makedirs("./checkpoints", exist_ok=True)

#Initialize Weights & Biases
def initialize_wandb():
    wandb.init(project="Transformer-LM-HW1")

    config = {
        "batch_size": 16,  
        "learning_rate": 3e-4,
        "num_epochs": 5,
        "seq_length": 128,
        "gradient_accumulation_steps": 4,
        "checkpoint_path": "./checkpoints",
        "log_interval": 10
    }
    
    wandb.config.update(config)
    return config


#Load Pre-Tokenized Data
def load_data(batch_size, context_length, device):
    dataset = np.load("tinystories_tokens.npy")

    # Ensure dataset is long enough
    if len(dataset) < context_length + 1:
        raise ValueError(f"Dataset too short ({len(dataset)} tokens). Needs at least {context_length + 1} tokens.")

    return dataset

#Function to Initialize Random Weights
def initialize_random_weights(vocab_size=10000, d_model=512, context_length=256, d_ff=2048, num_layers=4, num_heads=16):

    weights = {
        #Embeddings
        "token_embeddings.weight": torch.randn(vocab_size, d_model) * 0.02,  
        "position_embeddings.weight": torch.randn(context_length, d_model) * 0.02,

        #Final layer normalization
        "ln_final.weight": torch.ones(d_model),  

        #Language model output layer
        "lm_head.weight": torch.randn(vocab_size, d_model) * 0.02,
    }

    #Initialize transformer block weights
    for layer_idx in range(num_layers):
        weights.update({
            #Self-attention projections
            f"layers.{layer_idx}.attn.q_proj.weight": torch.randn(d_model, d_model) * 0.02,
            f"layers.{layer_idx}.attn.k_proj.weight": torch.randn(d_model, d_model) * 0.02,
            f"layers.{layer_idx}.attn.v_proj.weight": torch.randn(d_model, d_model) * 0.02,
            f"layers.{layer_idx}.attn.output_proj.weight": torch.randn(d_model, d_model) * 0.02,

            #Layer normalization for attention
            f"layers.{layer_idx}.ln1.weight": torch.ones(d_model),

            #Feed-forward network (FFN) layers
            f"layers.{layer_idx}.ffn.w1.weight": torch.randn(d_ff, d_model) * 0.02,
            f"layers.{layer_idx}.ffn.w2.weight": torch.randn(d_model, d_ff) * 0.02,

            #Layer normalization for FFN
            f"layers.{layer_idx}.ln2.weight": torch.ones(d_model),
        })

    return weights

#Initialize Model
def initialize_model():
    #Use randomly initialized weights since no checkpoint is available
    weights = initialize_random_weights()

    model = Transformer_LM(
        vocab_size=10000,
        context_length=128, 
        d_model=256,
        num_layers=2,
        num_heads=8,
        d_ff=1024, 
        attn_pdrop=0.1,
        residual_pdrop=0.1,
        weights=weights
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, device

#Define Training Function
def train_model(model, dataset, criterion, optimizer, config, device):
    model.train()

    for epoch in range(config["num_epochs"]):
        epoch_loss = 0

        for i in range(len(dataset) // config["batch_size"]):
            inputs, labels = data_loader(dataset, config["batch_size"], config["seq_length"], device)
            
            optimizer.zero_grad()

            #Forward pass
            output = model(inputs)
            loss = criterion(output.view(-1, output.shape[-1]), labels.view(-1))

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            if i % config["log_interval"] == 0:
                wandb.log({"epoch": epoch, "step_loss": loss.item()})
                print(f"Epoch [{epoch+1}/{config['num_epochs']}], Step [{i+1}], Loss: {loss.item():.4f}")

        val_inputs, val_labels = data_loader(dataset, config["batch_size"], config["seq_length"], device)
        val_loss = validate_model(model, val_inputs, val_labels, criterion)

        wandb.log({"epoch": epoch, "train_loss": epoch_loss / len(dataset), "val_loss": val_loss})
        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Train Loss: {epoch_loss / len(dataset):.4f}, Val Loss: {val_loss:.4f}")

        #Save Checkpoint
        checkpoint_path = os.path.join(config["checkpoint_path"], f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

#Define Validation Function
def validate_model(model, val_inputs, val_labels, criterion):
    model.eval()
    with torch.no_grad():
        output = model(val_inputs)
        loss = criterion(output.view(-1, output.shape[-1]), val_labels.view(-1))
    model.train()
    return loss.item()

#Main Execution
if __name__ == "__main__":
    config = initialize_wandb()
    dataset = load_data(config["batch_size"], config["seq_length"], "cuda" if torch.cuda.is_available() else "cpu")
    model, device = initialize_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)
    train_model(model, dataset, criterion, optimizer, config, device)
