import os
import torch

from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

from trainer_utils import save_checkpoint, load_checkpoint


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_model(
    model,
    dataset,
    val_dataset=None,
    epochs=20,
    batch_size=32,
    lr=1e-3,
    device='cpu',
    save_path=None,
    model_name=None,
    resume=False,
    patience=10,  
    scheduler_patience=5,       
    scheduler_factor=0.5,       
    min_lr=1e-6    
):
    model.to(device)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, min_lr=min_lr)

    start_epoch = 0
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    full_save_path = None
    if save_path and model_name:
        full_save_path = os.path.join(save_path, model_name)

    if resume and full_save_path and os.path.exists(full_save_path):
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, full_save_path, device)

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            # print(f"Output shape: {output.shape}, Target shape: {y.shape}")  # Debugging line
            # exit()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # Validation + Early stopping
        if val_loader:
            val_loss = evaluate_model(model, val_loader, criterion, device)
            print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

            scheduler.step(val_loss)

            if full_save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch + 1, best_val_loss, full_save_path)
                epochs_since_improvement = 0  # ✅ reset counter
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    print(f"🛑 Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.")
                    break
        else:
            scheduler.step(avg_train_loss)

            # No validation: use train loss
            if full_save_path and avg_train_loss < best_val_loss:
                best_val_loss = avg_train_loss
                save_checkpoint(model, optimizer, epoch + 1, best_val_loss, full_save_path)
