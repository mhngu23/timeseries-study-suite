from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm

def train_model(model, dataset, epochs=20, batch_size=32, lr=1e-3, device='cpu', save_path=None, model_name=None):
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            # x shape Batch size, input length, feature dimension
            # y shape batch size, output dimension
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")