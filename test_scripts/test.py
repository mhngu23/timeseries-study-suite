import torch
from torch.utils.data import DataLoader

from datasets.sliding_window_loader import SlidingWindowDataset

from data.loaders.processed_loader import load_processed_ett
from test_scripts.test_utils import load_model_checkpoint

etth1_train_stats = {
    'x_mean': [0.009869, -0.139674, 0.036506, -0.103898, -0.128803, -0.127520],
    'x_std': [0.898649, 1.034665, 0.901914, 1.065443, 1.020725, 1.104933],
    'y_mean': 16.29471486647764,
    'y_std': 8.34847203863057,
    'feature_names': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
}

def test_model(
    model,                 # Pass initialized model here
    test_path,
    checkpoint_path,
    input_len=24,
    pred_len=1,
    batch_size=64,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    X_test, y_test = load_processed_ett(test_path, target_col="OT", use_target=False)

    test_dataset = SlidingWindowDataset(X_test, y_test, input_len=input_len, pred_len=pred_len, normalize=True, stats=etth1_train_stats)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load checkpoint weights into the passed model
    model = load_model_checkpoint(model, checkpoint_path, device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            all_preds.append(output.cpu())
            all_targets.append(y_batch)

    all_preds = torch.cat(all_preds, dim=0).squeeze(-1).numpy()
    all_targets = torch.cat(all_targets, dim=0).squeeze(-1).numpy()

    mse_loss_fn = torch.nn.MSELoss()
    mae_loss_fn = torch.nn.L1Loss()

    mse = mse_loss_fn(torch.tensor(all_preds), torch.tensor(all_targets)).item()
    mae = mae_loss_fn(torch.tensor(all_preds), torch.tensor(all_targets)).item()


    return all_preds, all_targets, mse, mae