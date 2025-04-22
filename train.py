# fire_damage_pred/train.py
import random, numpy as np, torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from dataset import FireDataset
from model   import FireDamageModel
from tqdm.auto import tqdm                 # 进度条

# ------------------ 可调参数 ------------------ #
SEED        = 42
BATCH_SIZE  = 16
EPOCHS      = 20
LR          = 1e-3
HIDDEN_CH   = 32
VAL_SPLIT   = 0.2
IN_CHANNELS = 6        # 火灾 + 天气通道
NUM_CLASSES = 6
# --------------------------------------------- #

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 数据集 & DataLoader
    dataset = FireDataset()
    train_len = int((1 - VAL_SPLIT) * len(dataset))
    val_len   = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # 2. 模型、损失、优化器
    model = FireDamageModel(IN_CHANNELS, HIDDEN_CH, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # ---------- 训练 ----------
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for X_batch, y_batch in progress_bar:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total   += y_batch.size(0)
            correct += (preds == y_batch).sum().item()

            # 在进度条尾部动态显示当前 batch 情况
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "batch_acc": f"{(preds == y_batch).float().mean():.2%}"
            })

        epoch_loss = running_loss / len(train_loader)
        train_acc  = correct / total

        # ---------- 验证 ----------
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.float().to(device)
                y_val = y_val.long().to(device)
                out   = model(X_val)
                _, pred = torch.max(out, 1)
                val_total   += y_val.size(0)
                val_correct += (pred == y_val).sum().item()
        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch} summary: "
              f"Train Loss={epoch_loss:.4f} | Train Acc={train_acc:.2%} | Val Acc={val_acc:.2%}")

        # ---------- 保存最好模型 ----------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✓  New best model saved (Val Acc={best_acc:.2%})")

    print(f"\nTraining finished. Best Val Accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    main()
