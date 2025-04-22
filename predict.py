# fire_damage_pred/predict.py
import torch, numpy as np
from model import FireDamageModel
from utils import damage_mapping

inv_map = {v:k for k,v in damage_mapping.items()}

def predict(features_path="new_building.npy"):
    X = np.load(features_path)          # (T, C, H, W)
    X = torch.tensor(X).unsqueeze(0)    # (1, T, C, H, W)
    model = FireDamageModel(in_ch=6, hid_ch=32, num_cls=6)
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model(X.float())
        label = out.argmax(1).item()
        print("Predicted:", inv_map[label])

if __name__ == "__main__":
    predict()
