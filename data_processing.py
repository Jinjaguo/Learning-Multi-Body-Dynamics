import glob, os
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader

class SingleStepDynamicsDataset(Dataset):
    """
    Single-step residual dynamics dataset.
    Each sample is a dict with:
      - state:         torch.FloatTensor (16,)
      - action:        torch.FloatTensor ( 3,)
      - physics_next:  torch.FloatTensor (16,)
      - next_state:    torch.FloatTensor (16,)
    """
    def __init__(self, npz_path_or_dir: str):
        if os.path.isdir(npz_path_or_dir):
            paths = sorted(glob.glob(os.path.join(npz_path_or_dir, "*.npz")))
        else:
            paths = [npz_path_or_dir]
        # load numpy arrays
        states_list, actions_list, phys_list, next_list = [], [], [], []
        for p in paths:
            data = np.load(p)
            states_list.append(data["state"])
            actions_list.append(data["action"])
            phys_list.append(data["physics_next"])
            next_list.append(data["next_state"])

        self.states = np.concatenate(states_list, axis=0)
        self.actions = np.concatenate(actions_list, axis=0)
        self.physics_next = np.concatenate(phys_list, axis=0)
        self.next_states = np.concatenate(next_list, axis=0)

        assert len(self.states)==len(self.actions)==len(self.physics_next)==len(self.next_states)
        self.N = len(self.states)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return {
            "state":        torch.from_numpy(self.states[idx]).float(),
            "action":       torch.from_numpy(self.actions[idx]).float(),
            "physics_next": torch.from_numpy(self.physics_next[idx]).float(),
            "next_state":   torch.from_numpy(self.next_states[idx]).float()
        }

def process_data_single_step(npz_path: str,
                             batch_size: int = 500,
                             val_frac: float = 0.2,
                             shuffle: bool = True):
    """
    Loads the .npz, splits into train/val, returns two DataLoaders.
    """
    dataset = SingleStepDynamicsDataset(npz_path)
    N = len(dataset)
    val_size   = int(val_frac * N)
    train_size = N - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = process_data_single_step(
        "dataset",
        batch_size=32,
        val_frac=0.1
    )

    print("Train batches:", len(train_loader))
    print("Val   batches:", len(val_loader))

    batch = next(iter(train_loader))
    print({k: v.shape for k,v in batch.items()})
    # should print:
    # {'state': torch.Size([32,16]),
    #  'action': torch.Size([32,3]),
    #  'physics_next': torch.Size([32,16]),
    #  'next_state': torch.Size([32,16])}