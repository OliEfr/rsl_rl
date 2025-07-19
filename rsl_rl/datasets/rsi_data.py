import torch


rsi_params = {
    "obstacle_2_3126098000_amp.txt": {
        "reference_trajectory_yaw_rot": 210,
        "reference_trajectory_offset": torch.tensor([0.0, 0.4, 0.2], device = "cuda"),
        "reference_trajectory_scaling": torch.tensor([1.0, 1.8, 1.0], device = "cuda"),
    },
    "obstacle_3_3015061000_amp.txt": {
        "reference_trajectory_yaw_rot": 90,
        "reference_trajectory_offset": torch.tensor([0.0, 0.0, 0.19], device = "cuda"),
        "reference_trajectory_scaling": torch.tensor([1.0, 1.2, 1.2], device = "cuda"), # can easily increase z scaling to, say, 1.5
    },
    "obstacle_3_3015061000_amp_increased_feet_z.txt": {
        "reference_trajectory_yaw_rot": 90,
        "reference_trajectory_offset": torch.tensor([0.0, 0.0, 0.1], device = "cuda"),
        "reference_trajectory_scaling": torch.tensor([1.0, 1.2, 1.7], device = "cuda"), # can easily increase z scaling to, say, 1.5. 1.7 seems about max and is high enough for 30cm step size.
    }
}
