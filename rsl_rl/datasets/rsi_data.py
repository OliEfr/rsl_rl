import torch


rsi_params = {
    "datasets/fromVision_motions_DepthCam_obstacle/obstacle_2_3126098000_amp.txt": {
        "reference_trajectory_yaw_rot": 210,
        "reference_trajectory_offset": torch.tensor([0.0, 0.4, 0.25], device = "cuda"),
        "reference_trajectory_scaling": torch.tensor([1.0, 1.8, 1.0], device = "cuda"),
    },
    "datasets/fromVision_motions_DepthCam_obstacle/obstacle_3_3015061000_amp.txt": {
        "reference_trajectory_yaw_rot": 90,
        "reference_trajectory_offset": torch.tensor([0.0, 0.0, 0.19], device = "cuda"),
        "reference_trajectory_scaling": torch.tensor([1.0, 1.2, 1.2], device = "cuda"), # can easily increase z scaling to, say, 1.5
    },
    "datasets/fromVision_motions_DepthCam_obstacle/obstacle_3_3015061000_amp_increased_feet_z.txt": {
        "reference_trajectory_yaw_rot": 90,
        "reference_trajectory_offset": torch.tensor([0.0, -0.08, 0.15], device = "cuda"),
        "reference_trajectory_scaling": torch.tensor([1.0, 1.4, 1.5], device = "cuda"), # can easily increase z scaling to, say, 1.5. 1.7 seems about max and is high enough for 30cm step size.
    },
    "datasets/fromVision_motions_DepthCam_obstacle/slow_1313807000_amp.txt": {
        "reference_trajectory_yaw_rot": 110,
        "reference_trajectory_offset": torch.tensor([0.0, -0.5, 0.02], device = "cuda"),
        "reference_trajectory_scaling": torch.tensor([1.0, 1.0, 1.0], device = "cuda"),
    },
}
