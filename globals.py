dataset_name = "cct_large"

if dataset_name == "cct_small":
    dataset_root = "./dataset/cct"
elif dataset_name == "cct_large":
    dataset_root = "/cct_large"
else:
    raise ValueError(f"{dataset_name} is unknown.")

if dataset_name == "cct_small":
    n_classes = 16
elif dataset_name == "cct_large":
    n_classes = 22
else:
    raise ValueError(f"{dataset_name} is unknown.")

image_width = 320
image_height = 320
image_mean = (0.5, 0.5, 0.5)
image_std = (0.25, 0.25, 0.25)

faster_rcnn_ckpt_path = "best_faster_rcnn.ckpt"
crcnn_ckpt_path = "best_crcnn.ckpt"
memory_long_path = "memory_long.pt"
memory_long_date_path = "memory_long_date.pt"
memory_long_max_features_per_image = None
memory_long_interval = (-30 * 24 * 60 * 60, 30 * 24 * 60 * 60)

update_box_head_params = False

training_stage2_batch_size = 64
training_stage2_num_workers = 8

feature_size = 256

large_negative = -100000