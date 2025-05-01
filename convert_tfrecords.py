import tensorflow as tf
import numpy as np
import os


RAW_DIR = "raw"
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True)


def _parse_function(proto):
    keys_to_features = {
        'global_view': tf.io.FixedLenFeature([201], tf.float32),
        'local_view': tf.io.FixedLenFeature([61], tf.float32),
        'tic_id': tf.io.FixedLenFeature([], tf.int64),
        'Disposition': tf.io.FixedLenFeature([], tf.string),  # label
    }
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    return parsed_features['tic_id'], parsed_features['global_view'], parsed_features['local_view'], parsed_features['Disposition']

tfrecord_files = [os.path.join(RAW_DIR, f) for f in os.listdir(RAW_DIR) if not f.startswith(".")]

total_pc, total_npc = 0, 0
tic_id_counts = {}

for file in tfrecord_files:
    print(f"File: {file}")
    dataset = tf.data.TFRecordDataset(file).map(_parse_function)

    for tic_id, global_view, local_view, disposition in dataset:
        tic_id = int(tic_id.numpy())
        label = disposition.numpy().decode("utf-8")
        global_view = global_view.numpy()
        local_view = local_view.numpy()

        y_label = 1 if label == "PC" else 0

        combined_lightcurve = np.concatenate((global_view, local_view))  # 201 + 61 = 262

        if tic_id in tic_id_counts:
            tic_id_counts[tic_id] += 1
        else:
            tic_id_counts[tic_id] = 1
        
        unique_filename = f"{tic_id}_{tic_id_counts[tic_id]}.npy"

        if "train" in file:
            target_dir = TRAIN_DIR
        elif "val" in file:
            target_dir = VAL_DIR
        elif "test" in file:
            target_dir = TEST_DIR
        else:
            continue
        
        final_save_path = os.path.join(target_dir, unique_filename)
        np.save(final_save_path, {"lightcurve": combined_lightcurve, "label": y_label, "tic_id": tic_id})
        
        print(f"Saved: {final_save_path}")

        if y_label == 1:
            total_pc += 1
        else:
            total_npc += 1

print(f"Total: PC = {total_pc}, NPC = {total_npc}")
