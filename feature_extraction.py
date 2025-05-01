import numpy as np
import os
import pandas as pd
from tsfresh import extract_features
from sklearn.preprocessing import RobustScaler
from scipy.interpolate import interp1d

TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"

def resample_lightcurve(lightcurve, original_interval=30, target_interval=60):
    time = np.arange(0, len(lightcurve) * original_interval, original_interval)
    interp_func = interp1d(time, lightcurve, kind='linear', fill_value='extrapolate')
    resampled_time = np.arange(0, time[-1], target_interval)
    resampled_curve = interp_func(resampled_time)
    return resampled_curve

def extract_tsfresh_features(directory):
    feature_data = []
    labels = []
    tic_ids = []
    
    for file in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, file)
        data = np.load(file_path, allow_pickle=True).item()
        
        if "lightcurve" not in data:
            print(f"ERROR: 'lightcurve' not found in {file}, SKIPPING!")
            continue
        
        lightcurve = data["lightcurve"]
        global_view = lightcurve[:201]
        local_view = lightcurve[201:262]

        global_view_resampled = resample_lightcurve(global_view)
        local_view_resampled = resample_lightcurve(local_view)

        df_global = pd.DataFrame({"id": [file] * len(global_view_resampled), "time": range(len(global_view_resampled)), "flux": global_view_resampled})
        df_local = pd.DataFrame({"id": [file] * len(local_view_resampled), "time": range(len(local_view_resampled)), "flux": local_view_resampled})

        features_global = extract_features(df_global, column_id="id", column_sort="time", n_jobs=0)
        features_local = extract_features(df_local, column_id="id", column_sort="time", n_jobs=0)

        combined_features = pd.concat([features_global, features_local], axis=1)

        feature_data.append(combined_features)
        labels.append(data["label"])
        tic_ids.append(file)
    
    feature_df = pd.concat(feature_data)

    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    feature_df = feature_df.interpolate(method='linear', axis=0)

    feature_df = feature_df.fillna(feature_df.median())

    feature_df = feature_df.fillna(0)

    return feature_df, np.array(labels), np.array(tic_ids)


train_features, train_labels, train_tess_ids = extract_tsfresh_features(TRAIN_DIR)
val_features, val_labels, val_tess_ids = extract_tsfresh_features(VAL_DIR)
test_features, test_labels, test_tess_ids = extract_tsfresh_features(TEST_DIR)

X_train = train_features
y_train = train_labels

X_val = val_features
y_val = val_labels

X_test = test_features
y_test = test_labels

all_features = X_train.columns.tolist()
with open("all_extracted_features.txt", "w") as f:
    for feature in all_features:
        f.write(feature + "\n")

const_features = X_train.columns[X_train.nunique() == 1].tolist()

selected_features = list(set(X_train.columns) - set(const_features))
X_train = X_train[selected_features]
X_val = X_val[selected_features]
X_test = X_test[selected_features]

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

if np.isnan(X_train_scaled).any():
    print("⚠️ Warning: NaNs found in X_train_scaled!")

if np.isnan(X_val_scaled).any():
    print("⚠️ Warning: NaNs found in X_val_scaled!")

if np.isnan(X_test_scaled).any():
    print("⚠️ Warning: NaNs found in X_test_scaled!")


np.save("tess_filenames_train.npy", train_tess_ids)
np.save("tess_filenames_val.npy", val_tess_ids)
np.save("tess_filenames_test.npy", test_tess_ids)

np.save("X_train_tess.npy", X_train_scaled)
np.save("y_train_tess.npy", y_train)

np.save("X_val_tess.npy", X_val_scaled)
np.save("y_val_tess.npy", y_val)

np.save("X_test_tess.npy", X_test_scaled)
np.save("y_test_tess.npy", y_test)

with open("selected_features.txt", "w") as f:
    for feature in selected_features:
        f.write(feature + "\n")

print(f"DONE! Removed {len(const_features)} constant features.")
print(f"Total Features Extracted: {len(all_features)}")
print(f"Final Features After Cleaning: {len(selected_features)}")
