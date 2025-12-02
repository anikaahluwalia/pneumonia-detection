"""
Standalone Pneumonia Detection Training Script
Converted from Inspirit AI notebook
Author: <your name>
"""

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# IMPORT YOUR INSPIRIT FUNCTIONS
# (Make sure the helper file is in the same folder!)
# Example:
# from utils import (
#     get_train_data, get_test_data, get_field_data,
#     CNNClassifier, rotate, scale, shear,
#     flip_left_right, flip_up_down, remove_color,
#     combine_data, plot_one_image, monitor
# )
# ---------------------------------------------------------

from utils import (
    get_train_data, get_test_data, get_field_data,
    CNNClassifier, rotate, scale, shear,
    flip_left_right, flip_up_down, remove_color,
    combine_data, plot_one_image, monitor
)

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():

    # =========================================================
    # LOAD DATA
    # =========================================================

    train_data, train_labels = get_train_data()
    test_data, test_labels = get_test_data()
    field_data, field_labels = get_field_data()

    print("\n=== Training baseline CNN (2-layer) ===")

    # =========================================================
    # BASELINE MODEL TRAINING
    # =========================================================
    cnn = CNNClassifier(num_hidden_layers=2)
    cnn_history = cnn.fit(
        train_data,
        train_labels,
        epochs=50,
        validation_data=(test_data, test_labels),
        shuffle=True,
        callbacks=[monitor]
    )

    # Baseline field accuracy
    field_preds = (cnn.predict(field_data) > 0.5).astype("int32")
    baseline_acc = accuracy_score(field_labels, field_preds)
    print("\nBaseline Field Accuracy:", baseline_acc)

    # =========================================================
    # RUN 5 MODELS AND AVERAGE BASELINE ACCURACY
    # =========================================================
    print("\n=== Measuring Baseline Performance Variability (5 runs) ===")

    avg_baseline = 0

    for i in range(5):
        print(f"\n--- Training baseline run {i+1} ---")

        cnn_temp = CNNClassifier(num_hidden_layers=2)
        cnn_temp.fit(
            train_data,
            train_labels,
            epochs=5,
            validation_data=(test_data, test_labels),
            shuffle=True,
            callbacks=[monitor]
        )

        preds = (cnn_temp.predict(field_data) > 0.5).astype("int32")
        acc = accuracy_score(field_labels, preds)

        print(f"Run {i+1} Field Accuracy: {acc}")
        avg_baseline += acc / 5

    print("\nAverage Baseline Field Accuracy:", avg_baseline)

    # =========================================================
    # DATA AUGMENTATION
    # =========================================================
    print("\n=== Performing Data Augmentation ===")

    train_rot_15 = rotate(train_data, rotate=15)
    train_rot_neg15 = rotate(train_data, rotate=-15)
    train_scaled = scale(train_data, scale=1.2)
    train_sheared = shear(train_data, shear=20)
    train_flip_lr = flip_left_right(train_data, prob=1.0)
    train_flip_ud = flip_up_down(train_data, prob=1.0)
    train_grayish = remove_color(remove_color(train_data, channel=1), channel=2)

    # Combine everything
    all_data, all_labels = combine_data(
        [
            train_data,
            train_rot_15,
            train_rot_neg15,
            train_scaled,
            train_sheared,
            train_flip_lr,
            train_flip_ud,
            train_grayish
        ],
        [
            train_labels,
            train_labels,
            train_labels,
            train_labels,
            train_labels,
            train_labels,
            train_labels,
            train_labels
        ]
    )

    print("Augmented dataset size:", len(all_data))

    # =========================================================
    # RETRAIN ON AUGMENTED SET (5 RUNS)
    # =========================================================
    print("\n=== Training on Augmented Dataset (5 runs) ===")

    augmented_avg = 0

    for i in range(5):
        print(f"\n--- Augmented training run {i+1} ---")

        cnn_aug = CNNClassifier(num_hidden_layers=2)
        cnn_aug.fit(
            all_data,
            all_labels,
            epochs=10,
            validation_data=(test_data, test_labels),
            shuffle=True,
            callbacks=[monitor]
        )

        preds = (cnn_aug.predict(field_data) > 0.5).astype("int32")
        acc = accuracy_score(field_labels, preds)

        print(f"Augmented Run {i+1} Field Accuracy: {acc}")
        augmented_avg += acc / 5

    print("\nFINAL Average Field Accuracy after augmentation:", augmented_avg)

    print("\n=== DONE ===")


# ---------------------------------------------------------
# RUN MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
