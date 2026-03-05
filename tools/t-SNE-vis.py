import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import matplotlib.font_manager

def create_tsne_plot(features, labels, title, output_path, pdf=False):
    # Set font
    try:
        plt.rcParams["font.family"] = "Times New Roman"
    except:
        plt.rcParams["font.family"] = "DejaVu Serif"

    plt.rcParams["font.size"] = 16

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(8, 6))
    unique_labels = list(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = np.array(labels) == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[color],
            label=label,
            alpha=0.7,
            s=30,  # <<< 点更大
            edgecolors='none'
        )

    # Legend settings
    legend = plt.legend(
        loc='upper left',
        fontsize=12,
        frameon=True
    )
    # legend.get_frame().set_facecolor('lightgrey')  # 灰色背景
    legend.get_frame().set_edgecolor('black')      # 黑色边框

    plt.title(title, fontsize=16)
    # 不隐藏ticks，保留x轴和y轴刻度
    # plt.xlabel("t-SNE dimension 1", fontsize=16)
    # plt.ylabel("t-SNE dimension 2", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3)  # 小网格线，可选，风格更好看
    plt.tight_layout()

    # Save the plot
    if pdf:
        output_path = output_path.replace(".png", ".pdf")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main(feature_dir, max_points_per_file=500, pdf=False):
    encoder_features = []
    decoder_features = []
    encoder_labels = []
    decoder_labels = []

    excluded_datasets = ["waymo", "nuscenes", "semantickitti"]

    for root, dirs, files in os.walk(feature_dir):
        for file in files:
            if file.endswith(".npy"):
                label = file.split("_")[0].lower()
                if any(excluded in label for excluded in excluded_datasets):
                    print(f"Skipping {file} (excluded)")
                    continue

                feature_path = os.path.join(root, file)
                feature = np.load(feature_path)

                if len(feature) > max_points_per_file:
                    indices = np.random.choice(len(feature), max_points_per_file, replace=False)
                    feature = feature[indices]

                if "encoder" in file.lower():
                    encoder_features.append(feature)
                    encoder_labels.extend([label] * len(feature))
                elif "decoder" in file.lower():
                    decoder_features.append(feature)
                    decoder_labels.extend([label] * len(feature))

    encoder_output = os.path.join(feature_dir, "tsne_visualization_encoder.png")
    decoder_output = os.path.join(feature_dir, "tsne_visualization_decoder.png")

    if encoder_features:
        encoder_features = np.vstack(encoder_features)
        create_tsne_plot(
            encoder_features,
            encoder_labels,
            "t-SNE Visualization of Encoder Features",
            encoder_output,
            pdf=pdf,
        )
        print(f"Saved encoder visualization to: {encoder_output if not pdf else encoder_output.replace('.png', '.pdf')}")

    if decoder_features:
        decoder_features = np.vstack(decoder_features)
        create_tsne_plot(
            decoder_features,
            decoder_labels,
            "t-SNE Visualization of Decoder Features",
            decoder_output,
            pdf=pdf,
        )
        print(f"Saved decoder visualization to: {decoder_output if not pdf else decoder_output.replace('.png', '.pdf')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate t-SNE visualization from feature files")
    parser.add_argument("--feature_dir", type=str, required=True, help="Directory containing the feature files (.npy)")
    parser.add_argument("--max_points_per_file", type=int, default=500, help="Maximum number of points per file (default: 500)")
    parser.add_argument("--pdf", action="store_true", help="Save output as PDF instead of PNG")
    args = parser.parse_args()

    if not os.path.exists(args.feature_dir):
        print(f"Error: Feature directory '{args.feature_dir}' does not exist.")
        exit(1)

    main(args.feature_dir, args.max_points_per_file, pdf=args.pdf)