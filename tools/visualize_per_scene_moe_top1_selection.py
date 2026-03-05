import pickle
import os
import argparse
import numpy as np
import matplotlib

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MOE top1 expert selection using manual score-layer mapping")
    parser.add_argument('--large', action='store_true', help='Use large model')
    parser.add_argument('--unpooling_points_path', help='Path to unpooling points pickle')
    parser.add_argument('--pooling_points_path', help='Path to pooling points pickle')
    parser.add_argument('--scores_path', help='Path to scores pickle')
    parser.add_argument('--output_dir', default='ply_output', help='Output directory for PLY files')
    return parser.parse_args()

def write_colored_ply(coords: np.ndarray, scores: np.ndarray, filename: str):
    top1 = np.argmax(scores, axis=1)
    num_experts = scores.shape[1]

    custom_color_map = np.array([
        [255,   0,   0],     # Red
        [  0, 255,   0],     # Green
        [  0,   0, 255],     # Blue
        [255, 255,   0],     # Yellow
        [255,   0, 255],     # Magenta
        [  0, 255, 255],     # Cyan
        [255, 165,   0],     # Orange
        [128,   0, 128],     # Purple
        [  0, 128, 128],     # Teal
        [128, 128,   0],     # Olive
        [255, 105, 180],     # Hot pink
        [ 70, 130, 180],     # Steel blue
        [  0,   0,   0],     # Black
        [105, 105, 105],     # Dim gray
        [255, 215,   0],     # Gold
        [ 34, 139,  34],     # Forest green
        [  0,  100,   0],     # Dark green
        [  0,   0, 139],     # Dark blue
        [139,   0,   0],     # Dark red
        [ 75,   0, 130],     # Indigo
    ], dtype=np.uint8)
    color_map = custom_color_map[:num_experts]
    colors = np.array([color_map[i] for i in top1], dtype=np.uint8)

    with open(filename, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {coords.shape[0]}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(coords[:, :3], colors):
            f.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')

def process(points_dict, scores_dict, score_groups, tag, output_dir):
    points_list = points_dict['proj_head']
    scores_list = scores_dict['proj_head']

    for layer_id, coords in enumerate(points_list):
        score_ids = score_groups.get(tag, {}).get(layer_id, [])
        if not score_ids:
            print(f'[WARN] No scores matched for {tag}_layer{layer_id}')
            continue
        for sid in score_ids:
            scores = scores_list[sid]
            filename = f'{tag}_layer{layer_id}_score{sid}.ply'
            filepath = os.path.join(output_dir, filename)
            print(f'[INFO] Writing {filename} for layer {layer_id} (score idx {sid})')
            write_colored_ply(coords, scores, filepath)

def main():
    args = parse_args()

    with open(args.unpooling_points_path, 'rb') as f:
        unpooling_points = pickle.load(f)
    with open(args.pooling_points_path, 'rb') as f:
        pooling_points = pickle.load(f)
    with open(args.scores_path, 'rb') as f:
        scores = pickle.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # 💡 Manual mapping based on visual inspection
    score_groups = {
        'encoder': {
            0: [0, 1],                  # (147944, 3)
            1: [2, 3],                  # (53654, 3)
            2: [4, 5],                  # (14662, 3)
            3: [6, 7, 8, 9, 10, 11],    # (3905, 3)
            4: [12, 13],               # (1013, 3)
        },
        'decoder': {
            0: [14, 15],               # (3905, 3)
            1: [16, 17],               # (14662, 3)
            2: [18, 19],               # (53654, 3)
            3: [20, 21],               # (147944, 3)
        }
    }

    process(pooling_points, scores, score_groups, tag='encoder', output_dir=args.output_dir)
    process(unpooling_points, scores, score_groups, tag='decoder', output_dir=args.output_dir)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()