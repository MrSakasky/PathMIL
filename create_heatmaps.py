import numpy as np
import argparse
import os
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import h5py
import yaml
from utils.eval_utils import load_trained_model
from models import get_encoder
from wsi_core.batch_process_utils import initialize_df
from utils.file_utils import save_hdf5

PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="Generate PathMIL heatmaps.")
    parser.add_argument("--config", type=Path, default=Path("config_template.yaml"))
    parser.add_argument("--save-exp-code")
    parser.add_argument("--overlap", type=float)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Run without the interactive configuration confirmation.",
    )
    return parser


def infer_single_slide(
    model,
    features,
    coords,
    label,
    reverse_label_dict,
    device,
    k=1,
):
    """Run slide-level inference and return predictions and attention."""
    features = features.to(device)
    coords_t = torch.as_tensor(coords, dtype=torch.long, device=device)

    with torch.inference_mode():
        output = model(features, coords_t)
        probabilities = output["probabilities"]
        prediction = output["predictions"].item()
        attention = output["attention"].reshape(-1, 1).cpu().numpy()

        print('Prediction: {}, label: {}, probabilities: {}'.format(
            reverse_label_dict[prediction],
            label,
            ["{:.4f}".format(p) for p in probabilities.cpu().flatten()]
        ))

        k = min(k, probabilities.size(1))
        probs, ids = torch.topk(probabilities, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])

    return ids, preds_str, probs, attention


def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
    return params


def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict


def save_image_with_resize(image, save_path, quality=100, resize_factor=0.9):
    """Shrink an image only when its encoder rejects the current dimensions."""
    while True:
        try:
            image.save(save_path, quality=quality)
            print(f"Image saved successfully at resolution {image.size}.")
            break
        except OSError as e:
            if "image dimension" in str(e) or "broken data stream when writing image file" in str(e):
                print(f"Image too large or corrupted with size {image.size}, resizing...")
                new_width = int(image.width * resize_factor)
                new_height = int(image.height * resize_factor)
                if new_width < 1 or new_height < 1:
                    raise ValueError("Image dimensions have become too small to resize further.")
                image = image.resize((new_width, new_height), Image.LANCZOS)
            else:
                raise


if __name__ == '__main__':
    args = build_parser().parse_args()
    from vis_utils.heatmap_utils import (
        compute_from_patches,
        draw_heatmap,
        initialize_wsi,
    )
    from wsi_core.wsi_utils import sample_rois

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = args.config
    if not config_path.is_file():
        config_path = PROJECT_ROOT / "heatmaps" / "configs" / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle)
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n' + key)
            for value_key, value_value in value.items():
                print(value_key + " : " + str(value_value))
        else:
            print('\n' + key + " : " + str(value))

    if not args.yes:
        decision = input('Continue? Y/N ').strip().lower()
        if decision not in {"y", "yes"}:
            print("Cancelled")
            raise SystemExit(0)

    args_dict = config_dict
    patch_args = argparse.Namespace(**args_dict['patching_arguments'])
    data_args = argparse.Namespace(**args_dict['data_arguments'])
    model_args = args_dict['model_arguments']
    model_args.update({'n_classes': args_dict['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    encoder_args = argparse.Namespace(**args_dict['encoder_arguments'])
    exp_args = argparse.Namespace(**args_dict['exp_arguments'])
    heatmap_args = argparse.Namespace(**args_dict['heatmap_arguments'])
    sample_args = argparse.Namespace(**args_dict['sample_arguments'])

    patch_size = tuple([patch_args.patch_size for _ in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(
        patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]
    ))

    preset = data_args.preset
    def_seg_params = {
        'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False,
        'keep_ids': 'none', 'exclude_ids': 'none'
    }
    def_filter_params = {'a_t': 50.0, 'a_h': 8.0, 'max_n_holes': 10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]
        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]
        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]
        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(data_args.data_dir))

        slides = [slide for slide in slides if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                           use_heatmap_args=False)
    else:
        process_path = Path(data_args.process_list)
        if not process_path.is_file():
            process_path = PROJECT_ROOT / "heatmaps" / "process_lists" / process_path
        df = pd.read_csv(
            process_path,
            encoding=getattr(data_args, "csv_encoding", "utf-8"),
        )
        df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                           use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))

    model = load_trained_model(model_args, ckpt_path, device)

    model = model.to(device)
    model.eval()

    feature_extractor, img_transforms = get_encoder(
        encoder_args.model_name,
        target_img_size=encoder_args.target_img_size,
        checkpoint_path=getattr(encoder_args, "checkpoint_path", None),
    )
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    print('Done!')

    label_dict = data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)

    blocky_wsi_kwargs = {
        'top_left': None,
        'bot_right': None,
        'patch_size': patch_size,
        'step_size': patch_size,
        'custom_downsample': patch_args.custom_downsample,
        'level': patch_args.patch_level,
        'use_center_shift': heatmap_args.use_center_shift
    }

    for i in tqdm(range(len(process_stack))):
        slide_name = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        print('\nprocessing: ', slide_name)

        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(data_args.slide_ext, '')

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

        p_slide_save_dir = os.path.join(
            exp_args.production_save_dir,
            exp_args.save_exp_code,
            str(grouping)
        )
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(
            exp_args.raw_save_dir,
            exp_args.save_exp_code,
            str(grouping),
            slide_id
        )
        os.makedirs(r_slide_save_dir, exist_ok=True)

        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None

        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, dict):
            data_dir_key = process_stack.loc[i, data_args.data_dir_key]
            slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
        else:
            raise ValueError("data_dir must be a directory string or source mapping")

        mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')

        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))

        print('Initializing WSI object')
        wsi_object = initialize_wsi(
            slide_path,
            seg_mask_path=mask_file,
            seg_params=seg_params,
            filter_params=filter_params
        )
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
        vis_patch_size = tuple(
            (np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int)
        )

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))

        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level

        mask_img = wsi_object.visWSI(**vis_params, number_contours=True)
        mask_img.save(mask_path)

        features_path = os.path.join(r_slide_save_dir, slide_id + '.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id + '.h5')

        if not os.path.isfile(h5_path):
            _, _, wsi_object = compute_from_patches(
                wsi_object=wsi_object,
                model=None,
                feature_extractor=feature_extractor,
                img_transforms=img_transforms,
                batch_size=exp_args.batch_size,
                **blocky_wsi_kwargs,
                attention_save_path=None,
                feature_save_path=h5_path,
                reference_scores=None
            )

        if not os.path.isfile(features_path):
            with h5py.File(h5_path, "r") as file:
                features_np = file['features'][:]
            features = torch.tensor(features_np)
            torch.save(features, features_path)

        features = torch.load(features_path, map_location="cpu")

        with h5py.File(h5_path, "r") as file:
            coords = file['coords'][:]

        process_stack.loc[i, 'bag_size'] = len(features)

        wsi_object.saveSegmentation(mask_file)

        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(
            model,
            features,
            coords,
            label,
            reverse_label_dict,
            device,
            exp_args.n_classes,
        )
        del features

        if not os.path.isfile(block_map_save_path):
            asset_dict = {'attention_scores': A, 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

        for c in range(exp_args.n_classes):
            process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
            process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

        os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_stack.to_csv(
                'heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')),
                index=False
            )
        else:
            process_stack.to_csv(
                'heatmaps/results/{}.csv'.format(exp_args.save_exp_code),
                index=False
            )

        with h5py.File(block_map_save_path, 'r') as file:
            scores = file['attention_scores'][:]
            coords = file['coords'][:]

        samples = sample_args.samples
        for sample in samples:
            if sample['sample']:
                tag = "label_{}_pred_{}".format(label, Y_hats[0])
                sample_save_dir = os.path.join(
                    exp_args.production_save_dir,
                    exp_args.save_exp_code,
                    'sampled_patches',
                    str(tag),
                    sample['name']
                )
                os.makedirs(sample_save_dir, exist_ok=True)
                print('sampling {}'.format(sample['name']))
                sample_results = sample_rois(
                    scores, coords,
                    k=sample['k'],
                    mode=sample['mode'],
                    seed=sample['seed'],
                    score_start=sample.get('score_start', 0),
                    score_end=sample.get('score_end', 1)
                )
                for idx_s, (s_coord, s_score) in enumerate(
                        zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    patch = wsi_object.wsi.read_region(
                        tuple(s_coord),
                        patch_args.patch_level,
                        (patch_args.patch_size, patch_args.patch_size)
                    ).convert('RGB')
                    patch.save(os.path.join(
                        sample_save_dir,
                        '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx_s, slide_id, s_coord[0], s_coord[1], s_score)
                    ))

        wsi_kwargs = {
            'top_left': top_left,
            'bot_right': bot_right,
            'patch_size': patch_size,
            'step_size': step_size,
            'custom_downsample': patch_args.custom_downsample,
            'level': patch_args.patch_level,
            'use_center_shift': heatmap_args.use_center_shift
        }

        heatmap_save_name_block = '{}_blockmap.tiff'.format(slide_id)
        if not os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name_block)):
            heatmap_block = draw_heatmap(
                scores, coords, slide_path,
                wsi_object=wsi_object,
                cmap=heatmap_args.cmap,
                alpha=heatmap_args.alpha,
                use_holes=True,
                binarize=False,
                vis_level=-1,
                blank_canvas=False,
                thresh=-1,
                patch_size=vis_patch_size,
                convert_to_percentiles=True
            )
            heatmap_block.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
            del heatmap_block

        heatmap_vis_args = {
            'convert_to_percentiles': True,
            'vis_level': heatmap_args.vis_level,
            'blur': heatmap_args.blur,
            'custom_downsample': heatmap_args.custom_downsample
        }
        if heatmap_args.use_ref_scores:
            heatmap_vis_args['convert_to_percentiles'] = False

        heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(
            slide_id,
            float(patch_args.overlap),
            int(heatmap_args.use_roi),
            int(heatmap_args.blur),
            int(heatmap_args.use_ref_scores),
            int(heatmap_args.blank_canvas),
            float(heatmap_args.alpha),
            int(heatmap_args.vis_level),
            int(heatmap_args.binarize),
            float(heatmap_args.binary_thresh),
            heatmap_args.save_ext
        )

        if not os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
            heatmap = draw_heatmap(
                scores, coords, slide_path,
                wsi_object=wsi_object,
                cmap=heatmap_args.cmap,
                alpha=heatmap_args.alpha,
                **heatmap_vis_args,
                binarize=heatmap_args.binarize,
                blank_canvas=heatmap_args.blank_canvas,
                thresh=heatmap_args.binary_thresh,
                patch_size=vis_patch_size,
                overlap=patch_args.overlap,
                top_left=top_left,
                bot_right=bot_right
            )
            if heatmap_args.save_ext == 'jpg':
                save_path_img = os.path.join(p_slide_save_dir, heatmap_save_name)
                save_image_with_resize(heatmap, save_path_img, quality=100)
            else:
                heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

        if heatmap_args.save_orig:
            if heatmap_args.vis_level >= 0:
                vis_level = heatmap_args.vis_level
            else:
                vis_level = vis_params['vis_level']

            heatmap_save_name_orig = '{}_orig_{}.{}'.format(
                slide_id, int(vis_level), heatmap_args.save_ext
            )
            if not os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name_orig)):
                orig = wsi_object.visWSI(
                    vis_level=vis_level,
                    view_slide_only=True,
                    custom_downsample=heatmap_args.custom_downsample
                )
                if heatmap_args.save_ext == 'jpg':
                    orig.save(os.path.join(p_slide_save_dir, heatmap_save_name_orig), quality=100)
                else:
                    orig.save(os.path.join(p_slide_save_dir, heatmap_save_name_orig))

    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
