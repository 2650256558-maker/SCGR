import argparse
import os
# Must be set before CUDA context is created when deterministic CUDA kernels are requested.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn


parser = argparse.ArgumentParser("SCGR multimodal ablation")
parser.add_argument(
    '--dataset',
    choices=['Houston', 'Trento', 'Augsburg'],
    default='Augsburg',
    help='dataset to use'
)
parser.add_argument('--dataset_root', type=str, default=os.environ.get('SCGR_DATASET_ROOT', os.environ.get('CTFN_DATASET_ROOT', './dataset')),
                    help='dataset root directory; default uses SCGR_DATASET_ROOT, legacy CTFN_DATASET_ROOT, or ./dataset')
parser.add_argument("--num_run", type=int, default=3)
parser.add_argument('--epoches', type=int, default=100, help='epoch number')
parser.add_argument('--superpixel_scale', type=int, default=250, help='larger value means fewer superpixels')
parser.add_argument('--max_superpixels', type=int, default=600, help='upper bound of superpixel nodes for Graphormer bias')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma; kept for compatibility')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--same_seed_each_run', action='store_true',
                    help='debug option: use exactly the same seed for every run; otherwise run_seed=seed+run')
parser.add_argument('--strict_repro', action='store_true',
                    help='use torch deterministic algorithms; may raise/warn if a CUDA op has no deterministic implementation')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--test_freq', type=int, default=10, help='evaluation interval')
parser.add_argument('--cnn_nhid', type=int, default=32, help='CNN hidden channels; use 32 for large Houston to save memory')
parser.add_argument('--eval_start_epoch', type=int, default=1, help='start testing from this epoch, 1-indexed')
parser.add_argument('--early_stop_patience', type=int, default=30, help='early stop if best metric does not improve; <=0 disables')
parser.add_argument('--scheduler', choices=['none', 'cosine'], default='cosine', help='learning-rate scheduler')
parser.add_argument('--eta_min', type=float, default=1e-6, help='minimum lr for cosine scheduler')
parser.add_argument('--best_metric', choices=['OA', 'AA', 'Kappa', 'sum'], default='sum', help='metric used to choose best epoch; sum is recommended when OA/AA/Kappa must improve together')

# -------------------------------------------------------------------------------
# Manual multimodal / ablation switches.
# -------------------------------------------------------------------------------
parser.add_argument(
    '--ablation_mode',
    choices=['exp1_hsi_only', 'exp2_aux_concat', 'exp3_cross_nogate', 'exp4_gated_cross', 'exp5_full'],
    default='exp5_full',
    help='manual switch for ablation experiments'
)
parser.add_argument(
    '--use_aux',
    type=str,
    default='auto',
    choices=['auto', 'true', 'false', '1', '0', 'yes', 'no'],
    help=(
        'whether to use auxiliary modality. auto = use aux for exp2-exp5 when --aux_sources is not none; '
        'exp1_hsi_only always forces HSI-only.'
    )
)
parser.add_argument(
    '--aux_sources',
    type=str,
    default='sar',
    help=(
        'which auxiliary modalities to use: auto, none, lidar, sar, dsm, sar_dsm, lidar_sar, '
        'lidar_dsm, lidar_sar_dsm, all, or comma-separated names such as "lidar,sar". '
        'auto = LiDAR for Houston/Trento; SAR for Augsburg. Use --aux_sources dsm for the Augsburg DSM-only test.'
    )
)
parser.add_argument('--allow_missing_aux', action='store_true',
                    help='skip missing auxiliary files when multiple aux sources are requested')
parser.add_argument('--fusion_dim', type=int, default=64, help='token dimension for conservative HSI-guided fusion')
parser.add_argument('--fusion_heads', type=int, default=4, help='number of heads in HSI-guided cross-modal fusion')
parser.add_argument('--fusion_depth', type=int, default=1, help='depth of lightweight CLS-token fusion encoder')
parser.add_argument('--fusion_dropout', type=float, default=0.2, help='dropout in conservative cross-modal fusion')
parser.add_argument(
    '--fusion_direction',
    choices=['auto', 'hsi_to_aux', 'aux_to_hsi', 'bidirectional', 'none'],
    default='auto',
    help='cross-modal attention direction. auto/hsi_to_aux means HSI queries read auxiliary K/V; recommended for avoiding DSM negative transfer.'
)
parser.add_argument('--aux_nhid', type=int, default=0, help='auxiliary branch hidden channels; <=0 means dataset/source-aware auto')
parser.add_argument(
    '--aux_profile',
    type=str,
    default='auto',
    help='auxiliary preprocessing profile: auto, generic, lidar, sar, dsm, sar_dsm. auto is inferred from --aux_sources.'
)
parser.add_argument('--aux_gate_init', type=float, default=None, help='initial gate for HSI-guided auxiliary residual. None uses dataset preset: Houston=-2.0, Augsburg=-4.0, others=-3.0')
parser.add_argument('--logit_gate_init', type=float, default=None, help='initial gate for auxiliary logit residual. None uses dataset preset: Houston=-3.0, Augsburg=-4.0, others=-3.0')
parser.add_argument('--mbce_strength', type=float, default=0.05, help='weak MBCE scale strength used only in exp5_full')
parser.add_argument('--nogate_residual_scale', type=float, default=0.1, help='fixed residual scale for exp3_cross_nogate')
parser.add_argument('--disable_aux_input_detail', action='store_true',
                    help='disable raw/smoothed/high-pass decomposition before the auxiliary CNN')
parser.add_argument('--disable_aux_sp_detail_token', action='store_true',
                    help='disable the auxiliary local-detail token based on pixel minus superpixel mean')
parser.add_argument('--disable_fusion_dynamic_gate', action='store_true',
                    help='disable sample-wise reliability scaling for auxiliary fusion')
parser.add_argument('--disable_fusion_class_gate', action='store_true',
                    help='disable class-wise auxiliary logit scaling')
parser.add_argument('--disable_fusion_agreement_gate', action='store_true',
                    help='disable cosine-agreement scaling between HSI and auxiliary summaries')
parser.add_argument('--disable_fusion_confidence_gate', action='store_true',
                    help='disable base-HSI-confidence scaling for auxiliary residual logits')
parser.add_argument('--agreement_floor', type=float, default=0.50,
                    help='minimum scale of cosine-agreement gate; lower means stronger negative-transfer suppression')
parser.add_argument('--confidence_floor', type=float, default=0.50,
                    help='minimum scale of HSI-confidence gate; lower protects high-confidence HSI samples more')
args = parser.parse_args()

# Must be set before importing the SCGR module.
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
os.environ['SCGR_DATASET_ROOT'] = str(args.dataset_root)
os.environ['CTFN_DATASET_ROOT'] = str(args.dataset_root)  # legacy compatibility

try:
    import LDA_SLIC_sparse as LDA_SLIC
    print('[import] using LDA_SLIC_sparse.py')
except ImportError:
    import LDA_SLIC
    print('[import warning] LDA_SLIC_sparse.py not found; fallback to LDA_SLIC.py. This may allocate dense Q.')

import SCGR
from functions_multimodal_v1 import train_epoch, test_epoch, output_metric, get_data, normalize


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs('./result', exist_ok=True)

ABLATION_DESCRIPTIONS = {
    'exp1_hsi_only': 'Exp-1: original SCGR, HSI-only baseline.',
    'exp2_aux_concat': 'Exp-2: SCGR + SCGR-aligned aux pixel/superpixel tokens, direct token fusion, no cross attention.',
    'exp3_cross_nogate': 'Exp-3: SCGR + HSI-guided cross attention, fixed small residual, no learned gate.',
    'exp4_gated_cross': 'Exp-4: SCGR + conservative HSI-guided gated cross attention, no MBCE scale.',
    'exp5_full': 'Exp-5: SCGR + conservative HSI-guided gated cross attention + weak neutral MBCE scale.',
}


def safe_name(text):
    return str(text).replace(',', '_').replace('+', '_').replace('/', '_').replace(' ', '').lower()


def bool_from_arg(text):
    text = str(text).strip().lower()
    if text in ['true', '1', 'yes', 'y']:
        return True
    if text in ['false', '0', 'no', 'n']:
        return False
    if text == 'auto':
        return None
    raise ValueError(f'Invalid boolean-like value: {text}')


def normalize_source_name(name):
    name = str(name).strip().lower().replace('-', '_')
    aliases = {
        'li': 'lidar',
        'lidar': 'lidar',
        'laser': 'lidar',
        'sar': 'sar',
        'sarhr': 'sar',
        'sar_hr': 'sar',
        'dsm': 'dsm',
        'dem': 'dsm',
    }
    return aliases.get(name, name)


def parse_aux_sources(dataset, aux_sources):
    text = str(aux_sources).strip().lower().replace('+', ',').replace(';', ',')
    text = text.replace('-', '_')
    if text in ['', 'none', 'no', 'false', '0']:
        return []
    if text == 'auto':
        if dataset.lower() in ['houston', 'trento']:
            return ['lidar']
        if dataset.lower() == 'augsburg':
            return ['sar']
        return []
    if text == 'all':
        return ['lidar', 'sar', 'dsm']

    combo_aliases = {
        'sar_dsm': ['sar', 'dsm'],
        'sard_sm': ['sar', 'dsm'],
        'sardsm': ['sar', 'dsm'],
        'lidar_sar': ['lidar', 'sar'],
        'lidar_dsm': ['lidar', 'dsm'],
        'lidar_sar_dsm': ['lidar', 'sar', 'dsm'],
        'lidar_sar_dsm': ['lidar', 'sar', 'dsm'],
    }
    if text in combo_aliases:
        return combo_aliases[text]

    out = []
    for item in text.split(','):
        item = normalize_source_name(item)
        if item == '':
            continue
        if item not in ['lidar', 'sar', 'dsm']:
            raise ValueError(f"Unsupported auxiliary source '{item}'. Use lidar, sar, dsm, auto, none, or all.")
        if item not in out:
            out.append(item)
    return out


def aux_sources_key(source_names):
    if not source_names:
        return 'none'
    order = {'lidar': 0, 'sar': 1, 'dsm': 2}
    clean = []
    for name in source_names:
        name = normalize_source_name(name)
        if name not in clean:
            clean.append(name)
    clean = sorted(clean, key=lambda name: order.get(name, 99))
    return '_'.join(clean)


def resolve_aux_profile(dataset, source_names):
    text = str(args.aux_profile).strip().lower().replace('-', '_')
    if text not in ['', 'auto']:
        return text
    key = aux_sources_key(source_names)
    if key == 'none':
        return 'generic'
    return key


def apply_dataset_multimodal_presets():
    """Apply source-aware conservative defaults.

    The important change for Augsburg+DSM is not to force the auxiliary branch to be weaker everywhere.
    DSM needs enough residual strength to move OA/Kappa, but the new agreement/confidence gates prevent
    broad negative transfer on samples where DSM disagrees with the HSI representation.
    """
    dataset_key = str(args.dataset).strip().lower()
    requested_sources = parse_aux_sources(args.dataset, args.aux_sources)
    source_key = aux_sources_key(requested_sources)

    if args.fusion_direction == 'auto':
        # Default to one-way fusion. HSI is the query/main stream; auxiliary data only enters as gated K/V residual.
        args.fusion_direction = 'hsi_to_aux'

    if args.aux_nhid <= 0:
        if dataset_key == 'augsburg' and source_key == 'dsm':
            args.aux_nhid = 24
        else:
            args.aux_nhid = 16

    if args.aux_gate_init is None:
        if dataset_key == 'houston':
            args.aux_gate_init = -2.0
        elif dataset_key == 'augsburg' and source_key == 'dsm':
            args.aux_gate_init = -3.0
        elif dataset_key == 'augsburg' and source_key == 'sar_dsm':
            args.aux_gate_init = -3.2
        elif dataset_key == 'augsburg':
            args.aux_gate_init = -3.5
        else:
            args.aux_gate_init = -3.0

    if args.logit_gate_init is None:
        if dataset_key == 'augsburg' and source_key == 'dsm':
            args.logit_gate_init = -2.5
        elif dataset_key == 'augsburg' and source_key == 'sar_dsm':
            args.logit_gate_init = -2.7
        elif dataset_key == 'augsburg':
            args.logit_gate_init = -3.0
        else:
            args.logit_gate_init = -3.0

    if dataset_key == 'augsburg' and source_key == 'dsm':
        print('[preset] Augsburg DSM-only: one-way hsi_to_aux fusion, aux_nhid=24, gates aux=-3.0/logit=-2.5 unless overridden.')
    elif dataset_key == 'augsburg' and source_key == 'sar_dsm':
        print('[preset] Augsburg SAR+DSM: one-way hsi_to_aux fusion with conservative source-aware gates.')
    elif dataset_key == 'augsburg' and str(args.aux_sources).strip().lower() == 'auto':
        print('[preset] Augsburg auto auxiliary source is SAR only. Use --aux_sources dsm for DSM-only or --aux_sources sar_dsm for SAR+DSM.')
    if dataset_key == 'houston' and str(args.aux_sources).strip().lower() == 'auto':
        print('[preset] Houston auto auxiliary source is LiDAR; gates use aux=-2.0, logit=-3.0 unless overridden.')

def resolve_use_aux(ablation_mode, use_aux, aux_sources):
    requested = bool_from_arg(use_aux)
    parsed = parse_aux_sources(args.dataset, aux_sources)

    if ablation_mode == 'exp1_hsi_only':
        if requested is True or len(parsed) > 0:
            print('[aux warning] ablation_mode=exp1_hsi_only forces HSI-only; auxiliary input will be ignored.')
        return False, []

    if requested is False:
        return False, []

    if len(parsed) == 0:
        return False, []

    # requested is True or auto. Both use auxiliary data when sources are non-empty.
    return True, parsed


def find_dataset_dir(dataset):
    candidates = [
        Path(args.dataset_root) / dataset,
        Path('./dataset') / dataset,
        Path('../dataset') / dataset,
        Path('/root/autodl-tmp/agcl-main/dataset') / dataset,
        Path('/root/autodl-tmp/SCGR/dataset') / dataset,
        Path('/root/autodl-tmp/CTFN/dataset') / dataset,  # legacy path
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f'Cannot find dataset directory for {dataset}. Tried: ' + ', '.join(str(p) for p in candidates)
    )


def candidate_aux_files(dataset, source_name):
    source_name = normalize_source_name(source_name)
    dataset_key = dataset.lower()
    mapping = {
        'houston': {
            'lidar': ['LiDAR.mat', 'Lidar.mat', 'lidar.mat'],
            'sar': ['SAR.mat', 'sar.mat', 'data_SAR_HR.mat'],
            'dsm': ['DSM.mat', 'dsm.mat', 'data_DSM.mat'],
        },
        'trento': {
            'lidar': ['LiDAR.mat', 'Lidar.mat', 'lidar.mat'],
            'sar': ['SAR.mat', 'sar.mat', 'data_SAR_HR.mat'],
            'dsm': ['DSM.mat', 'dsm.mat', 'data_DSM.mat'],
        },
        'augsburg': {
            'lidar': ['LiDAR.mat', 'Lidar.mat', 'lidar.mat'],
            'sar': ['data_SAR_HR.mat', 'SAR.mat', 'sar.mat'],
            'dsm': ['data_DSM.mat', 'DSM.mat', 'dsm.mat'],
        },
    }
    return mapping.get(dataset_key, {}).get(source_name, [f'{source_name}.mat'])


def choose_mat_array(mat_dict, source_name, file_stem):
    numeric_items = []
    key_hints = [source_name.lower(), file_stem.lower(), file_stem.lower().replace('data_', '')]
    for k, v in mat_dict.items():
        if k.startswith('__'):
            continue
        arr = np.asarray(v)
        if not np.issubdtype(arr.dtype, np.number):
            continue
        if arr.ndim < 2:
            continue
        numeric_items.append((k, arr))
    if not numeric_items:
        raise ValueError(f'No numeric 2D/3D array found in {file_stem}.mat')

    for hint in key_hints:
        for k, arr in numeric_items:
            if hint and hint in k.lower():
                return k, arr

    return max(numeric_items, key=lambda item: item[1].size)


def to_hwc_array(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3:
        # CHW usually has a small first dimension; HWC usually has a small last dimension.
        if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f'Unsupported auxiliary array shape after squeeze: {arr.shape}')
    return arr.astype(np.float32)


def normalize_aux_array(arr):
    arr = to_hwc_array(arr)
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[-1]):
        x = arr[:, :, c].astype(np.float32)
        finite = np.isfinite(x)
        if not finite.any():
            out[:, :, c] = 0.0
            continue
        valid = x[finite]
        lo = float(np.percentile(valid, 1.0))
        hi = float(np.percentile(valid, 99.0))
        # Fall back to exact min/max when the robust range collapses.
        if hi - lo < 1e-12:
            lo = float(np.min(valid))
            hi = float(np.max(valid))
        if hi - lo < 1e-12:
            out[:, :, c] = 0.0
        else:
            y = (np.clip(x, lo, hi) - lo) / (hi - lo)
            y[~finite] = 0.0
            out[:, :, c] = y
    return out.astype(np.float32)


def load_one_aux_source(dataset, source_name):
    import scipy.io as sio
    dataset_dir = find_dataset_dir(dataset)
    source_name = normalize_source_name(source_name)
    tried = []
    for filename in candidate_aux_files(dataset, source_name):
        path = dataset_dir / filename
        tried.append(str(path))
        if path.exists():
            mat = sio.loadmat(str(path))
            key, arr = choose_mat_array(mat, source_name, path.stem)
            arr = normalize_aux_array(arr)
            print(f'[aux] loaded {source_name}: file={path}, key={key}, HWC_shape={arr.shape}')
            return arr
    raise FileNotFoundError(
        f"Cannot find auxiliary source '{source_name}' for {dataset}. Tried files: {tried}"
    )


def prepare_auxiliary_input(dataset, source_names, device, allow_missing=False):
    if len(source_names) == 0:
        print('[aux] disabled')
        return None, 0, []

    aux_arrays = []
    loaded_names = []
    for src in source_names:
        try:
            arr = load_one_aux_source(dataset, src)
            aux_arrays.append(arr)
            loaded_names.append(src)
        except FileNotFoundError as exc:
            if allow_missing:
                print(f'[aux warning] skip missing source {src}: {exc}')
                continue
            raise

    if len(aux_arrays) == 0:
        print('[aux] disabled because no requested auxiliary source was loaded')
        return None, 0, []

    aux_changel = int(sum(arr.shape[-1] for arr in aux_arrays))
    aux_tensors = [torch.from_numpy(arr).float().to(device) for arr in aux_arrays]
    aux_input = aux_tensors[0] if len(aux_tensors) == 1 else aux_tensors
    print(f'[aux] using sources={loaded_names}, total_aux_channels={aux_changel}')
    return aux_input, aux_changel, loaded_names


# -------------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------------
def seed_everything(seed: int, strict: bool = False):
    """Reset all known RNG states used by this script.

    Call this once before data preparation and again at the beginning of each run.
    The second call is important: model initialization, DataLoader shuffling,
    dropout masks, and superpixel generation can all consume random numbers.
    """
    seed = int(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    cudnn.allow_tf32 = False

    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_label_loader(dataset, batch_size, shuffle, seed, num_workers=0):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return Data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
        drop_last=False,
    )


seed_everything(args.seed, strict=args.strict_repro)

# -------------------------------------------------------------------------------
# Prepare data
# -------------------------------------------------------------------------------
input_data, num_classes, total_pos_train, total_pos_test, total_pos_true, \
    y_train, y_test, y_true, gt_reshape, y_train_flatten, y_test_flatten = get_data(args.dataset)

input_normalize = normalize(input_data)
height, width, band = input_normalize.shape
print('height={0},width={1},band={2}'.format(height, width, band))

input_numpy = np.array(input_normalize, dtype=np.float32)
input_normalize = torch.from_numpy(input_numpy).to(device)

apply_dataset_multimodal_presets()
use_aux_effective, requested_source_names = resolve_use_aux(args.ablation_mode, args.use_aux, args.aux_sources)
aux_input, aux_changel, aux_source_names = prepare_auxiliary_input(
    args.dataset,
    requested_source_names if use_aux_effective else [],
    device,
    allow_missing=args.allow_missing_aux,
)
use_aux_effective = use_aux_effective and aux_changel > 0
resolved_aux_profile = resolve_aux_profile(args.dataset, aux_source_names if use_aux_effective else [])

if not use_aux_effective:
    effective_ablation_mode = 'exp1_hsi_only'
    aux_name_for_file = 'none'
else:
    effective_ablation_mode = args.ablation_mode
    aux_name_for_file = '_'.join(aux_source_names)

result_file = os.path.join('./result', f'{args.dataset}_{effective_ablation_mode}_{safe_name(aux_name_for_file)}_results.txt')
print('================ Experiment setting ================')
print(f'Dataset: {args.dataset}')
print(f'Dataset root: {args.dataset_root}')
print(f'Ablation argument: {args.ablation_mode}')
print(f'Effective ablation mode: {effective_ablation_mode}')
print(f'Use auxiliary: {use_aux_effective}')
print(f'Auxiliary sources: {aux_source_names if use_aux_effective else []}')
print(f'Fusion setting: dim={args.fusion_dim}, heads={args.fusion_heads}, depth={args.fusion_depth}, dropout={args.fusion_dropout}, direction={args.fusion_direction}')
print(f'Aux branch setting: aux_nhid={args.aux_nhid}, aux_profile={resolved_aux_profile}, aux_gate_init={args.aux_gate_init}, logit_gate_init={args.logit_gate_init}, mbce_strength={args.mbce_strength}, nogate_residual_scale={args.nogate_residual_scale}')
print(f'Aux/fusion enhancements: input_detail={not args.disable_aux_input_detail}, sp_detail_token={not args.disable_aux_sp_detail_token}, dynamic_gate={not args.disable_fusion_dynamic_gate}, class_gate={not args.disable_fusion_class_gate}, agreement_gate={not args.disable_fusion_agreement_gate}, confidence_gate={not args.disable_fusion_confidence_gate}')
print(f'Result file: {result_file}')
print(f'Reproducibility: seed={args.seed}, same_seed_each_run={args.same_seed_each_run}, strict_repro={args.strict_repro}')
print('====================================================')

# -------------------------------------------------------------------------------
# Label loaders
# -------------------------------------------------------------------------------
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

Label_train = Data.TensorDataset(y_train, y_train_flatten)
Label_test = Data.TensorDataset(y_test, y_test_flatten)
# DataLoader is deliberately created inside each run with a run-specific generator.
# Creating it once outside the loop makes the shuffle RNG state drift across runs.

# -------------------------------------------------------------------------------
# Effective superpixel scale
# -------------------------------------------------------------------------------
min_scale_for_limit = int(np.ceil((height * width) / max(args.max_superpixels, 1)))
effective_scale = max(args.superpixel_scale, min_scale_for_limit)

if effective_scale != args.superpixel_scale:
    print(
        f'[auto superpixel scale] requested scale={args.superpixel_scale}, '
        f'but image is large. Use scale={effective_scale} to keep superpixels <= {args.max_superpixels}.'
    )
else:
    print(f'[superpixel scale] use scale={effective_scale}')

# -------------------------------------------------------------------------------
# Run
# -------------------------------------------------------------------------------
results = []
for run in range(args.num_run):
    print('\n**************************************************')
    print(f'Run {run + 1}/{args.num_run}')
    print('**************************************************')

    run_seed = int(args.seed if args.same_seed_each_run else args.seed + run)
    seed_everything(run_seed, strict=args.strict_repro)
    print(f'[repro] run_seed={run_seed}')

    label_train_loader = make_label_loader(
        Label_train, args.batch_size, shuffle=True, seed=run_seed, num_workers=0
    )
    label_test_loader = make_label_loader(
        Label_test, args.batch_size, shuffle=False, seed=run_seed, num_workers=0
    )

    best_OA2 = 0.0
    best_AA_mean2 = 0.0
    best_Kappa2 = 0.0
    best_AA2 = []

    train_samples_gt = np.zeros(height * width, dtype=np.int64)
    y_train_flatten_np = y_train_flatten.numpy()
    for idx in y_train_flatten_np:
        train_samples_gt[int(idx)] = int(gt_reshape[int(idx)])

    train_samples_gt_2d = np.reshape(train_samples_gt, [height, width])

    scale_now = effective_scale
    for attempt in range(5):
        ls = LDA_SLIC.LDA_SLIC(input_numpy, train_samples_gt_2d, num_classes - 1)
        Q, S, A, Edge_index, Edge_atter, Seg = ls.simple_superpixel(scale=scale_now)
        SP_size = int(S.shape[0])
        if SP_size <= args.max_superpixels:
            break
        ratio = SP_size / max(args.max_superpixels, 1)
        scale_now = int(np.ceil(scale_now * ratio * 1.08))
        scale_now = max(scale_now, int(scale_now + 1))
        print(
            f'[Warning] superpixel_count={SP_size} > max_superpixels={args.max_superpixels}. '
            f'Retry with adaptive scale={scale_now}.'
        )
    else:
        raise RuntimeError(
            f'Unable to keep superpixel_count <= {args.max_superpixels}. '
            f'Last SP_size={SP_size}. Please increase --superpixel_scale or --max_superpixels.'
        )

    print(f'[superpixel] final scale={scale_now}, SP_size={SP_size}')

    A = torch.from_numpy(A).float().to(device)
    S = torch.from_numpy(S).float().to(device)
    Edge_index = torch.from_numpy(Edge_index).long().to(device)
    Edge_atter = torch.from_numpy(Edge_atter).long().to(device)
    Seg = torch.from_numpy(Seg).long().to(device)

    CNN_nhid = args.cnn_nhid
    if use_aux_effective:
        net = SCGR.SCGR_MM(
            height,
            width,
            band,
            num_classes,
            None,
            A,
            S,
            Edge_index,
            Edge_atter,
            SP_size,
            CNN_nhid,
            Seg=Seg,
            aux_changel=aux_changel,
            aux_nhid=(args.aux_nhid if args.aux_nhid > 0 else None),
            fusion_dim=args.fusion_dim,
            fusion_heads=args.fusion_heads,
            fusion_depth=args.fusion_depth,
            fusion_dropout=args.fusion_dropout,
            ablation_mode=effective_ablation_mode,
            aux_gate_init=args.aux_gate_init,
            logit_gate_init=args.logit_gate_init,
            mbce_strength=args.mbce_strength,
            nogate_residual_scale=args.nogate_residual_scale,
            aux_use_input_detail=(not args.disable_aux_input_detail),
            aux_use_sp_detail_token=(not args.disable_aux_sp_detail_token),
            fusion_dynamic_gate=(not args.disable_fusion_dynamic_gate),
            fusion_class_gate=(not args.disable_fusion_class_gate),
            aux_profile=resolved_aux_profile,
            fusion_direction=args.fusion_direction,
            fusion_agreement_gate=(not args.disable_fusion_agreement_gate),
            fusion_confidence_gate=(not args.disable_fusion_confidence_gate),
            agreement_floor=args.agreement_floor,
            confidence_floor=args.confidence_floor,
        )
    else:
        net = SCGR.SCGR(
            height,
            width,
            band,
            num_classes,
            None,
            A,
            S,
            Edge_index,
            Edge_atter,
            SP_size,
            CNN_nhid,
            Seg=Seg,
        )

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epoches, eta_min=args.eta_min
        )
    else:
        scheduler = None

    best_epoch = 0
    best_score = -1.0
    no_improve_epochs = 0

    torch.cuda.empty_cache()
    for epoch in range(args.epoches):
        net.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(
            net,
            input_normalize,
            label_train_loader,
            criterion,
            optimizer,
            aux_input=(aux_input if use_aux_effective else None),
        )
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)

        if (epoch + 1) % 10 == 0:
            print('Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}'.format(
                epoch + 1, train_obj, train_acc
            ))

        do_test = (((epoch + 1) % args.test_freq == 0) or (epoch == 0) or (epoch == args.epoches - 1)) \
                  and ((epoch + 1) >= args.eval_start_epoch)

        if do_test:
            net.eval()
            tar_v, pre_v = test_epoch(
                net,
                input_normalize,
                label_test_loader,
                criterion,
                aux_input=(aux_input if use_aux_effective else None),
            )
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            print('Test Epoch: {:03d} OA: {:.2f} AA: {:.2f} Kappa: {:.2f}'.format(
                epoch + 1, OA2 * 100, AA_mean2 * 100, Kappa2 * 100
            ))

            if args.best_metric == 'OA':
                score = OA2
            elif args.best_metric == 'AA':
                score = AA_mean2
            elif args.best_metric == 'Kappa':
                score = Kappa2
            else:
                score = OA2 + AA_mean2 + Kappa2

            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                best_OA2 = OA2
                best_AA_mean2 = AA_mean2
                best_Kappa2 = Kappa2
                best_AA2 = AA2
                no_improve_epochs = 0
            else:
                no_improve_epochs += args.test_freq

            if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                print(
                    f'Early stop at epoch {epoch + 1}. Best epoch={best_epoch}, '
                    f'best_{args.best_metric}_score={best_score:.6f}.'
                )
                break

        if scheduler is not None:
            scheduler.step()

    torch.cuda.empty_cache()

    print('\nbest_epoch:{:03d}, best_OA:{:.2f}, best_AA:{:.2f}, best_Kappa:{:.2f}'.format(
        best_epoch, best_OA2 * 100, best_AA_mean2 * 100, best_Kappa2 * 100
    ))

    results.append({
        'run': run + 1,
        'best_epoch': best_epoch,
        'OA': best_OA2 * 100,
        'AA': best_AA_mean2 * 100,
        'Kappa': best_Kappa2 * 100,
        'best_AA2': np.asarray(best_AA2) * 100,
    })

    with open(result_file, 'a+') as f:
        str_results = '\n\n************************************************' \
                      + f'\nRun={run + 1}' \
                      + f'\nDataset={args.dataset}' \
                      + f'\nAblationMode={effective_ablation_mode}' \
                      + f'\nUseAux={use_aux_effective}' \
                      + f'\nAuxSources={aux_source_names if use_aux_effective else []}' \
                      + f'\nBestEpoch={best_epoch}' \
                      + f'\nOA={best_OA2 * 100:.2f}' \
                      + f'\nAA={best_AA_mean2 * 100:.2f}' \
                      + f'\nKappa={best_Kappa2 * 100:.2f}' \
                      + '\nbest_AA2=' + str(np.around(np.asarray(best_AA2) * 100, 2))
        f.write(str_results)

# -------------------------------------------------------------------------------
# Average best results over all runs
# -------------------------------------------------------------------------------
if len(results) > 0:
    ddof = 1 if len(results) > 1 else 0

    best_epoch_values = np.asarray([r['best_epoch'] for r in results], dtype=np.float64)
    OA_values = np.asarray([r['OA'] for r in results], dtype=np.float64)
    AA_values = np.asarray([r['AA'] for r in results], dtype=np.float64)
    Kappa_values = np.asarray([r['Kappa'] for r in results], dtype=np.float64)

    mean_best_epoch = np.mean(best_epoch_values)
    std_best_epoch = np.std(best_epoch_values, ddof=ddof)
    mean_OA = np.mean(OA_values)
    std_OA = np.std(OA_values, ddof=ddof)
    mean_AA = np.mean(AA_values)
    std_AA = np.std(AA_values, ddof=ddof)
    mean_Kappa = np.mean(Kappa_values)
    std_Kappa = np.std(Kappa_values, ddof=ddof)

    def mean_std_str(mean_value, std_value):
        return f'{mean_value:.2f}±{std_value:.2f}'

    valid_AA2 = [r['best_AA2'] for r in results if len(r['best_AA2']) > 0]
    mean_AA2 = None
    std_AA2 = None
    if len(valid_AA2) == len(results):
        AA2_stack = np.stack(valid_AA2, axis=0)
        mean_AA2 = np.mean(AA2_stack, axis=0)
        std_AA2 = np.std(AA2_stack, axis=0, ddof=ddof)

    print('\n================ Average of best results ================')
    print(f'Runs: {len(results)}')
    print(f'BestEpoch: {mean_std_str(mean_best_epoch, std_best_epoch)}')
    print(f'OA: {mean_std_str(mean_OA, std_OA)}')
    print(f'AA: {mean_std_str(mean_AA, std_AA)}')
    print(f'Kappa: {mean_std_str(mean_Kappa, std_Kappa)}')
    if mean_AA2 is not None:
        mean_std_AA2 = [mean_std_str(m, s) for m, s in zip(mean_AA2, std_AA2)]
        print('best_AA2:', mean_std_AA2)

    with open(result_file, 'a+') as f:
        avg_results = '\n\n================ Average of best results ================' \
                      + f'\nRuns={len(results)}' \
                      + f'\nDataset={args.dataset}' \
                      + f'\nAblationMode={effective_ablation_mode}' \
                      + f'\nUseAux={use_aux_effective}' \
                      + f'\nAuxSources={aux_source_names if use_aux_effective else []}' \
                      + f'\nMeanBestEpoch={mean_best_epoch:.2f}' \
                      + f'\nStdBestEpoch={std_best_epoch:.2f}' \
                      + f'\nBestEpoch_mean±std={mean_std_str(mean_best_epoch, std_best_epoch)}' \
                      + f'\nMeanOA={mean_OA:.2f}' \
                      + f'\nStdOA={std_OA:.2f}' \
                      + f'\nOA_mean±std={mean_std_str(mean_OA, std_OA)}' \
                      + f'\nMeanAA={mean_AA:.2f}' \
                      + f'\nStdAA={std_AA:.2f}' \
                      + f'\nAA_mean±std={mean_std_str(mean_AA, std_AA)}' \
                      + f'\nMeanKappa={mean_Kappa:.2f}' \
                      + f'\nStdKappa={std_Kappa:.2f}' \
                      + f'\nKappa_mean±std={mean_std_str(mean_Kappa, std_Kappa)}'
        if mean_AA2 is not None:
            mean_std_AA2 = [mean_std_str(m, s) for m, s in zip(mean_AA2, std_AA2)]
            avg_results += '\nMean_best_AA2=' + str(np.around(mean_AA2, 2))
            avg_results += '\nStd_best_AA2=' + str(np.around(std_AA2, 2))
            avg_results += '\nbest_AA2_mean±std=' + str(mean_std_AA2)
        f.write(avg_results)
