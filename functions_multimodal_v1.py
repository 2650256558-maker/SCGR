from sklearn.metrics import confusion_matrix
import numpy as np
import scipy.io as sio
import torch
import os
from sklearn import preprocessing


# ============================================================
# Dataset root
# ============================================================
# 默认数据目录：
#   ./dataset/Houston
#   ./dataset/Trento
#   ./dataset/Augsburg
#
# 如需改路径，可设置环境变量：
#   export CTFN_DATASET_ROOT=/your/path/to/dataset
# ============================================================
DATASET_ROOT = os.environ.get("CTFN_DATASET_ROOT", "./dataset")


# ============================================================
# Basic utilities
# ============================================================
def normalize(data):
    """
    对 HSI 做全局 band-wise 标准化。
    输入: data, shape = [H, W, C]
    输出: data, shape = [H, W, C]
    """
    height, width, bands = data.shape
    data = np.reshape(data, [height * width, bands])
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    data = np.reshape(data, [height, width, bands])
    return data


def get_label(indices, gt_hsi):
    """根据二维坐标索引读取标签。indices: [N, 2]，每行是 [row, col]。"""
    dim_0 = indices[:, 0]
    dim_1 = indices[:, 1]
    label = gt_hsi[dim_0, dim_1]
    return label


def get_label_flatten(indices, width):
    """二维坐标 [row, col] 转一维像素索引 row * width + col。"""
    dim_0 = indices[:, 0]
    dim_1 = indices[:, 1]
    label = dim_0 * width + dim_1
    return label.astype(np.int64)


# ============================================================
# .mat loading helpers
# ============================================================
def _mat_valid_keys(mat):
    return [k for k in mat.keys() if not k.startswith("__")]


def _load_first_array(mat_path, prefer_keys=None, ndim=None):
    """
    从 .mat 文件中读取 ndarray。

    1. 先按 prefer_keys 优先读取。
    2. 如果变量名不确定，则自动选择符合 ndim 要求且 size 最大的 ndarray。
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"File not found: {mat_path}")

    mat = sio.loadmat(mat_path)

    if prefer_keys is not None:
        for key in prefer_keys:
            if key in mat and isinstance(mat[key], np.ndarray):
                arr = np.squeeze(mat[key])
                if ndim is None or arr.ndim == ndim:
                    print(f"[loadmat] {mat_path} -> key='{key}', shape={arr.shape}")
                    return arr

    candidates = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if not isinstance(value, np.ndarray):
            continue

        arr = np.squeeze(value)
        if ndim is not None and arr.ndim != ndim:
            continue
        if arr.dtype == np.dtype("O"):
            continue

        candidates.append((key, arr))

    if len(candidates) == 0:
        raise KeyError(
            f"No valid ndarray found in {mat_path}. "
            f"Available keys: {_mat_valid_keys(mat)}"
        )

    key, arr = max(candidates, key=lambda item: item[1].size)
    print(f"[loadmat] {mat_path} -> auto key='{key}', shape={arr.shape}")
    return arr


def _read_hsi(mat_path, prefer_keys=None):
    arr = _load_first_array(mat_path, prefer_keys=prefer_keys, ndim=3)
    if arr.ndim != 3:
        raise ValueError(f"HSI data must be 3D, but got shape {arr.shape} from {mat_path}")
    return arr.astype(np.float32)


def _read_label(mat_path, prefer_keys=None):
    arr = _load_first_array(mat_path, prefer_keys=prefer_keys, ndim=2)
    if arr.ndim != 2:
        raise ValueError(f"Label data must be 2D, but got shape {arr.shape} from {mat_path}")
    return arr.astype(np.int64)


def _sanitize_label_map(label_map):
    """
    统一标签格式：
    - 0 表示背景/未标注；
    - 负数，例如 Houston 中的 -1，也视作背景；
    - NaN / inf 也视作背景。
    """
    arr = np.asarray(label_map).copy()
    arr = np.nan_to_num(arr, nan=0, posinf=0, neginf=0).astype(np.int64)
    arr[arr <= 0] = 0
    return arr


def _merge_train_test_label(train_gt, test_gt):
    """Augsburg 没有单独 gt.mat 时，用 TrainImage + TestImage 合成 gt。"""
    train_gt = _sanitize_label_map(train_gt)
    test_gt = _sanitize_label_map(test_gt)

    if train_gt.shape != test_gt.shape:
        raise ValueError(
            f"Train/Test label shape mismatch: train={train_gt.shape}, test={test_gt.shape}"
        )

    conflict = (train_gt > 0) & (test_gt > 0) & (train_gt != test_gt)
    if np.any(conflict):
        raise ValueError("TrainImage and TestImage have conflicting labels at some pixels.")

    gt = train_gt.copy()
    mask = gt == 0
    gt[mask] = test_gt[mask]
    return gt.astype(np.int64)


def _match_hsi_to_label_shape(data_hsi, label_shape):
    """
    保证 HSI 的前两个维度是空间维度 [H, W]。
    如果 .mat 文件中存为 [C, H, W] 等格式，这里尝试转为 [H, W, C]。
    """
    if data_hsi.shape[:2] == label_shape:
        return data_hsi.astype(np.float32)

    perms = [
        (1, 2, 0),
        (0, 2, 1),
        (1, 0, 2),
        (2, 0, 1),
        (2, 1, 0),
    ]

    for perm in perms:
        transposed = np.transpose(data_hsi, perm)
        if transposed.shape[:2] == label_shape:
            print(f"[transpose] HSI shape {data_hsi.shape} -> {transposed.shape}, perm={perm}")
            return transposed.astype(np.float32)

    raise ValueError(
        f"HSI spatial shape does not match label shape. "
        f"HSI shape={data_hsi.shape}, label shape={label_shape}. "
        f"Please check whether HSI and label maps have the same spatial resolution. "
        f"For Augsburg, data_HS_LR may be lower-resolution than TrainImage/TestImage."
    )


def _print_label_stat(name, arr):
    values, counts = np.unique(arr, return_counts=True)
    info = ", ".join([f"{int(v)}:{int(c)}" for v, c in zip(values, counts) if int(v) != 0])
    if info == "":
        info = "no positive labels"
    print(f"[label stat] {name}: {info}")


def _remap_label_maps(gt_hsi, train_gt, test_gt):
    """
    将有效类别重映射为连续的 1...C。

    重要修正：
    - Houston 的 gt 中可能有 -1，必须视作背景 0，不能当成一个类别。
    - 有效类别以 train_gt 与 test_gt 中实际出现的正标签为准。
    - gt_hsi 中不属于有效类别的像素会被置为 0。
    """
    gt_hsi = _sanitize_label_map(gt_hsi)
    train_gt = _sanitize_label_map(train_gt)
    test_gt = _sanitize_label_map(test_gt)

    _print_label_stat("gt_hsi before remap", gt_hsi)
    _print_label_stat("train_gt before remap", train_gt)
    _print_label_stat("test_gt before remap", test_gt)

    train_labels = set(int(x) for x in np.unique(train_gt).tolist() if int(x) > 0)
    test_labels = set(int(x) for x in np.unique(test_gt).tolist() if int(x) > 0)
    labels = sorted(list(train_labels | test_labels))

    if len(labels) == 0:
        raise ValueError("No positive labels found in train_gt and test_gt.")

    only_test = sorted(list(test_labels - train_labels))
    if len(only_test) > 0:
        raise ValueError(
            f"These classes appear in test_gt but not in train_gt: {only_test}. "
            f"The model cannot learn classes with zero training samples."
        )

    mapping = {old: new for new, old in enumerate(labels, start=1)}
    expected = list(range(1, len(labels) + 1))

    print(f"[effective labels] original labels used for classification: {labels}")
    if labels != expected:
        print(f"[label remap] mapping={mapping}")
    else:
        print("[label remap] labels are already continuous 1...C")

    def remap_one(label_map):
        out = np.zeros_like(label_map, dtype=np.int64)
        for old, new in mapping.items():
            out[label_map == old] = new
        return out

    gt_hsi = remap_one(gt_hsi)
    train_gt = remap_one(train_gt)
    test_gt = remap_one(test_gt)

    _print_label_stat("gt_hsi after remap", gt_hsi)
    _print_label_stat("train_gt after remap", train_gt)
    _print_label_stat("test_gt after remap", test_gt)

    return gt_hsi.astype(np.int64), train_gt.astype(np.int64), test_gt.astype(np.int64), len(labels)


# ============================================================
# Dataset loading
# ============================================================
def load_dataset(Dataset):
    """只读取 HSI，不读取 LiDAR / SAR / DSM。"""
    root = os.path.join(DATASET_ROOT, Dataset)

    if Dataset == "Houston":
        data_hsi = _read_hsi(
            os.path.join(root, "HSI.mat"),
            prefer_keys=["HSI", "hsi", "Houston", "data", "Data"]
        )
        gt_hsi = _read_label(
            os.path.join(root, "gt.mat"),
            prefer_keys=["gt", "GT", "label", "labels", "groundtruth"]
        )
        train_gt = _read_label(
            os.path.join(root, "TRLabel.mat"),
            prefer_keys=["TRLabel", "tr_label", "train", "train_label", "TrainImage"]
        )
        test_gt = _read_label(
            os.path.join(root, "TSLabel.mat"),
            prefer_keys=["TSLabel", "ts_label", "test", "test_label", "TestImage"]
        )

    elif Dataset == "Trento":
        data_hsi = _read_hsi(
            os.path.join(root, "HSI.mat"),
            prefer_keys=["HSI", "hsi", "Trento", "data", "Data"]
        )
        gt_hsi = _read_label(
            os.path.join(root, "gt.mat"),
            prefer_keys=["gt", "GT", "label", "labels", "groundtruth"]
        )
        train_gt = _read_label(
            os.path.join(root, "TRLabel.mat"),
            prefer_keys=["TRLabel", "tr_label", "train", "train_label", "TrainImage"]
        )
        test_gt = _read_label(
            os.path.join(root, "TSLabel.mat"),
            prefer_keys=["TSLabel", "ts_label", "test", "test_label", "TestImage"]
        )

    elif Dataset == "Augsburg":
        data_hsi = _read_hsi(
            os.path.join(root, "data_HS_LR.mat"),
            prefer_keys=["data_HS_LR", "HSI", "hsi", "Augsburg", "data", "Data"]
        )
        train_gt = _read_label(
            os.path.join(root, "TrainImage.mat"),
            prefer_keys=["TrainImage", "TRLabel", "tr_label", "train", "train_label"]
        )
        test_gt = _read_label(
            os.path.join(root, "TestImage.mat"),
            prefer_keys=["TestImage", "TSLabel", "ts_label", "test", "test_label"]
        )
        gt_hsi = _merge_train_test_label(train_gt, test_gt)

    else:
        raise ValueError(
            f"Unsupported dataset: {Dataset}. Supported datasets: Houston, Trento, Augsburg."
        )

    data_hsi = _match_hsi_to_label_shape(data_hsi, gt_hsi.shape)

    if train_gt.shape != gt_hsi.shape:
        raise ValueError(f"train_gt shape {train_gt.shape} != gt_hsi shape {gt_hsi.shape}")
    if test_gt.shape != gt_hsi.shape:
        raise ValueError(f"test_gt shape {test_gt.shape} != gt_hsi shape {gt_hsi.shape}")

    gt_hsi, train_gt, test_gt, classes_num = _remap_label_maps(gt_hsi, train_gt, test_gt)

    print("**************************************************")
    print(f"Dataset: {Dataset}")
    print(f"Dataset root: {root}")
    print(f"HSI shape: {data_hsi.shape}")
    print(f"GT shape: {gt_hsi.shape}")
    print(f"Train label shape: {train_gt.shape}")
    print(f"Test label shape: {test_gt.shape}")
    print(f"Classes: {classes_num}")
    print(f"Train labeled pixels: {int(np.sum(train_gt > 0))}")
    print(f"Test labeled pixels: {int(np.sum(test_gt > 0))}")
    print(f"Total labeled pixels: {int(np.sum(gt_hsi > 0))}")
    print("**************************************************")

    return data_hsi, gt_hsi, train_gt, test_gt, classes_num


def _indices_from_label_map(label_map, num_classes, name="label"):
    all_indices = []
    class_num = []

    for c in range(1, num_classes + 1):
        idx = np.argwhere(label_map == c)
        class_num.append(idx.shape[0])
        if idx.shape[0] == 0:
            print(f"[Warning] {name}: class {c} has 0 samples.")
        else:
            all_indices.append(idx)

    if len(all_indices) == 0:
        raise ValueError(f"No labeled pixels found in {name}.")

    all_indices = np.concatenate(all_indices, axis=0).astype(int)
    np.random.shuffle(all_indices)
    return all_indices, class_num


def get_data(dataset):
    """MAIN 调用的数据接口。"""
    data_hsi, gt_hsi, train_gt, test_gt, CLASSES_NUM = load_dataset(dataset)

    train_indices, train_num = _indices_from_label_map(train_gt, CLASSES_NUM, name="train_gt")
    test_indices, test_num = _indices_from_label_map(test_gt, CLASSES_NUM, name="test_gt")
    total_indices = np.argwhere(gt_hsi > 0).astype(int)

    y_train = get_label(train_indices, train_gt).astype(np.int64) - 1
    y_test = get_label(test_indices, test_gt).astype(np.int64) - 1
    y_true = get_label(total_indices, gt_hsi).astype(np.int64) - 1

    height, width = gt_hsi.shape
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )

    y_train_flatten = get_label_flatten(train_indices, width)
    y_test_flatten = get_label_flatten(test_indices, width)

    # 检查官方训练/测试划分是否重叠。正常应为 0。
    train_flat_set = set(y_train_flatten.tolist())
    test_flat_set = set(y_test_flatten.tolist())
    overlap = train_flat_set & test_flat_set
    if len(overlap) > 0:
        raise ValueError(
            f"Train/Test overlap pixels: {len(overlap)}. "
            f"Please check TRLabel/TSLabel or TrainImage/TestImage."
        )

    # 返回 CPU tensor，DataLoader 后在 train_epoch/test_epoch 中再搬到 device。
    y_train_flatten = torch.from_numpy(y_train_flatten).long()
    y_test_flatten = torch.from_numpy(y_test_flatten).long()

    print("**************************************************")
    print(f"train samples: {len(y_train)}")
    print(f"test samples: {len(y_test)}")
    print(f"total labeled samples: {len(y_true)}")
    print(f"train/test overlap pixels: 0")
    print(f"train per class: {train_num}")
    print(f"test per class: {test_num}")
    print("**************************************************")

    return (
        data_hsi,
        CLASSES_NUM,
        train_indices,
        test_indices,
        total_indices,
        y_train,
        y_test,
        y_true,
        gt,
        y_train_flatten,
        y_test_flatten,
    )


# ============================================================
# Training / testing utilities
# ============================================================
class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.item()
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res, target, pred.squeeze()


def train_epoch(net, input_normalize, train_loader, criterion, optimizer, aux_input=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    device = input_normalize.device

    for batch_idx, (batch_target, y_train_flatten) in enumerate(train_loader):
        batch_target = batch_target.long().to(device)
        y_train_flatten = y_train_flatten.long().to(device)

        optimizer.zero_grad()
        if aux_input is None:
            batch_pred = net(input_normalize, y_train_flatten)
        else:
            batch_pred = net(input_normalize, y_train_flatten, aux=aux_input)
        loss = criterion(batch_pred, batch_target)

        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = y_train_flatten.shape[0]

        objs.update(loss.detach(), n)
        top1.update(prec1[0].detach(), n)

        tar = np.append(tar, t.detach().cpu().numpy())
        pre = np.append(pre, p.detach().cpu().numpy())

    return top1.avg, objs.avg, tar, pre


def test_epoch(net, input_normalize, test_loader, criterion, aux_input=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    device = input_normalize.device

    with torch.no_grad():
        for batch_idx, (batch_target, y_test_flatten) in enumerate(test_loader):
            batch_target = batch_target.long().to(device)
            y_test_flatten = y_test_flatten.long().to(device)

            if aux_input is None:
                batch_pred = net(input_normalize, y_test_flatten)
            else:
                batch_pred = net(input_normalize, y_test_flatten, aux=aux_input)
            loss = criterion(batch_pred, batch_target)

            prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
            n = y_test_flatten.shape[0]

            objs.update(loss.detach(), n)
            top1.update(prec1[0].detach(), n)

            tar = np.append(tar, t.detach().cpu().numpy())
            pre = np.append(pre, p.detach().cpu().numpy())

    return tar, pre


# ============================================================
# Metrics
# ============================================================
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


def cal_results(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    shape = np.shape(matrix)

    if matrix.size == 0 or np.sum(matrix) == 0:
        raise ValueError("Empty confusion matrix, cannot calculate metrics.")

    correct = np.trace(matrix)
    total = np.sum(matrix)

    row_sum = np.sum(matrix, axis=1)
    col_sum = np.sum(matrix, axis=0)

    AA = np.zeros([shape[0]], dtype=np.float64)
    valid = row_sum > 0
    AA[valid] = np.diag(matrix)[valid] / row_sum[valid]

    OA = correct / total
    AA_mean = np.mean(AA[valid]) if np.any(valid) else 0.0

    pe = np.sum(row_sum * col_sum) / (total ** 2)
    if abs(1 - pe) < 1e-12:
        Kappa = 0.0
    else:
        Kappa = (OA - pe) / (1 - pe)

    return OA, AA_mean, Kappa, AA


def metrics(best_OA2, best_AA_mean2, best_Kappa2, AA2):
    results = {}
    results["OA"] = best_OA2 * 100.0
    results["AA"] = best_AA_mean2 * 100.0
    results["Kappa"] = best_Kappa2 * 100.0
    results["class acc"] = AA2 * 100.0
    return results


def show_results(results, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["OA"] for r in results]
        aa = [r["AA"] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)

        text += "---\n"
        text += "class acc :\n"
        for score, std in zip(class_acc_mean, class_acc_std):
            text += "\t{:.02f} +- {:.02f}\n".format(score, std)
        text += "---\n"
        text += "OA: {:.02f} +- {:.02f}\n".format(np.mean(accuracies), np.std(accuracies))
        text += "AA: {:.02f} +- {:.02f}\n".format(np.mean(aa), np.std(aa))
        text += "Kappa: {:.02f} +- {:.02f}\n".format(np.mean(kappas), np.std(kappas))

    else:
        accuracy_value = results["OA"]
        aa_value = results["AA"]
        classacc = results["class acc"]
        kappa = results["Kappa"]

        text += "---\n"
        text += "class acc :\n"
        for score in classacc:
            text += "\t {:.02f}\n".format(score)
        text += "---\n"
        text += "OA : {:.02f}%\n".format(accuracy_value)
        text += "AA: {:.02f}%\n".format(aa_value)
        text += "Kappa: {:.02f}\n".format(kappa)

    print(text)
