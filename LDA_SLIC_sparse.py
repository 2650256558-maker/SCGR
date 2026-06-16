import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.segmentation import slic, mark_boundaries, felzenszwalb, quickshift, random_walker
from sklearn import preprocessing
import cv2
import math


def LSC_superpixel(I, nseg):
    ratio = 0.075
    size = int(math.sqrt(((I.shape[0] * I.shape[1]) / nseg)))
    superpixelLSC = cv2.ximgproc.createSuperpixelLSC(
        I,
        region_size=size,
        ratio=0.005)
    superpixelLSC.iterate()
    superpixelLSC.enforceLabelConnectivity(min_element_size=25)
    segments = superpixelLSC.getLabels()
    return np.array(segments, np.int64)


def SEEDS_superpixel(I, nseg):
    I = np.array(I[:, :, 0:3], np.float32).copy()
    I_new = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
    height, width, channels = I_new.shape

    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, int(nseg), num_levels=2, prior=1,
                                               histogram_bins=5)
    seeds.iterate(I_new, 4)
    segments = seeds.getLabels()
    return segments


def SegmentsLabelProcess(labels):
    """对 segments 后处理，防止 label 不连续。"""
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    unique_labels = sorted(set(np.reshape(labels, [-1]).tolist()))
    mapping = {old: new for new, old in enumerate(unique_labels)}

    new_labels = labels.copy()
    for old, new in mapping.items():
        new_labels[labels == old] = new
    return new_labels.astype(np.int64)


class SLIC(object):
    def __init__(self, HSI, labels, n_segments=1000, compactness=20, max_iter=20, sigma=0, min_size_factor=0.3,
                 max_size_factor=2):
        self.n_segments = max(2, int(n_segments))
        self.compactness = compactness
        self.max_iter = max_iter
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.sigma = sigma

        height, width, bands = HSI.shape
        data = np.reshape(HSI, [height * width, bands])
        scaler = preprocessing.StandardScaler()
        data = scaler.fit_transform(data)
        self.data = np.reshape(data, [height, width, bands]).astype(np.float32)
        self.labels = labels

    def get_Q_and_S_and_Segments(self):
        """
        执行 SLIC 并得到：
        - Q: 这里返回 None，不再构造 dense Q，避免大图像 OOM。
        - S: [num_superpixels, bands]，每个超像素的均值特征。
        - segments: [H, W]，每个像素所属超像素编号。

        原代码会构造 Q: [H*W, num_superpixels] 的 dense one-hot 矩阵。
        Houston 这类大图会导致几十 GB 显存/内存占用，因此改为 segments 稀疏索引。
        """
        img = self.data
        h, w, d = img.shape

        segments = slic(
            img,
            n_segments=self.n_segments,
            compactness=self.compactness,
            max_num_iter=self.max_iter,
            convert2lab=False,
            sigma=self.sigma,
            enforce_connectivity=True,
            min_size_factor=self.min_size_factor,
            max_size_factor=self.max_size_factor,
            slic_zero=False
        )

        if segments.max() + 1 != len(set(np.reshape(segments, [-1]).tolist())):
            segments = SegmentsLabelProcess(segments)

        self.segments = segments.astype(np.int64)
        superpixel_count = int(self.segments.max() + 1)
        self.superpixel_count = superpixel_count
        print("superpixel_count", superpixel_count)

        # 可视化变量，不参与训练；通道数不足 3 时跳过。
        if img.shape[2] >= 3:
            _ = mark_boundaries(img[:, :, [0, 1, 2]], self.segments)

        segments_flat = np.reshape(self.segments, [-1]).astype(np.int64)
        x = np.reshape(img, [-1, d]).astype(np.float32)

        S = np.zeros([superpixel_count, d], dtype=np.float32)
        counts = np.bincount(segments_flat, minlength=superpixel_count).astype(np.float32)
        counts[counts == 0] = 1.0

        # 比逐像素构造 Q 更省内存。
        for band in range(d):
            S[:, band] = np.bincount(segments_flat, weights=x[:, band], minlength=superpixel_count) / counts

        self.S = S
        return None, S, self.segments

    def get_A(self, sigma: float):
        """根据 segments 构建超像素邻接矩阵和 edge_index。"""
        Edge_index = []
        Edge_atter = []
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        h, w = self.segments.shape

        for i in range(h - 1):
            for j in range(w - 1):
                sub = self.segments[i:i + 2, j:j + 2]
                labels = np.unique(sub)
                if labels.size <= 1:
                    continue

                # 2x2 小块内所有不同超像素两两连接。
                for a_i in range(labels.size):
                    for b_i in range(a_i + 1, labels.size):
                        idx1 = int(labels[a_i])
                        idx2 = int(labels[b_i])
                        if idx1 == idx2 or A[idx1, idx2] != 0:
                            continue

                        pix1 = self.S[idx1]
                        pix2 = self.S[idx2]
                        diss = float(np.exp(-np.sum(np.square(pix1 - pix2)) / (sigma ** 2)))
                        A[idx1, idx2] = A[idx2, idx1] = diss

                        Edge_index.append([idx1, idx2])
                        Edge_index.append([idx2, idx1])
                        # GraphormerEncoder 中 edge_encoder 需要离散类型。
                        # 这里用 1 表示相邻边，避免原代码 float->int 后大量变成 0。
                        Edge_atter.append(1)
                        Edge_atter.append(1)

        if len(Edge_index) == 0:
            raise ValueError("No superpixel adjacency edges were generated. Please check SLIC segments.")

        Edge_index2 = np.array(Edge_index, dtype=np.int64).transpose(1, 0)
        Edge_atter2 = np.array(Edge_atter, dtype=np.int64)
        return A, Edge_index2, Edge_atter2


class LDA_SLIC(object):
    def __init__(self, data, labels, n_component):
        self.data = data.astype(np.float32)
        self.init_labels = labels.astype(np.int64)
        self.curr_data = self.data
        self.n_component = n_component
        self.height, self.width, self.bands = data.shape
        self.x_flatt = np.reshape(self.data, [self.width * self.height, self.bands])
        self.y_flatt = np.reshape(labels, [self.height * self.width])
        self.labes = labels

    def LDA_Process(self, curr_labels):
        curr_labels = np.reshape(curr_labels, [-1]).astype(np.int64)
        idx = np.where(curr_labels > 0)[0]
        if idx.size == 0:
            raise ValueError("LDA_Process received no positive training labels.")

        x = self.x_flatt[idx]
        y = curr_labels[idx]

        unique_y = np.unique(y)
        if unique_y.size < 2:
            print("[Warning] LDA requires at least 2 classes. Skip LDA and use first bands for SLIC.")
            return self.data[:, :, :min(3, self.bands)]

        lda = LinearDiscriminantAnalysis()
        lda.fit(x, y - 1)
        X_new = lda.transform(self.x_flatt)
        return np.reshape(X_new, [self.height, self.width, -1]).astype(np.float32)

    def SLIC_Process(self, img, scale=25):
        n_segments_init = max(2, int((self.height * self.width) / scale))
        print("n_segments_init", n_segments_init)

        myslic = SLIC(
            img,
            n_segments=n_segments_init,
            labels=self.labes,
            compactness=0.06,
            sigma=1,
            min_size_factor=0.1,
            max_size_factor=2
        )
        Q, S, Segments = myslic.get_Q_and_S_and_Segments()
        A, Edge_index, Edge_atter = myslic.get_A(sigma=10)
        return Q, S, A, Edge_index, Edge_atter, Segments

    def simple_superpixel(self, scale):
        curr_labels = self.init_labels
        X = self.LDA_Process(curr_labels)
        Q, S, A, Edge_index, Edge_atter, Seg = self.SLIC_Process(X, scale=scale)
        return Q, S, A, Edge_index, Edge_atter, Seg
