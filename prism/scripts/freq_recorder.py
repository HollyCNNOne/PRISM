"""freq_recorder.py — 训练中记录 3 频带 z 轴频域占比（低/中/高）。"""
import json, glob, os
import numpy as np


def z_band_proportions(z_signal, n_bands=3):
    z = np.asarray(z_signal, np.float64)
    if z.ndim != 1 or z.shape[0] < 3:
        return np.ones(n_bands) / n_bands
    z = z - z.mean()
    spec = np.abs(np.fft.rfft(z)) ** 2
    spec[0] = 0.0
    n_freq = len(spec)
    if n_freq <= 1:
        return np.ones(n_bands) / n_bands
    lo, hi = np.log(1.0), np.log(n_freq)
    edges = np.exp(np.linspace(lo, hi, n_bands + 1))
    props = np.zeros(n_bands)
    for b in range(n_bands):
        m = (np.arange(n_freq) >= edges[b]) & (np.arange(n_freq) < edges[b + 1])
        props[b] = spec[m].sum()
    s = props.sum()
    return props / s if s > 1e-12 else np.ones(n_bands) / n_bands


def select_pillars(clean_vol, metal_mask, n=4, stride=16):
    Z, H, W = clean_vol.shape
    nonmetal = metal_mask.sum(axis=0) < 0.5
    ys, xs = np.nonzero(nonmetal)
    cand = [(int(y), int(x)) for y, x in zip(ys, xs)
            if y % stride == 0 and x % stride == 0]
    if len(cand) < n:
        cand = [(int(y), int(x)) for y, x in zip(ys, xs)]
    scores = [z_band_proportions(clean_vol[:, y, x])[-1] for y, x in cand]
    order = np.argsort(scores)
    mid = len(order) // 2
    idx = list(order[max(0, mid - n // 2): mid + n // 2 + (n % 2)])[:n]
    return [cand[i] for i in idx]


class FreqRecorder:
    def __init__(self, data_dir, patient="patients_head2", n_pillars=4,
                 out_csv="/root/autodl-tmp/PRISM/v2.2/fre_image/fre_epoch/freq_epoch.csv"):
        self.data_dir = data_dir
        self.out_csv = out_csv
        self.rows = []
        self._load_heldout(patient, n_pillars)

    def _find_split(self, sid, sub):
        for sp in ("train", "val"):
            p = os.path.join(self.data_dir, sp, sub, f"{sid}.raw")
            if os.path.exists(p):
                return p
        return None

    def _load_heldout(self, patient, n_pillars):
        meta_dir = os.path.join(self.data_dir, "meta")
        by_seq = {}
        for f in glob.glob(os.path.join(meta_dir, "*.json")):
            m = json.load(open(f))
            if m.get("patient_id") == patient:
                by_seq.setdefault(m["sequence_id"], []).append(m["sample_id"])
        if not by_seq:
            raise RuntimeError(f"held-out patient {patient} 无 meta")
        best = max(by_seq, key=lambda s: len(by_seq[s]))
        sids = sorted(by_seq[best], key=lambda s: int(s.rsplit("_", 1)[1]))
        Z = len(sids); H = W = 512
        clean = np.zeros((Z, H, W), np.float32)
        dirty = np.zeros((Z, H, W), np.float32)
        li = np.zeros((Z, H, W), np.float32)
        metal = np.zeros((Z, H, W), np.float32)
        for i, sid in enumerate(sids):
            for sub, arr in (("Target", clean), ("Baseline", dirty),
                             ("LI", li), ("Mask", metal)):
                p = self._find_split(sid, sub)
                if p:
                    arr[i] = np.fromfile(p, np.float32).reshape(H, W)
        self.clean = clean; self.dirty = dirty; self.li = li; self.metal = metal
        self.sids = sids; self.Z = Z
        self.pillars = select_pillars(clean, metal, n=n_pillars)
        self.gt_props = np.array([z_band_proportions(clean[:, y, x])
                                  for y, x in self.pillars])

    def record(self, out_vol, epoch):
        for j, (y, x) in enumerate(self.pillars):
            op = z_band_proportions(out_vol[:, y, x])
            for b in range(3):
                self.rows.append({
                    "epoch": epoch, "pillar": j,
                    "band": ["low", "mid", "high"][b],
                    "gt": float(self.gt_props[j, b]), "out": float(op[b]),
                    "err": float(abs(self.gt_props[j, b] - op[b])),
                })

    def save(self):
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
        with open(self.out_csv, "w") as f:
            f.write("epoch,pillar,band,gt,out,err\n")
            for r in self.rows:
                f.write(f"{r['epoch']},{r['pillar']},{r['band']},{r['gt']:.6f},{r['out']:.6f},{r['err']:.6f}\n")
        return self.out_csv
