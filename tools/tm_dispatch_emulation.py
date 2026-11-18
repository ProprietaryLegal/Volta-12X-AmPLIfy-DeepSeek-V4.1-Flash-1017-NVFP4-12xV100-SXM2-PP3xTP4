# Faithful-as-read re-implementation (float32 where the C++ uses float) of
# turbomind Context::Populate (context.cu:158-238) + Gemm::Find ranking (gemm.cu:770-870)
# for the layer-0 wkv GEMM: Q8_0 -> u8 affine, N=512, K=5120, group 32.
# Kernels: kernel/sm70_884_8.cu Config_U8_Affine tiles; chunk = max(CTA_K 32, group 32) = 32.
# SM70 tuning: max_splits 16, max_waves 32 (gemm.cu:538-539). sm_count 80 (V100 SXM2).
# max_active_ctas (cudaOccupancy) is unknown on a CPU: swept.
import sys, itertools
import numpy as np
f32 = np.float32
def cdiv(a, b): return -(-a // b)
import os
SM, N, K, CHUNK = 80, int(os.environ.get("TMN", "512")), 5120, 32
TILES = [128, 64, 32, 16, 8]
def rank(m, occ):
    specs = []
    for tx in TILES:
        a = occ[tx]
        tm, tn = cdiv(m, tx), cdiv(N, 128)
        chunks = cdiv(K, CHUNK)
        conc = SM * a
        wps = f32(tm * tn) / f32(conc)
        spw = f32(1) / wps
        ceil_m, ceil_n = tm * tx, tn * 128
        for s in range(1, min(16, chunks) + 1):
            sck = cdiv(chunks, s) * CHUNK
            smc = ceil_m * ceil_n * sck
            wmc = int(f32(smc) * spw)          # int64_t = int64 * float
            grid = tm * tn * s
            fw, res = grid // conc, grid % conc
            pw = f32(cdiv(res, SM)) / f32(a)
            waves = f32(fw) + pw
            if s > 1 and waves > 32: break
            mma = int(f32(wmc) * waves)
            mio = 2 * tn * m * sck * s + 1 * tm * N * sck * s + 2 * m * N * (s - 1) * 2
            specs.append((tx, s, mio, mma))
    mio_max = max(x[2] for x in specs); mma_max = max(x[3] for x in specs)
    avg = [f32(0.5 * (float(f32(x[2]) / f32(mio_max)) + float(f32(x[3]) / f32(mma_max)))) for x in specs]
    order = sorted(range(len(specs)), key=lambda i: avg[i])   # stable
    return [(specs[i][0], specs[i][1], float(avg[i])) for i in order[:3]]
Ms = [int(x) for x in sys.argv[1:]] or [2292, 2308, 6376, 6392, 3696, 3712, 4092, 8188]
seen = {}
for o128, osm in itertools.product([1, 2, 3, 4], [1, 2, 3, 4, 6, 8]):
    occ = {128: o128, 64: osm, 32: osm, 16: osm, 8: osm}
    key = tuple((m, rank(m, occ)[0][:2]) for m in Ms)
    seen.setdefault(key, []).append((o128, osm))
for key, occs in seen.items():
    print("occupancies", occs)
    for m, (tx, s) in key: print("   M=%5d -> CTA_M %3d, splits %2d" % (m, tx, s))
occ = {128: 2, 64: 3, 32: 3, 16: 3, 8: 3}
for m in Ms: print(m, rank(m, occ))
