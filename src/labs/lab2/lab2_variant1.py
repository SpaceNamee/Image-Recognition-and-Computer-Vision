
import os
import sys
import json
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # рендер без вікна (для збереження рисунків)
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# ---------------------------------------------------------------------------
# Параметри варіанту 1
# ---------------------------------------------------------------------------
NOISE_PROB = 0.05          # щільність імпульсного шуму 5 %
KSIZES = (3, 5)            # апертури фільтрів
T_LOW, T_HIGH = 30, 100    # пороги детектора Canny
SEED = 42                  # фіксований seed для відтворюваності результатів
OUT = "results"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "figure.dpi": 110})


# ---------------------------------------------------------------------------
# 1. Завантаження вхідного зображення технічної деталі
# ---------------------------------------------------------------------------
def load_image(path):
    """Завантажує зображення у градаціях сірого.
    Якщо файл не знайдено або його не вдалося прочитати — повідомляє користувача і завершує роботу."""
    if not os.path.exists(path):
        sys.exit(f"Помилка: файл зображення '{path}' не знайдено. "
                 f"Покладіть зображення поруч зі скриптом або вкажіть шлях: "
                 f"python lab2_variant1.py <шлях_до_зображення>")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"Помилка: не вдалося прочитати '{path}' як зображення "
                 f"(підтримуються формати JPG, PNG, BMP тощо).")
    print(f"Завантажено зображення: {path}  {img.shape}")
    return img


# ---------------------------------------------------------------------------
# 2. Моделювання імпульсного шуму «сіль та перець»
# ---------------------------------------------------------------------------
def add_salt_pepper(img, prob, seed=SEED):
    """prob/2 пікселів -> 0 (перець), prob/2 пікселів -> 255 (сіль)."""
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    rnd = rng.random(img.shape)
    noisy[rnd < prob / 2] = 0
    noisy[rnd > 1 - prob / 2] = 255
    return noisy


# ---------------------------------------------------------------------------
# 3. Метрики якості
# ---------------------------------------------------------------------------
def calc_metrics(orig, proc):
    """Повертає MSE, PSNR (dB) та SSIM між еталоном і обробленим зображенням."""
    diff = orig.astype(np.float64) - proc.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    return mse, psnr, float(ssim(orig, proc, data_range=255))


def timed(fn, *args, repeats=20):
    """Середній час виконання фільтра, мс."""
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(repeats):
        res = fn(*args)
    return res, (time.perf_counter() - t0) / repeats * 1000


# ---------------------------------------------------------------------------
# ОСНОВНИЙ ЕКСПЕРИМЕНТ
# ---------------------------------------------------------------------------
img = load_image(sys.argv[1] if len(sys.argv) > 1 else "detail.png")
cv2.imwrite(f"{OUT}/00_original.png", img)

noisy = add_salt_pepper(img, NOISE_PROB)
cv2.imwrite(f"{OUT}/01_noisy_sp5.png", noisy)
n_imp = int(np.sum(noisy != img))
print(f"Змінено імпульсами пікселів: {n_imp} ({n_imp / img.size * 100:.2f} %)")

# --- фільтрація: Median та Box з однаковими апертурами ---
results = {"Без фільтра": noisy}
times = {}
for k in KSIZES:
    results[f"Median {k}×{k}"], times[f"Median {k}×{k}"] = timed(cv2.medianBlur, noisy, k)
    results[f"Box {k}×{k}"], times[f"Box {k}×{k}"] = timed(cv2.blur, noisy, (k, k))
for name, im in results.items():
    cv2.imwrite(f"{OUT}/f_{name.replace(' ', '_').replace('×', 'x')}.png", im)

# --- метрики ---
table = []
print(f"\n{'Метод':<14}{'MSE':>10}{'PSNR, dB':>10}{'SSIM':>8}{'t, мс':>8}")
for name, im in results.items():
    mse, psnr, s = calc_metrics(img, im)
    row = dict(method=name, mse=round(mse, 2), psnr=round(psnr, 2), ssim=round(s, 4),
               time_ms=round(times.get(name, 0.0), 3))
    table.append(row)
    print(f"{name:<14}{mse:>10.2f}{psnr:>10.2f}{s:>8.4f}{row['time_ms']:>8.3f}")

# виграш медіани над Box
gain = {}
for k in KSIZES:
    m = next(r for r in table if r["method"] == f"Median {k}×{k}")
    b = next(r for r in table if r["method"] == f"Box {k}×{k}")
    gain[k] = dict(mse_ratio=round(b["mse"] / m["mse"], 2), psnr_gain=round(m["psnr"] - b["psnr"], 2))
    print(f"Апертура {k}×{k}: MSE_Box / MSE_Median = {gain[k]['mse_ratio']}, "
          f"виграш PSNR = {gain[k]['psnr_gain']} dB")

# --- Рис. 1: вхідні дані та результати фільтрації ---
fig, axes = plt.subplots(2, 3, figsize=(12, 9.2))
order = ["Без фільтра", "Median 3×3", "Median 5×5", None, "Box 3×3", "Box 5×5"]
for ax, name in zip(axes.ravel(), order):
    if name is None:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.set_title("Оригінал (еталон)")
    else:
        r = next(r for r in table if r["method"] == name)
        ttl = "Зашумлене S&P 5 %" if name == "Без фільтра" else name
        ax.imshow(results[name], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{ttl}\nMSE={r['mse']:.1f}  PSNR={r['psnr']:.2f} dB")
    ax.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_filtering.png", dpi=130); plt.close(fig)

# --- Рис. 2: збільшений фрагмент (видно залишки імпульсів / розмиття) ---
y0, x0, s = 60, 330, 110      # фрагмент з кріпильним отвором
fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
for ax, (name, im) in zip(axes, [("Оригінал", img), ("S&P 5 %", noisy),
                                  ("Median 3×3", results["Median 3×3"]),
                                  ("Median 5×5", results["Median 5×5"]),
                                  ("Box 5×5", results["Box 5×5"])]):
    ax.imshow(im[y0:y0 + s, x0:x0 + s], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(name); ax.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_zoom.png", dpi=130); plt.close(fig)

# --- Рис. 3: стовпчикова діаграма MSE (основне спец. дослідження) ---
fig, ax = plt.subplots(figsize=(7, 4))
xs = np.arange(len(KSIZES)); wbar = 0.36
mse_med = [next(r["mse"] for r in table if r["method"] == f"Median {k}×{k}") for k in KSIZES]
mse_box = [next(r["mse"] for r in table if r["method"] == f"Box {k}×{k}") for k in KSIZES]
b1 = ax.bar(xs - wbar / 2, mse_med, wbar, label="Median", color="#2a6fbb")
b2 = ax.bar(xs + wbar / 2, mse_box, wbar, label="Box", color="#e08a2c")
ax.bar_label(b1, fmt="%.1f"); ax.bar_label(b2, fmt="%.1f")
ax.set_xticks(xs, [f"{k}×{k}" for k in KSIZES]); ax.set_ylabel("MSE")
ax.set_title("MSE для Median та Box при однакових апертурах (S&P 5 %)")
ax.legend(); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_mse_bar.png", dpi=130); plt.close(fig)

# --- Рис. 4: 1D-профіль інтенсивності вздовж рядка через отвір ---
row_y = 256
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(noisy[row_y], color="#bbbbbb", lw=0.8, label="S&P 5 %")
ax.plot(img[row_y], color="black", lw=1.6, label="Оригінал")
ax.plot(results["Median 5×5"][row_y], color="#2a6fbb", lw=1.3, label="Median 5×5")
ax.plot(results["Box 5×5"][row_y], color="#e08a2c", lw=1.3, ls="--", label="Box 5×5")
ax.set_xlim(0, img.shape[1] - 1); ax.set_xlabel("x, піксель"); ax.set_ylabel("Інтенсивність")
ax.set_title(f"Профіль інтенсивності вздовж рядка y = {row_y}")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_profile.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Математична причина стійкості медіани: аналіз одного вікна 3×3 зі спайком
# ---------------------------------------------------------------------------
win = np.array([[150, 152, 149], [151, 255, 150], [148, 153, 151]], dtype=float)  # «сіль» у центрі
win_mean, win_med = win.mean(), np.median(win)
print(f"\nВікно 3×3 зі спайком 255: mean = {win_mean:.2f}, median = {win_med:.0f}")
# скільки імпульсів має потрапити у вікно, щоб медіана «зламалась»
# медіана 9 значень — 5-те за порядком; вона стає імпульсом, коли однакових імпульсів ≥ 5
breakdown = {k: (k * k) // 2 + 1 for k in (3, 5, 7)}
# ймовірність, що у вікні ≥ m імпульсів одного знаку (p/2 = 2.5 %) — біноміальний розподіл
from math import comb
def p_break(k, p=NOISE_PROB / 2):
    n = k * k; m = n // 2 + 1
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(m, n + 1))
p_fail = {k: p_break(k) for k in (3, 5)}
for k in (3, 5):
    print(f"P(медіана {k}×{k} дорівнює імпульсу) = {p_fail[k]:.2e}")

# Емпірична перевірка: частка імпульсів, що залишились після фільтра
imp_mask = noisy != img
residual_imp = {}
for name in ["Median 3×3", "Median 5×5", "Box 3×3", "Box 5×5"]:
    f = results[name].astype(int)
    # «залишковий» імпульс — піксель, відхилення якого від еталону > 40 рівнів
    residual_imp[name] = int(np.sum(np.abs(f - img.astype(int)) > 40))
print("Пікселів з похибкою > 40:", residual_imp)

# ---------------------------------------------------------------------------
# 5. Додатково: залежність MSE від щільності шуму (1…30 %) — межі ефективності
# ---------------------------------------------------------------------------
densities = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
sweep = {n: [] for n in ["Median 3×3", "Median 5×5", "Box 3×3", "Box 5×5"]}
for d in densities:
    nz = add_salt_pepper(img, d)
    for k in KSIZES:
        sweep[f"Median {k}×{k}"].append(calc_metrics(img, cv2.medianBlur(nz, k))[0])
        sweep[f"Box {k}×{k}"].append(calc_metrics(img, cv2.blur(nz, (k, k)))[0])
fig, ax = plt.subplots(figsize=(8, 4.2))
sty = {"Median 3×3": ("#7fb3e6", "-"), "Median 5×5": ("#2a6fbb", "-"),
       "Box 3×3": ("#f2b878", "--"), "Box 5×5": ("#e08a2c", "--")}
for n, v in sweep.items():
    ax.plot(np.array(densities) * 100, v, marker="o", ms=4, color=sty[n][0], ls=sty[n][1], label=n)
ax.axvline(5, color="gray", lw=0.8, ls=":"); ax.text(5.3, ax.get_ylim()[1] * 0.92, "варіант 1", color="gray")
ax.set_yscale("log"); ax.set_xlabel("Щільність шуму S&P, %"); ax.set_ylabel("MSE (лог. шкала)")
ax.set_title("Залежність MSE від щільності імпульсного шуму")
ax.legend(); ax.grid(alpha=0.3, which="both"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_density_sweep.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Виділення контурів: Sobel, Laplacian, Canny (30/100)
# ---------------------------------------------------------------------------
base = results["Median 5×5"]
sobelx = cv2.Sobel(base, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(base, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = cv2.magnitude(sobelx, sobely)
laplacian = cv2.Laplacian(base, cv2.CV_64F, ksize=3)

edges_ref = cv2.Canny(img, T_LOW, T_HIGH)          # «еталонна» карта контурів оригіналу
edge_stats = []
canny_maps = {}
for name in ["Без фільтра", "Median 3×3", "Median 5×5", "Box 3×3", "Box 5×5"]:
    e = cv2.Canny(results[name], T_LOW, T_HIGH)
    canny_maps[name] = e
    # точність / повнота з допуском 1 піксель (розширюємо еталон та результат)
    k3 = np.ones((3, 3), np.uint8)
    tp_p = np.sum((e > 0) & (cv2.dilate(edges_ref, k3) > 0))
    tp_r = np.sum((edges_ref > 0) & (cv2.dilate(e, k3) > 0))
    prec = tp_p / max(np.sum(e > 0), 1); rec = tp_r / max(np.sum(edges_ref > 0), 1)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    edge_stats.append(dict(method=name, n_edge=int(np.sum(e > 0)),
                           precision=round(float(prec), 3), recall=round(float(rec), 3),
                           f1=round(float(f1), 3)))
n_ref = int(np.sum(edges_ref > 0))
print(f"\nCanny {T_LOW}/{T_HIGH}: еталон (оригінал) N_edge = {n_ref}")
for r in edge_stats:
    print(f"  {r['method']:<12} N_edge={r['n_edge']:>6}  P={r['precision']:.3f}  "
          f"R={r['recall']:.3f}  F1={r['f1']:.3f}")

fig, axes = plt.subplots(2, 3, figsize=(12, 9.2))
axes[0, 0].imshow(base, cmap="gray"); axes[0, 0].set_title("Median 5×5 (вхід для контурів)")
axes[0, 1].imshow(sobel_mag, cmap="gray"); axes[0, 1].set_title("Модуль градієнта Собеля |G|")
axes[0, 2].imshow(np.abs(laplacian), cmap="gray", vmax=np.percentile(np.abs(laplacian), 99.5))
axes[0, 2].set_title("|Laplacian|")
axes[1, 0].imshow(canny_maps["Без фільтра"], cmap="gray")
axes[1, 0].set_title(f"Canny {T_LOW}/{T_HIGH} без фільтра\nN={edge_stats[0]['n_edge']}")
axes[1, 1].imshow(canny_maps["Median 5×5"], cmap="gray")
axes[1, 1].set_title(f"Canny {T_LOW}/{T_HIGH} після Median 5×5\nN={edge_stats[2]['n_edge']}")
axes[1, 2].imshow(canny_maps["Box 5×5"], cmap="gray")
axes[1, 2].set_title(f"Canny {T_LOW}/{T_HIGH} після Box 5×5\nN={edge_stats[4]['n_edge']}")
for ax in axes.ravel(): ax.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_edges.png", dpi=130); plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Вплив попереднього Gaussian Blur на Canny (пункт порядку виконання роботи)
# ---------------------------------------------------------------------------
blur_study = []
for sigma in [0, 0.8, 1.4, 2.0, 3.0]:
    src = results["Median 3×3"] if sigma == 0 else cv2.GaussianBlur(results["Median 3×3"], (0, 0), sigma)
    e = cv2.Canny(src, T_LOW, T_HIGH)
    k3 = np.ones((3, 3), np.uint8)
    prec = np.sum((e > 0) & (cv2.dilate(edges_ref, k3) > 0)) / max(np.sum(e > 0), 1)
    rec = np.sum((edges_ref > 0) & (cv2.dilate(e, k3) > 0)) / max(n_ref, 1)
    blur_study.append(dict(sigma=sigma, n_edge=int(np.sum(e > 0)), precision=round(float(prec), 3),
                           recall=round(float(rec), 3), f1=round(float(2 * prec * rec / (prec + rec + 1e-12)), 3)))
print("\nGaussian Blur перед Canny (на Median 3×3):")
for r in blur_study:
    print(f"  σ={r['sigma']:<4} N={r['n_edge']:>6} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

# ---------------------------------------------------------------------------
# Збереження числових результатів
# ---------------------------------------------------------------------------
summary = dict(image_shape=list(img.shape), impulses=n_imp, metrics=table, gain=gain,
               window=dict(values=win.astype(int).tolist(), mean=round(win_mean, 2), median=float(win_med)),
               breakdown=breakdown, p_fail={k: float(v) for k, v in p_fail.items()},
               residual_imp=residual_imp, sweep=dict(densities=densities,
               **{k: [round(x, 2) for x in v] for k, v in sweep.items()}),
               canny_ref=n_ref, edges=edge_stats, blur_study=blur_study)
with open(f"{OUT}/summary.json", "w", encoding="utf-8") as fjs:
    json.dump(summary, fjs, ensure_ascii=False, indent=2)
print(f"\nРезультати збережено у папці '{OUT}/'.")
