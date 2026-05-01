import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("Image Segmentation using GA & PSO (Improved Auto-K)")

# -------------------------------
# MODE
# -------------------------------
mode = st.radio("Select Mode", ["Auto K (Elbow - KMeans)", "Manual K"])

if mode == "Manual K":
    k = st.slider("Number of Clusters (K)", 2, 8, 4)

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
uploaded_gt = st.file_uploader("Upload Ground Truth (optional)", type=["jpg", "png", "jpeg"])

# -------------------------------
# UTIL: IoU & Dice
# -------------------------------
def iou_score(gt, pred):
    gt = gt / 255
    pred = pred / 255
    inter = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    return np.sum(inter) / (np.sum(union) + 1e-6)

def dice_score(gt, pred):
    gt = gt / 255
    pred = pred / 255
    inter = np.sum(gt * pred)
    return (2. * inter) / (np.sum(gt) + np.sum(pred) + 1e-6)

# -------------------------------
# BETTER AUTO-K (KMeans ELBOW)
# -------------------------------
def find_best_k_kmeans(image_rgb, k_range=range(2, 9)):
    # downsample for speed/stability
    small = cv2.resize(image_rgb, (200, 200), interpolation=cv2.INTER_AREA)
    pixels = small.reshape((-1, 3)).astype(np.float32)

    errors = []
    for k_val in k_range:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
        # kmeans++ init gives better centers
        compactness, labels, centers = cv2.kmeans(
            pixels, k_val, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        # compactness is already SSE; use it directly
        errors.append(compactness)

    k_vals = list(k_range)

    # elbow via second derivative (curvature)
    g1 = np.gradient(errors)
    g2 = np.gradient(g1)
    best_idx = int(np.argmax(g2))
    best_k = k_vals[best_idx]

    return k_vals, errors, best_k

# -------------------------------
# MAIN
# -------------------------------
if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    st.image(image, caption="Original")

    # Auto-K
    if mode == "Auto K (Elbow - KMeans)":
        k_values, errors, k = find_best_k_kmeans(image)
        st.write("Auto-selected K:", int(k))

        fig_k, ax_k = plt.subplots()
        ax_k.plot(k_values, errors, marker='o')
        ax_k.set_title("Elbow (K vs SSE)")
        ax_k.set_xlabel("K")
        ax_k.set_ylabel("Error (SSE)")
        st.pyplot(fig_k)

    # Preprocess
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    pixels = blur.reshape((-1, 3)).astype(float)

    # -------------------------------
    # FITNESS
    # -------------------------------
    def fitness_function(centers):
        dists = np.linalg.norm(pixels[:, None] - centers, axis=2)
        labels = np.argmin(dists, axis=1)
        segmented = centers[labels]
        fitness = np.sum((pixels - segmented) ** 2)  # SSE
        return fitness, labels

    # -------------------------------
    # GA
    # -------------------------------
    def genetic_algorithm(k, pop_size=8, generations=10, mut_rate=0.1):
        population = [np.random.randint(0, 256, (k, 3)) for _ in range(pop_size)]
        history = []

        for _ in range(generations):
            fits = [fitness_function(ind)[0] for ind in population]
            history.append(min(fits))

            idx = np.argsort(fits)
            selected = [population[i] for i in idx[:pop_size // 2]]

            new_pop = selected.copy()
            while len(new_pop) < pop_size:
                p1 = selected[np.random.randint(len(selected))]
                p2 = selected[np.random.randint(len(selected))]
                point = np.random.randint(1, k)
                child = np.vstack((p1[:point], p2[point:]))

                if np.random.rand() < mut_rate:
                    i = np.random.randint(0, k)
                    child[i] = np.random.randint(0, 256, 3)

                new_pop.append(child)

            population = new_pop

        best = min(population, key=lambda ind: fitness_function(ind)[0])
        return best, history

    # -------------------------------
    # PSO
    # -------------------------------
    def pso(k, particles=8, iterations=10, w=0.5, c1=1.5, c2=1.5):
        pos = [np.random.randint(0, 256, (k, 3)).astype(float) for _ in range(particles)]
        vel = [np.zeros((k, 3)) for _ in range(particles)]

        pbest = pos.copy()
        pscores = [fitness_function(p)[0] for p in pos]
        gbest = pbest[int(np.argmin(pscores))]
        history = []

        for _ in range(iterations):
            history.append(min(pscores))
            for i in range(particles):
                r1, r2 = np.random.rand(), np.random.rand()
                vel[i] = (w * vel[i] +
                          c1 * r1 * (pbest[i] - pos[i]) +
                          c2 * r2 * (gbest - pos[i]))
                pos[i] += vel[i]
                pos[i] = np.clip(pos[i], 0, 255)

                score = fitness_function(pos[i])[0]
                if score < pscores[i]:
                    pbest[i] = pos[i]
                    pscores[i] = score

            gbest = pbest[int(np.argmin(pscores))]

        return gbest, history

    # -------------------------------
    # RUN
    # -------------------------------
    if st.button("Run Segmentation"):
        # GA
        t0 = time.time()
        ga_centers, ga_hist = genetic_algorithm(int(k))
        ga_time = time.time() - t0
        ga_fit, ga_labels = fitness_function(ga_centers)
        ga_img = ga_centers[ga_labels].reshape(image.shape).astype(np.uint8)
        ga_img = cv2.medianBlur(ga_img, 5)

        # PSO
        t0 = time.time()
        pso_centers, pso_hist = pso(int(k))
        pso_time = time.time() - t0
        pso_fit, pso_labels = fitness_function(pso_centers)
        pso_img = pso_centers[pso_labels].reshape(image.shape).astype(np.uint8)
        pso_img = cv2.medianBlur(pso_img, 5)

        c1, c2 = st.columns(2)
        c1.image(ga_img, caption=f"GA (time {ga_time:.2f}s)")
        c2.image(pso_img, caption=f"PSO (time {pso_time:.2f}s)")

        # Downloads
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(ga_img, cv2.COLOR_RGB2BGR))
        st.download_button("Download GA Image", buffer.tobytes(), "ga.png")

        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(pso_img, cv2.COLOR_RGB2BGR))
        st.download_button("Download PSO Image", buffer.tobytes(), "pso.png")

        # -------------------------------
        # ACCURACY (if GT)
        # -------------------------------
        if uploaded_gt is not None:
            gt_bytes = np.asarray(bytearray(uploaded_gt.read()), dtype=np.uint8)
            gt = cv2.imdecode(gt_bytes, 0)

            if len(gt.shape) == 3:
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

            gt = cv2.resize(gt, (image.shape[1], image.shape[0]))
            _, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)

            ga_gray = cv2.cvtColor(ga_img, cv2.COLOR_RGB2GRAY)
            _, ga_bin = cv2.threshold(ga_gray, 127, 255, cv2.THRESH_BINARY)

            pso_gray = cv2.cvtColor(pso_img, cv2.COLOR_RGB2GRAY)
            _, pso_bin = cv2.threshold(pso_gray, 127, 255, cv2.THRESH_BINARY)

            st.write("GA IoU:", iou_score(gt, ga_bin))
            st.write("GA Dice:", dice_score(gt, ga_bin))
            st.write("PSO IoU:", iou_score(gt, pso_bin))
            st.write("PSO Dice:", dice_score(gt, pso_bin))
        else:
            st.info("Upload ground truth to compute IoU/Dice.")

        # -------------------------------
        # CONVERGENCE
        # -------------------------------
        fig, ax = plt.subplots()
        ax.plot(ga_hist, label="GA")
        ax.plot(pso_hist, label="PSO")
        ax.set_title("Convergence (SSE)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness (lower is better)")
        ax.legend()
        st.pyplot(fig)

        # -------------------------------
        # DIFFERENCE MAP
        # -------------------------------
        diff = cv2.absdiff(ga_img, pso_img)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

        st.subheader("Difference Map (GA vs PSO)")
        d1, d2 = st.columns(2)
        d1.image(diff_gray, caption="Raw difference")
        d2.image(diff_thresh, caption="Highlighted differences")

        # Overlay on original
        overlay = image.copy()
        overlay[diff_thresh == 255] = [255, 0, 0]
        st.image(overlay, caption="Differences highlighted (red)")