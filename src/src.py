#!/usr/bin/env python3
from cp_hw5 import integrate_poisson, integrate_frankot, load_sources
from skimage import io
from skimage.color import rgb2xyz
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import os


def initials():
    print("Initials")
    images = []
    # tiffs = [f for f in os.listdir("../data/shoev2") if f.endswith('.tiff')]
    tiffs = [f for f in os.listdir("../data/bowl") if f.endswith('.tiff')]

    for i in range(1, 8):
        # img = io.imread(f"../data/input_{i}.tif").astype(np.float64)
        # img = io.imread(f"../data/shoev2/{tiffs[i-1]}")
        img = io.imread(f"../data/bowl/{tiffs[i-1]}")
        # img = io.imread(f"../data/shoe/{tiffs[i-1]}")
        # plt.imshow(img)
        # plt.show()
        #shoe
        # img = img[1850:3500, 1900:4600][::6, ::6]
        #shoev2
        # img = img[1750:2750, 2050:3550][::6, ::6]
        #bowl
        img = img[1600:2600, 2300:3800][::6, ::6]
        # plt.imshow(img)
        # plt.show()
        img_xyz = rgb2xyz(img)
        luminance = img_xyz[:, :, 1]  # Extract the Y channel
        images.append(luminance.flatten())
    H, W, C = img.shape
    I = np.stack(images).astype(np.float64)
    I = np.clip(I, 0, None)
    return I, H, W, C


def extract(Be, H, W, C):
    Ae = np.linalg.norm(Be, axis=0)
    Ne = Be / Ae
    Ne = Ne.T
    Ne = Ne.reshape((H, W, C))
    Ae = Ae.reshape((H, W))
    return Ae, Ne


def uncalibrated_photometric_stereo():
    I, H, W, C = initials()
    # I = (7, 159039)
    print("Uncalibrated Photometric Stereo")
    U, S, Vt = np.linalg.svd(I, full_matrices=False)
    S3 = np.diag(S[:3])
    U3 = U[:, :3]
    Le = U3 @ np.sqrt(S3)
    Be = np.sqrt(S3) @ Vt[:3, :]
    Ae, Ne = extract(Be, H, W, C)
    # visualize_albedo_normal_maps(Ae, Ne)
    ##################
    Q = np.array([[2, 0, 1], [1, 3, 1], [0, 1, 2]])
    BQ = np.linalg.inv(Q.T) @ Be
    AQ = np.linalg.norm(BQ, axis=0)
    NQ = BQ / AQ
    NQ = np.moveaxis(NQ, 0, -1)
    NQ = NQ.reshape((H, W, 3))
    AQ = AQ.reshape((H, W))
    # visualize_albedo_normal_maps(AQ, NQ)
    ##########
    return Le, Be, Ae, Ne, H, W, C

def visualize_albedo_normal_maps(Ae, Ne):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    cax1 = ax1.imshow(Ae, cmap="gray")
    ax1.set_title("Albedo Map")
    fig.colorbar(cax1, ax=ax1)

    ax2.imshow(normalize_normals(Ne))
    ax2.set_title("Normal Map")

    plt.show()


def enforcing_integrability():
    G = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
    Le, Be, Ae, Ne, H, W, C = uncalibrated_photometric_stereo()
    # B 3xP
    # Ne ...,3
    print("Enforcing Integrability")
    Be = np.linalg.inv(G.T) @ Be
    Be = Be.T
    Be = Be.reshape((H, W, C))
    # sigma = 10
    # sigma = 2
    #shoev2
    # sigma = 11
    # #bowl
    sigma = 2.0
    Be_blurred = np.zeros_like(Be)
    for i in range(3):
        Be_blurred[:, :, i] = gaussian_filter(Be[:, :, i], sigma=sigma)

    be_dy, be_dx = np.gradient(Be_blurred, axis=(0, 1))  # dy (rows), dx (columns)

    # Construct matrix A
    A = np.zeros((H * W, 6))  # Each pixel contributes one row to A
    index = 0
    for y in range(H):
        for x in range(W):
            b = Be[y, x]
            db_dx = be_dx[y, x]
            db_dy = be_dy[y, x]

            A1 = b[0] * db_dx[1] - b[1] * db_dx[0]
            A2 = b[0] * db_dx[2] - b[2] * db_dx[0]
            A3 = b[1] * db_dx[2] - b[2] * db_dx[1]
            A4 = -b[0] * db_dy[1] + b[1] * db_dy[0]
            A5 = -b[0] * db_dy[2] + b[2] * db_dy[0]
            A6 = -b[1] * db_dy[2] + b[2] * db_dy[1]

            A[index] = [A1, A2, A3, A4, A5, A6]
            index += 1
    # Solve Ax = 0 using SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    x = Vt[-1]  # The solution is the last row of Vt
    delta = np.array([[-x[2], x[5], 1], [x[1], -x[4], 0], [-x[0], x[3], 0]])
    Be_delt = Be @ delta
    plotting_Be(Be_delt, H, W, C)




def plotting_Be(Be, H, W, C):
    Be = np.moveaxis(Be, -1, 0)
    Be = Be.reshape((C, H * W))
    Ae, Ne = extract(Be, H, W, C)
    Z = normal_integration(np.moveaxis(Ne, -1, 0))

    visualize_albedo_normal_maps(Ae, Ne)

    plot3d(Z)
    plt.imshow(normalize_image(Z), cmap="gray")
    plt.title("Depth Map")
    plt.show()


def normal_integration(N):
    e = 1e-8
    N[0, :, :] /= N[2, :, :] + e
    N[1, :, :] /= N[2, :, :] + e
    grad = np.gradient(N, axis=0)
    df_dx = grad[0]
    df_dy = grad[1]
    Z = integrate_frankot(df_dx, df_dy)
    # Z = integrate_poisson(df_dx, df_dy)
    return Z


def calibrated_photometric_stereo():
    G = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    assert np.linalg.matrix_rank(G) == 3
    L = load_sources()
    I, H, W, C = initials()
    # I = (7, 159039)
    B = np.linalg.lstsq(L, I)[0]
    # B = np.linalg.inv(G.T) @ B
    A = np.linalg.norm(B, axis=0)
    N = B / A
    print("I: ", I.shape)
    print("B: ", B.shape)
    print("A: ", A.shape)
    print("N: ", N.shape)
    N = N.reshape((C, H, W))
    Z = normal_integration(N)
    A = A.reshape((H, W))
    plot3d(Z)
    plt.imshow(normalize_image(Z), cmap="gray")
    plt.title("Z")
    plt.show()
    visualize_albedo_normal_maps(normalize_image(A), np.moveaxis(N, 0, -1))


def normalize_image(I):
    return (I - I.min()) / (I.max() - I.min())


def normalize_normals(N):
    return (N + 1) / 2


# For surface depths
def plot3d(Z):
    # Z is an HxW array of surface depths
    H, W = Z.shape
    x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
    # set 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # add a light and shade to the axis for visual effect
    # (use the ‘-’ sign since our Z-axis points down)
    ls = LightSource()
    color_shade = ls.shade(-Z, plt.get_cmap("gray"))
    # display a surface
    # (control surface resolution using rstride and cstride)
    surf = ax.plot_surface(x, y, -Z, facecolors=color_shade, rstride=1, cstride=1)
    # turn off axis
    plt.axis("off")
    plt.show()


# uncalibrated_photometric_stereo()
enforcing_integrability()
# calibrated_photometric_stereo()
