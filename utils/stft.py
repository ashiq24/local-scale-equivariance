from typing import Literal
import torch
import torch.nn.functional as F
from einops import rearrange


def stft2d(image, window_size, hop_size,
           window_fn=torch.hann_window, freq_lim=None, re_patch=True):
    r"""
    Compute the 2D Short-Time Fourier Transform (STFT) of a multi-channel image.

    Parameters:
    -----------
    image: torch.Tensor
        Input image as a 3D tensor (CxHxW) or a batch of images (BxCxHxW).
        C is the number of channels (e.g., 3 for RGB images).
    window_size: int
        Size of the window (both height and width).
    hop_size: int
        Hop size (stride) between successive windows.
    window_fn: function
        Window function to apply (default: Hann window).

    Returns:
    --------
    torch.Tensor
        STFT result with shape (B, C, num_patches_y, num_patches_x, window_size, window_size).
    """

    if len(image.shape) == 3:  # If single image, add batch dimension
        image = image.unsqueeze(0)

    batch_size, channels, height, width = image.shape

    # Create a 2D window using outer product of 1D window
    window_1d = window_fn(window_size, device=image.device)
    window_2d = window_1d[:, None] * window_1d[None, :]

    # Unfold the image into overlapping patches for the STFT
    patches = image.unfold(
        2, window_size, hop_size).unfold(
        3, window_size, hop_size)
    # Number of patches in y and x directions
    num_patches_y, num_patches_x = patches.shape[2], patches.shape[3]
    patches = patches.contiguous().view(
        batch_size, channels, -1, window_size, window_size)

    # Apply the window function to each patch
    windowed_patches = patches * window_2d

    # Compute 2D FFT on each windowed patch for each channel
    stft_result_whole = torch.fft.fft2(windowed_patches)
    if freq_lim is not None:
        stft_result = torch.zeros(
            batch_size,
            channels * num_patches_y * num_patches_x,
            2 * freq_lim,
            2 * freq_lim,
            device=image.device,
            dtype=stft_result_whole.dtype)
        stft_result[...,
                    :freq_lim,
                    :freq_lim] = stft_result_whole[...,
                                                   :freq_lim,
                                                   :freq_lim]
        stft_result[..., -
                    freq_lim:, :freq_lim] = stft_result_whole[..., -
                                                              freq_lim:, :freq_lim]
        stft_result[..., :freq_lim, -
                    freq_lim:] = stft_result_whole[..., :freq_lim, -freq_lim:]
        stft_result[..., -
                    freq_lim:, -
                    freq_lim:] = stft_result_whole[..., -
                                                   freq_lim:, -
                                                   freq_lim:]
        window_size = 2 * freq_lim
    else:
        stft_result = stft_result_whole[:, :, :, :, :]

    stft_result = torch.fft.fftshift(stft_result, dim=(-2, -1))

    out = stft_result.view(
        batch_size,
        channels,
        num_patches_y,
        num_patches_x,
        window_size,
        window_size)
    # print(out.shape)
    if re_patch:
        out = rearrange(out, 'b c h w i j -> b c (h i) (w j)')
    else:
        out = rearrange(out, 'b c h w i j -> b c (h w) i j')

    return out


def gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply Gaussian blur to the image.

    Parameters:
    -----------
    image : torch.Tensor
        Input image tensor (BxCxHxW).
    kernel_size : int
        Size of the Gaussian kernel.
    sigma : float
        Standard deviation of the Gaussian.

    Returns:
    --------
    torch.Tensor
        Blurred image tensor.
    """
    channels = image.shape[1]

    # Create Gaussian kernel
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 +
                     1, device=image.device, dtype=image.dtype)
    x = torch.exp(-0.5 * (x / sigma) ** 2)
    gaussian_kernel_1d = x / x.sum()

    # Create 2D Gaussian kernel by outer product
    gaussian_kernel_2d = gaussian_kernel_1d[:,
                                            None] * gaussian_kernel_1d[None, :]
    gaussian_kernel_2d = gaussian_kernel_2d.expand(
        channels, 1, kernel_size, kernel_size)

    # Apply the Gaussian kernel to the image
    smoothed_image = F.conv2d(
        image,
        gaussian_kernel_2d,
        padding=kernel_size // 2,
        groups=channels)

    return smoothed_image


def image_gradient(image):
    r"""
    Compute the gradient of a multi-channel image using Sobel filters.

    Parameters:
    -----------
    image: torch.Tensor
        Input image as a 3D tensor (CxHxW) or a batch of images (BxCxHxW).
        C is the number of channels (e.g., 3 for RGB images).

    Returns:
    --------
    torch.Tensor
        Gradient of the image with shape (B, C, H, W) representing the magnitude of gradients.
    """

    # Ensure batch dimension is present
    if len(image.shape) == 3:  # If single image, add batch dimension
        image = image.unsqueeze(0)

    # Optionally smooth the image using average pooling
    # image = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)

    # Define Sobel filters
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=image.device, dtype=image.dtype)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=image.device, dtype=image.dtype)

    # Apply Sobel filters across all channels
    grad_x = F.conv2d(
        image, sobel_x[None, None, :, :], padding=1, groups=image.shape[1])
    grad_y = F.conv2d(
        image, sobel_y[None, None, :, :], padding=1, groups=image.shape[1])

    # Compute gradient magnitude and stabilize by adding epsilon
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

    # Clamp the gradients to avoid exploding values
    grad_magnitude = torch.clamp(grad_magnitude, min=0.0, max=5.0)

    return grad_magnitude


def image_laplacian(image):
    r"""
    Compute the Laplacian of a multi-channel image.

    Parameters:
    -----------
    image: torch.Tensor
        Input image as a 3D tensor (CxHxW) or a batch of images (BxCxHxW).
        C is the number of channels (e.g., 3 for RGB images).

    Returns:
    --------
    torch.Tensor
        Laplacian of the image with shape (B, C, H, W).
    """

    if len(image.shape) == 3:  # If single image, add batch dimension
        image = image.unsqueeze(0)

    # apply smoothing on the image
    image = gaussian_blur(image, kernel_size=3, sigma=1.0)

    laplacian = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=image.device, dtype=image.dtype)

    # Apply the Laplacian filter to each channel of the image
    lap = F.conv2d(image, laplacian[None, None, :, :],
                   padding=1, groups=image.shape[1])

    # magnitude of the laplacian
    lap = torch.abs(lap + 1e-6)
    return lap


def finite_difference_filter(image):
    r"""
    Compute the gradients of a multi-channel image using finite difference filters
    and return the magnitude of the gradient.

    Parameters:
    -----------
    image: torch.Tensor
        Input image as a 3D tensor (C, H, W) or a batch of images (B, C, H, W).
        C is the number of channels (e.g., 3 for RGB images).

    Returns:
    --------
    grad_x: torch.Tensor
        Horizontal gradient with shape (B, C, H, W).
    grad_y: torch.Tensor
        Vertical gradient with shape (B, C, H, W).
    magnitude: torch.Tensor
        Magnitude of the gradient with shape (B, C, H, W).
    """

    # Ensure batch dimension is present
    if len(image.shape) == 3:  # If single image, add batch dimension
        image = image.unsqueeze(0)

    # Check for NaN values in the input image
    if torch.isnan(image).any():
        raise ValueError("Input image contains NaN values.")

    # Define finite difference kernels
    kernel_x = torch.tensor([[1, -1]], device=image.device,
                            dtype=image.dtype)  # Horizontal kernel
    kernel_y = torch.tensor(
        [[1], [-1]], device=image.device, dtype=image.dtype)  # Vertical kernel

    # Apply convolution to compute gradients
    grad_x = F.conv2d(image, kernel_x[None, None, :, :], padding=(
        0, 1), groups=image.shape[1])  # Convolve along width
    grad_y = F.conv2d(image, kernel_y[None, None, :, :], padding=(
        1, 0), groups=image.shape[1])  # Convolve along height

    min_height = min(grad_x.shape[2], grad_y.shape[2])
    min_width = min(grad_x.shape[3], grad_y.shape[3])
    grad_x = grad_x[:, :, :min_height, :min_width]
    grad_y = grad_y[:, :, :min_height, :min_width]


    # Compute the magnitude of the gradient
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    return magnitude
