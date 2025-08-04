from typing import Literal
import torch
import torch.nn.functional as F
__all__ = ["HilbertTransformations"]


class HilbertTransformations(torch.nn.Module):
    def __init__(
        self,
        method: Literal["combined", "single_orthant", "basic", "directional"],
        mode: Literal["HT", "AS"],
        instantanious_frequency: bool = False,
    ):
        super().__init__()

        self.instantanious_frequency = instantanious_frequency

        if self.instantanious_frequency:
            mode = "AS"

        self.method = self._chooseMethod(method, mode)

    def _sign(self, X, n):
        # Because the intersection of both conditions is empty, we can use the
        # (+) pixel-wise with 1.0 or 0.0 values
        return ((0 < X) & (X <= n / 2 - 1)).float() + -(
            (n / 2 + 1 <= X) & (X <= n - 1)
        ).float()

    def _chooseMethod(self, method, mode):
        match method:
            case "combined":

                def maskHilbertTransform(X, Y, img, axis=0):
                    return -0.5j * (
                        (1.0 if axis == 0 else -1.0)
                        * self._sign(X, img.size(-2))
                        + self._sign(Y, img.size(-1))
                    )

                def maskAnalyticSignal(X, Y, img, axis=0):
                    return 1 + 0.5 * (
                        (1.0 if axis == 0 else -1.0)
                        * self._sign(X, img.size(-2))
                        + self._sign(Y, img.size(-1))
                    )

            case "single_orthant":

                def maskHilbertTransform(X, Y, img, axis=None):
                    return (
                        self._sign(X, img.size(-2))
                        + self._sign(Y, img.size(-1))
                        + self._sign(X, img.size(-2))
                        * self._sign(Y, img.size(-1))
                    )

                def maskAnalyticSignal(X, Y, img, axis=None):
                    return (1 + self._sign(X, img.size(-2))) * (
                        1 + self._sign(Y, img.size(-1))
                    )

            case "basic":

                def maskHilbertTransform(X, Y, img, axis=None):
                    return -self._sign(X, img.size(-2)) * self._sign(
                        Y, img.size(-1)
                    )

                def maskAnalyticSignal(X, Y, img, axis=None):
                    return 1 - 1j * self._sign(X, img.size(-2)) * self._sign(
                        Y, img.size(-1)
                    )

            case "directional":

                def maskHilbertTransform(X, Y, img, axis=0):
                    return -1j * (
                        self._sign(X, img.size(-2))
                        if axis == 0
                        else self._sign(Y, img.size(-1))
                    )

                def maskAnalyticSignal(X, Y, img, axis=0):
                    return 1 + (
                        self._sign(X, img.size(-2))
                        if axis == 0
                        else self._sign(Y, img.size(-1))
                    )

            case _:
                raise ValueError(f"Invalid method: {method}")

        return maskHilbertTransform if mode == "HT" else maskAnalyticSignal

    def forward(self, img, axis=0):
        X, Y = torch.meshgrid(
            torch.fft.fftfreq(img.size(-2)),
            torch.fft.fftfreq(img.size(-1)),
            indexing="ij",
        )
        if self.instantanious_frequency:
            freqs = self.method(X, Y, img, axis=axis) * \
                torch.fft.fft2(img, norm='forward')
            # Compute the phase of the analytic signal
            phase = torch.angle(freqs)
            # Compute the gradients of the phase
            # Gradient in x-direction (difference between neighboring pixels in
            # the horizontal axis)
            phase_dx = torch.zeros_like(phase)
            phase_dx[:, :, :, 1:] = phase[:, :, :, 1:] - phase[:, :, :, :-1]

            # Gradient in y-direction (difference between neighboring pixels in
            # the vertical axis)
            phase_dy = torch.zeros_like(phase)
            phase_dy[:, :, 1:, :] = phase[:, :, 1:, :] - phase[:, :, :-1, :]

            # Compute the instantaneous frequency
            ifreq_x = phase_dx.abs()
            ifreq_y = phase_dy.abs()
            ifreq = torch.sqrt(ifreq_x**2 + ifreq_y**2)

            return ifreq

        return torch.fft.ifft2(
            self.method(
                X,
                Y,
                img,
                axis=axis) *
            torch.fft.fft2(
                img,
                norm='forward'),
            norm='forward')
