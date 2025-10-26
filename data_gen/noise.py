import numpy as np
import torch


def noise_torch(t, snr ,snr_max, kind='gaussian_blind', n_corr=None):
    if kind == 'gaussian':
        return gaussian_noise(t, snr)
    elif kind == 'gaussian_blind':
        return gaussian_blind_noise(t, snr,snr_max)
    elif kind == 'sparse':
        return sparse_noise(t, n_corr)
    elif kind == 'variable_sparse':
        return variable_sparse_noise(t, n_corr)


def gaussian_blind_noise(s, snr,snr_max):
    """
    Add Gaussian noise to the input signal. The std of the gaussian noise is uniformly chosen between 0 and 1/sqrt(snr).
    """
    s = torch.from_numpy(s)
    bsz, _, signal_dim = s.size()
    s = s.view(bsz, -1)
    low = snr
    high = snr_max
    scpu = s.cpu().numpy()
    snr_array = (low + (high - low) * torch.rand(bsz))[:, None]
    snr_array=snr_array.cpu().numpy()
    noise = torch.randn(s.size(), device=s.device, dtype=s.dtype)
    s = torch.from_numpy(scpu * (10 ** (snr_array / 20))).to(s.device)
    return (s + noise).view(bsz, -1, signal_dim).numpy()

def gaussian_noise(s, snr):
    """
    Add Gaussian noise to the input signal.
    """
    s = torch.from_numpy(s)
    bsz, _, signal_dim = s.size()
    s = s.view(s.size(0), -1)
    sigma = 1/(10 ** (snr / 20))
    noise = torch.randn(s.size(), device=s.device, dtype=s.dtype)
    # mult = sigma * torch.norm(s, 2, dim=1) / (torch.norm(noise, 2, dim=1))
    noise = noise * sigma
    return (s + noise).view(bsz, -1, signal_dim).numpy()


def sparse_noise(s, n_corr):
    """
    Add sparse noise to the input signal. The number of corrupted elements is equal to n_corr.
    """
    noisy_signal = s.clone()
    corruption = 0.5 * torch.randn((s.size(0), s.size(1), n_corr), device=s.device, dtype=s.dtype)
    for i in range(s.size(0)):
        idx = torch.multinomial(torch.ones(s.size(-1)), n_corr, replacement=False)
        noisy_signal[i, :, idx] += corruption[i, :]
    return noisy_signal


def variable_sparse_noise(s, max_corr):
    """
    Add sparse noise to the input signal. The number of corrupted elements is drawn uniformaly between 1 and
    max_corruption.
    """
    noisy_signal = s.clone()
    corruption = 0.5 * torch.randn((s.size(0), s.size(1), max_corr), device=s.device, dtype=s.dtype)
    n_corr = np.random.randint(1, max_corr + 1, (s.size(0)))
    for i in range(s.size(0)):
        idx = torch.multinomial(torch.ones(s.size(-1)), int(n_corr[i]), replacement=False)
        noisy_signal[i, :, idx] += corruption[i, :, :n_corr[i]]
    return noisy_signal
