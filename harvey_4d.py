# Single MLP: Linear(D→H) → ReLU → Dropout(optional) → Linear(H→2D), then split into s and t. Spec: D=1 here.
class STNet(nn.Module):
    """Scale/shift network used inside a 2D affine coupling layer.

    Given the fixed dimension (shape (B, 1)), this MLP outputs coupling parameters
    
    (s, t)
    (each shape (B, 1)) for transforming the other dimension.

    Notes:
    - The scale output is bounded with `tanh` for numerical stability.
    - Dropout is optional and disabled by default (dropout_p=0.0).
    """

    def __init__(self, hidden=64, dropout_p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout_p)) if float(dropout_p) > 0 else nn.Identity(),
            nn.Linear(hidden, 2)
        )

    def forward(self, h_fixed):
        """Compute coupling parameters.

        Args:
            h_fixed: Tensor of shape (B, 1).

        Returns:
            s: Tensor of shape (B, 1), bounded via `tanh`.
            t: Tensor of shape (B, 1).
        """
        st = self.net(h_fixed)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)
        return s, t


# Affine coupling: always fix dim 0 and transform dim 1.
class AffineCoupling2D(nn.Module):
    """RealNVP-style affine coupling transform for 2D vectors.

    Dimension 0 is kept fixed and dimension 1 is affine-transformed using parameters
    produced by `STNet`. The transform returns the output and the corresponding
    log-determinant of the Jacobian.
    """

    def __init__(self, hidden=64, dropout_p=0.0):
        super().__init__()
        self.st_net = STNet(hidden=hidden, dropout_p=dropout_p)

    def forward(self, z):
        """Forward map (base z -> data x).

        Args:
            z: Tensor of shape (B, 2).

        Returns:
            x: Tensor of shape (B, 2).
            log_det: Tensor of shape (B,), log|det J_f(z)|.
        """
        fixed, trans = z[:, :1], z[:, 1:]
        s, t = self.st_net(fixed)
        scale = torch.exp(s)
        out = torch.cat([fixed, trans * scale + t], dim=-1)
        log_det = s.squeeze(-1)
        return out, log_det

    def inverse(self, x):
        """Inverse map (data x -> base z).

        Args:
            x: Tensor of shape (B, 2).

        Returns:
            z: Tensor of shape (B, 2).
            log_det: Tensor of shape (B,), log|det J_{f^{-1}}(x)|.
        """
        fixed, trans = x[:, :1], x[:, 1:]
        s, t = self.st_net(fixed)
        scale = torch.exp(s)
        out = torch.cat([fixed, (trans - t) / scale], dim=-1)
        log_det = -s.squeeze(-1)
        return out, log_det
    

class Permute2D(nn.Module):
    """Swap the two dimensions. Log-determinant is strictly zero."""

    def forward(self, z):
        return z.flip(-1), torch.zeros(z.size(0), device=z.device, dtype=z.dtype)

    def inverse(self, x):
        return x.flip(-1), torch.zeros(x.size(0), device=x.device, dtype=x.dtype)


class Flow2D(nn.Module):
    """A small 2D normalising flow built by stacking coupling layers."""

    def __init__(self, n_layers, hidden, dropout_p=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(AffineCoupling2D(hidden=hidden, dropout_p=dropout_p))
            if i < n_layers - 1:
                self.layers.append(Permute2D())

    def f(self, z):
        log_det_total = 0.0
        x = z
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total = log_det_total + log_det
        return x, log_det_total

    def f_inv(self, x):
        log_det_total = 0.0
        z = x
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total = log_det_total + log_det
        return z, log_det_total

    def log_prob(self, x):
        """Compute log p(x) via the change-of-variables formula."""
        z, log_det_inv = self.f_inv(x)
        log_p_z = standard_normal_logprob(z)
        return log_p_z + log_det_inv
    

def standard_normal_logprob(z):
    """Log-density under a standard multivariate normal."""
    d = z.size(-1)
    return -0.5 * (d * torch.log(torch.tensor(2 * np.pi, device=z.device)) + (z ** 2).sum(dim=-1))

def make_train_loader(x_tensor, batch_size, seed, shuffle=True):
    """Create a reproducible training DataLoader for tensor-only inputs."""
    ds = TensorDataset(x_tensor)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle, generator=g)