class NormalisingFlow(nn.Module):
    """
    Normalising flow model built by stacking affine coupling layers.
    
    Parameters:
        dim (int): Input dimensionality.
        hidden (int): Hidden width for each coupling MLP.
        n_layers (int): Number of coupling layers.
    
    Returns:
        None.
    """
    def __init__(self, dim: int, hidden: int, n_layers: int) -> None:
        """
        Initialise a stacked affine-coupling normalising flow.
        
        Parameters:
            dim (int): Input dimensionality.
            hidden (int): Hidden width for each coupling MLP.
            n_layers (int): Number of coupling layers.
        
        Returns:
            None.
        """
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.n_layers = n_layers

        masks = build_alternating_masks(dim=dim, n_layers=n_layers)
        self.layers = nn.ModuleList([
            CouplingLayer(dim=dim, hidden=hidden, mask=mask)
            for mask in masks
        ])

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply all coupling layers in forward order (base to data space).
        
        Parameters:
            z (torch.Tensor): Input tensor with shape (batch_size, dim).
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                x (torch.Tensor): Forward-transformed outputs.
                total_log_det (torch.Tensor): Summed log-determinants with shape (batch_size,).
        """
        x = z
        total_log_det = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            x, log_det = layer(x)
            total_log_det = total_log_det + log_det
        return x, total_log_det

    def inverse(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply all coupling layers in reverse order (data to base space).
        
        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, dim).
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                z (torch.Tensor): Inverse-transformed outputs.
                total_inv_log_det (torch.Tensor): Summed inverse log-determinants with shape (batch_size,).
        """
        z = x
        total_inv_log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in reversed(self.layers):
            z, inv_log_det = layer.inverse(z)
            total_inv_log_det = total_inv_log_det + inv_log_det
        return z, total_inv_log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x) under the flow using a standard normal base density.
        
        Parameters:
            x (torch.Tensor): Data tensor with shape (batch_size, dim).
        
        Returns:
            torch.Tensor: Per-sample log-likelihood values with shape (batch_size,).
        """
        z, log_det = self.inverse(x)

        #standard normal log-probability for each sample.
        log_pz = -0.5 * (z.pow(2) + np.log(2.0 * np.pi)).sum(dim=1)
        log_px = log_pz + log_det
        return log_px

def count_flops(*, dim: int, n_layers: int, hidden: int, batch_size: int) -> int:
    """Estimate total FLOPs for one full inverse-and-log-prob pass over a batch.

    Parameters:
        dim (int): Input dimensionality (must be even for coupling splits).
        n_layers (int): Number of affine coupling layers.
        hidden (int): Hidden width of each coupling MLP.
        batch_size (int): Number of samples processed in the batch.

    Returns:
        int: Estimated total floating-point operations for the batch.
    """
    
    assert dim % 2 == 0, "dim must be even for affine coupling split" #only allows even number of dimenisons
    
    d_u = dim // 2  # transformed dims per coupling layer

    # MLP: Linear(D->H) -> ReLU -> Linear(H->2D)
    l1 = (2 * dim) * hidden + hidden          # mult+add + bias
    relu = hidden                              # 1 per hidden unit
    l2 = (2 * hidden) * (2 * dim) + (2 * dim) # mult+add + bias
    mlp_cost = l1 + relu + l2

    # Coupling inverse arithmetic on transformed dims only
    tanh_cost = 10 * d_u
    inverse_affine = (1 + 1 + 10 + 1) * d_u   # sub + neg + exp + mul
    logdet_sum = d_u + max(d_u - 1, 0)        # neg + adds
    per_layer = mlp_cost + tanh_cost + inverse_affine + logdet_sum

    # accumulate K log-dets + combine with base log-prob + base density
    logdet_accum = max(n_layers - 1, 0)
    combine_with_base = 1
    base_density = 20

    total_per_sample = n_layers * per_layer + logdet_accum + combine_with_base + base_density
    return int(batch_size * total_per_sample)