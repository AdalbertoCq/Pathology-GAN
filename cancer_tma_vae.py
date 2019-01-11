
# Model and parameters.
from models.generative.vae_se_densenet_strict.vae import VAE
from models.generative.vae_se_densenet_strict.vae_se_densenet_strict2 import *


# Run training.
# with VAE(bottleneck_dim, enc_details, dec_details, kl_loss_factor, vars_job_id, restore, batch_size, epochs, lr) as cancer_tma_vae:
with VAE(bottleneck_dim, enc_details, dec_details, vars_job_id, restore, batch_size, epochs, lr) as cancer_tma_vae:
    cancer_tma_vae.run()
