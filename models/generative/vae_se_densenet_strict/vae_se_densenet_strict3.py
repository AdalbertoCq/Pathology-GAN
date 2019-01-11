import models.ops as ops

bottleneck_dim = 28
batch_size     = 25
epochs         = 1
lr             = 1e-3
vars_job_id    = 'vae_se_densenet_strict3'
restore        = False

# Parameters for Encoder.
enc_details = dict()
enc_details['enc_blocks']        = (2, 3, 2)
enc_details['enc_growth_rates']  = (16, 16, 16)
enc_details['enc_final_filters'] = 2
enc_details['enc_shape']         = [-1, bottleneck_dim * bottleneck_dim, 2]
enc_details['downsample']        = ops.conv_5_2
enc_details['activation']        = ops.lrelu

# Parameters for Generator.
dec_details = dict()
dec_details['gen_blocks']        = (2, 3, 2)
dec_details['gen_growth_rates']  = (16, 16, 16)
dec_details['gen_final_filters'] = 3
dec_details['gen_shape']         = [-1, bottleneck_dim, bottleneck_dim, 1]
dec_details['downsample']        = ops.conv_t_5


