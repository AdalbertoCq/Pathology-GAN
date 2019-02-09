# DCGAN
model_param['DCGAN'] = {epochs : 10, batch_size : 32, z_dim : 100, learning_rate_g : 1e-4, learning_rate_d : 1e-4, beta1 : 0.5, alpha : 0.2, use_bn : True, restore : False}

# LSGAN
model_param['LSGAN'] = {epochs : 10, batch_size : 32, z_dim : 100, learning_rate_g : 1e-4, learning_rate_d : 1e-4, beta1 : 0.5, alpha : 0.2, use_bn : True, restore : False}

# WGAN
model_param['WGAN'] = {epochs : 10, batch_size : 64, z_dim : 100, learning_rate_g : 1e-4, learning_rate_d : 1e-4, beta_1 : 0.5, beta_2 : 0.9, alpha : 0.2, use_bn : True, restore : False, clipping : 0.01, n_critic : 5}

# WGAN-GP
model_param['WGAN_GP'] = {epochs : 10, batch_size : 64, z_dim : 100, learning_rate_g : 1e-4, learning_rate_d : 1e-4, beta_1 : 0.5, beta_2 : 0.9, alpha : 0.2, use_bn : False, restore : False, n_critic : 5, gp_coeff : .5}

# RaSGAN
model_param['RaSGAN'] = {epochs : 10, batch_size : 64, z_dim : 100, learning_rate_g : 5e-5, learning_rate_d : 5e-5, beta1 : 0.5, alpha : 0.2, use_bn : True, restore : False}

# RaLSGAN
model_param['RaLSGAN'] = {epochs : 10, batch_size : 64, z_dim : 100, learning_rate_g : 5e-5, learning_rate_d : 5e-5, beta_1 : 0.5, alpha : 0.2, use_bn : True, restore : False}

# RaSGAN-GP.
model_param['RaSGAN_GP'] = {epochs : 10, batch_size : 64, z_dim : 100, learning_rate_g : 1e-4, learning_rate_d : 1e-4, alpha : 0.2, beta_1 : 0.5, beta_2 : 0.9, n_critic : 5, gp_coeff : .5, use_bn : False}

# SNGAN.
model_param['SNGAN'] = {epochs : 10, batch_size : 64, z_dim : 100, learning_rate : 1e-4, alpha : 0.2, beta_1 : 0.5, beta_2 : 0.9, n_critic : 5, gp_coeff : .5, use_bn : False}