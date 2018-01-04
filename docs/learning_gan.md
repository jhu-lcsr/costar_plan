# Learning a GAN

Working on training pretrain_gan, here are some of the useful things we've found:

* (ganhacks)[https://github.com/soumith/ganhacks] has some good tips.
* We initially got stuck in a race to the bottom in terms of losses.
* Additionally, we've tried
    * Most helpful was using noisy labels: instead of 1 for fake and 0 for real, we use a uniform distribution of around 0.9-1 for fake and 0-0.1 for real.
    * A learning rate of 0.0001
    * No batchnorm on the discriminator
    * Momentum of 0.9
    * Mostly use lrelu
* Once on the right track, you should really be seeing a gradually decreasing discriminator loss.
* Command:
./scripts/ctp_model_tool --data_file test2.h5f --model pretrain_image_gan -e 500 --features multi --batch_size 64  --optimizer adam --lr 0.0001 --upsampling conv_transpose --use_noise true --noise_dim 0  --steps_per_epoch 100 --dropout_rate 0.1 --skip_connections 0 --decoder_dropout_rate 0.1 --hypothesis_dropout 1 -i 1000 --clipnorm 10
