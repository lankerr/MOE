# Earthformer Training on Moving MNIST

Run the following command to train Earthformer on Moving MNIST dataset.
Change the configurations in [cfg.yaml](./cfg.yaml)

```bash
MASTER_ADDR=localhost MASTER_PORT=10001 python train_cuboid_mnist.py --gpus 2 --cfg cfg.yaml --ckpt_name last.ckpt --save tmp_mnist
```

Run the tensorboard command to upload experiment records

```bash
tensorboard dev upload --logdir ./experiments/tmp_mnist/lightning_logs --name 'tmp_mnist'
```

运行:

wsl -d Ubuntu-24.04 -u extrapolation bash -c "source ~/earthformer_env/bin/activate && cd /mnt/c/Users/97290/Desktop/datswinlstm_memory/earth-forecasting-transformer && python3 -u scripts/cuboid_transformer/moving_mnist/train_cuboid_mnist.py --gpus 1 --cfg scripts/cuboid_transformer/moving_mnist/cfg.yaml"
