In this example, we train a MNIST autoencoder with a discrete channel as a Sender/Receiver game.
The end-to-end training over the channel is done via Gumbel-Softmax relaxation.


The training can be started as:
```bash
python -m egg.zoo.mnist_autoenc.train --vocab_size=10 --n_epochs=50 --batch-size=16 --random_seed=7
```

Please refer to the [MNIST autoencoder tutorial](https://github.com/facebookresearch/EGG/blob/master/tutorials/EGG%20walkthrough%20with%20a%20MNIST%20autoencoder.ipynb) for more details.
