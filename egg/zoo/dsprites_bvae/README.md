[beta-VAE](https://openreview.net/pdf?id=Sy2fzU9gl) as a communication game between Sender (Encoder) and Receiver (Decoder). 
The game can be used to play around with continuous channel and visual input. By default, topographic similarity and positional disentanglement are logged.
```bash
cd EGG
pip install --editable .;
python -m egg.zoo.dsprites_bvae.train --lr=1e-3 --batch_size=128 --n_epochs=100 --vocab_size=6
```
