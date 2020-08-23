In this small example, we cast VAE as a communication game between Sender (Encoder) and Receiver (Decoder).

This would be used as an example for the compositionality-as-disentanglement metrics.
You can run the game as 
```bash
python -m egg.zoo.mnist_vae.train --lr=1e-3 --batch_size=128 --n_epochs=100 --vocab_size=2
```
where `vocab_size` sets the dimensionality of the latent representation.

