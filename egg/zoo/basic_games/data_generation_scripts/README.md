This directory contains perl scripts to generate data in the formats expected by the basic reconstruction and discriminatio games, as well as example files generated with them. The headers of both scripts show usage and provide more information.

The example data were generated with the following commands.

Reconstruction game:

```bash
perl -w generate_reconstruction_dataset.pl 100 5 3 > example_reconstruction_input.txt
```

Discrimination game:

```bash
perl -w generate_discriminative_dataset.pl 100 5 3 2 > example_discriminative_input.txt
```
