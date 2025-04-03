This code implements the experiments reported in the following paper:
* _Interpretable agent communication from scratch (with a generic visual processor emerging on the side)_, Roberto Dessi, Eugene Kharitonov, Marco Baroni, NeurIPS 2021, [[paper]](https://papers.nips.cc/paper/2021/hash/e250c59336b505ed411d455abaa30b4d-Abstract.html)

To reproduce the expriments from the paper using 16 GPUs the following command should be launched from the EGG root directory specifying a checkpoint directory, a slurm partition and the specific json:

```bash
$ python egg/nest/nest.py --game=egg.zoo.emcom_as_ssl.train --nodes=2 --tasks=8 --partition=<SPECIFY_SLURM_PARTITION> --sweep=egg/zoo/emcom_as_ssl/paper_sweeps/<ADD_JSON_FILE> --checkpoint_dir="<PATH_TO_CHECKPOINTING_DIR>" --checkpoint_freq=5
```
