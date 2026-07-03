# A generative process based on energy models incorporating implicit latent spaces

Official code repository for the paper titled "A generative process based on energy models incorporating implicit latent spaces".

We provide scripts for training, evaluating and installing all dependant libraries in "code/script".

Pytorch should automatically download the Cifar10 dataset once one of the scripts are run.
We also provided a checkpoint for our smaller model, together with it's configuration file in the directories "model" and "options" respectively.


The required libraries can be installed by running the install_requirements bash file:
```
bash code/scripts/install_requirements.sh
```

After activating the `newpt` conda environment we provide the fid evlaution script that can be run by:

```
bash code/scripts/evaluate_fid_parallel.py
```

Finally, ou can train your own energy model, on the CIFAR10 dataset, using our method by running:

```
bash code/scripts/train_cifar.py
```
