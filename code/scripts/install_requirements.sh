# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
#FIRST JUST
conda create --name newpt

#THEN
conda activate newpt
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install matplotlib imageio einops scipy==1.11.1 opencv-python pytorch_lightning
python -m pip install torchmetrics[image]
conda install tqdm
# pip install scipy==1.11.1