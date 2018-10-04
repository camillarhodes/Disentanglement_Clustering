# Disentanglement_Clustering
Disentanglement method for [Deep Continous Clustering]

Based on the arcicles [Deep Continuous Clustering](https://arxiv.org/abs/1803.01449)[code](https://github.com/shahsohil/DCC) and [Two-Step Disentanglement for Financial Data](https://arxiv.org/abs/1709.00199)

The commands used to run the project are:
```
$ python make_data.py

$ python pretraining.py --data mnist --tensorboard --id 1 --lr 1

$ python extract_feature.py --data mnist --net checkpoint_4.pth.tar --features pretrained

$ python edgeConstruction.py --dataset mnist/pretrained.pkl --samples 70000

$ python copyGraph.py --data mnist --graph pretrained.mat --features pretrained.pkl --out pretrained  

$ python DCC.py --data mnist --net-pretraining checkpoint_4.pth.tar --tensorboard --id 1 --step 1

$ python pretraining.py --data mnist --tensorboard --id 1 --lr 1 --fake --niter 0

$ python extract_feature.py --data mnist --net checkpoint_4.pth.tar.fake --features pretrained-fake

$ python edgeConstruction.py --dataset mnist/pretrained-fake.pkl --samples 70000

$ python copyGraph.py --data mnist --graph pretrained-fake.mat --features pretrained-fake.pkl --out pretrained-fake

$ python DCC.py --data mnist --net-pretraining checkpoint_4.pth.tar --net FTcheckpoint_500.pth.tar --tensorboard --id 1 --step 2 --M 5

```
where FTcheckpoint_500.pth.tar should be replaced with the last checkpoint of step 1

