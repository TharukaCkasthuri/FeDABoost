# FeDABoost:  Fairness Aware Federated Learning with Adaptive Boosting

This repository contains implementations of **FedAvg**, **Ditto**, and our proposed algorithm **FeDABoost**.  

Federated Learning (FL) makes it possible to train machine learning models across distributed clients (like mobile devices or edge servers) without sharing raw data. While promising privacy, FL often struggles in **non-IID settings** (when data across clients is imbalanced or very different). This can lead to poor global performance and unfair outcomes, where some clients benefit far more than others.  

**FeDABoost** is a new FL framework designed to tackle these challenges. It combines two key ideas:  

- **Adaptive gradient aggregation**  
  Inspired by AdaBoost’s weighting mechanism, FeDABoost gives more influence to reliable clients with lower local error rates, strengthening the global model.  

- **Dynamic boosting for underperforming clients**  
  By adjusting the focal loss focusing parameter during local training, FeDABoost highlights hard-to-classify examples, helping weaker clients improve.  

Together, these mechanisms improve fairness across clients while keeping performance competitive with standard FL baselines.  

We evaluate FeDABoost on **MNIST, FEMNIST, and CIFAR10**, and show that it improves fairness while achieving strong overall accuracy compared to **FedAvg** and **Ditto**.  

## Cite

If you use this repository or build upon our work, please cite:  
```
@inproceedings{fedaboost2025,
  author    = {Tharuka Kasthuri Arachchige and
               Veselka Boeva and
               Shahrooz Abghari},
  title     = {FeDABoost: Fairness Aware Federated Learning with Adaptive Boosting},
  booktitle = {Proceedings of the Workshop on Advances in Federated Learning (WAFL 2025) at ECML PKDD},
  year      = {2025},
  address   = {Porto, Portugal},
  note      = {To appear in Springer LNCS proceedings},
  url       = {https://wafl2025.di.unito.it}
}
```