# HoGA: Higher-Order Graph Attention via <br> Diversity-Aware k-Hop Sampling

This is the codebase for the main track WSDM'26 paper, _HoGA: Higher-Order Graph Attention via Diversity-Aware k-Hop Sampling_, which provides the implementation for the Higher-Order Graph Attention (HoGA) module.

HoGA extends existing single-hop GNN models to a k-hop setting by sampling the k-hop feature space with a diversity-driven walk. Our paper is available on arxiv: https://arxiv.org/abs/2411.12052.

## 🧠 Repository Overview

- `main.py`: Entry point. Parses config settings, initializes the models, and runs experiments.  
- `train.py`: Contains the training loop logic, checkpointing, logging, and evaluation hooks.  
- `multi_hop.py`: Implements the HoGA module. 
- `hop_utils.py`: Utility functions, which, for example, support various k-hop sampling methods
- `utils.py`: Miscellaneous helper functions (data loading, metrics, logging, etc.).  
- `config`: Directory that stores experiment, model, and dataset configuration files. 

## ⚙️ Running the Code

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TB862/Higher_Order.git
   cd Higher_Order

2. **Run an experiment on the Cora dataset:**
   ```bash
   python main.py --train --dataset Cora --model HoGA_GAT --gpu 0


## 📚 Consider Citing Our Work

If you find this repository or the HoGA module useful in your research, please consider citing our paper:

```bibtex
@inproceedings{bailie2026hoga,
  title     = {HoGA: Higher-Order Graph Attention via Diversity-Aware k-Hop Sampling},
  author    = {Bailie, Thomas and Koh, Yun Sing and Mukkavilli, Karthik},
  booktitle = {Proceedings of the 19th ACM International Conference on Web Search and Data Mining (WSDM '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA},
  url       = {https://arxiv.org/abs/2411.12052},
  note      = {arXiv:2411.12052 [cs.LG]}
}



