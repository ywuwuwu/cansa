# CANSA Context‑Aware Network Slicing Auction

Edge & split inference: dynamic, multi‑branch models + context‑aware fusion.

Game‑theoretic allocation: two‑phase RB allocation aligned to urgency and truthfulness incentives.

Results: CANSA improves success rate under tight latency by large margins vs. Average/Proportional/Fairness; see the paper.


Clients estimate the minimal RBs to satisfy latency/accuracy under a chosen branch/fusion config and bid with urgency.

Edge runs Phase‑1 Priority (smallest R_min first, ties by higher bid), then Phase‑2 Proportional on residual RBs.

Clients adapt to the allocation by selecting the most energy‑efficient feasible model.

This repo contains a **minimal, runnable** release of the CANSA simulation and plotting code:
a dual‑scale control framework (auction‑based + client adaptation) for **context‑aware,
multi‑branch sensor‑fusion DNNs** over device/edge. It reproduces core analyses like
**success rate by method** and **Jain’s fairness** with a single command.

> Method overview: clients bid for RBs using context‑aware urgency; the edge orchestrator
> runs a two‑phase allocation (priority, then proportional); clients adapt post‑allocation to
> pick the most energy‑efficient feasible model. See Algorithm 1 and the system diagram for detail.

## What’s here

- `bid_final.py` — client/server simulation + two‑phase allocation + post‑allocation adaptation (runs end‑to‑end).
- `utils1.py` — latency/energy/accuracy utilities, action decoding, context maps, and a small channel sampler.
- `paper/` — PDF of the paper for context.

## 📚 Citation

If you use this code, data, or figures in your work, please cite our IEEE MASS 2025 paper:

> **Y. Wu, C. F. Chiasserini, and M. Levorato**,  
> “Distributed Context-Aware Resource Allocation for Dynamic Sensor Fusion in Edge Inference,”  
> *Proceedings of the IEEE 22nd International Conference on Mobile Ad Hoc and Smart Systems (MASS)*,  
> Chicago, USA, 2025.  

```bibtex
@inproceedings{wu2025cansa,
  author    = {Yashuo Wu and Carla Fabiana Chiasserini and Marco Levorato},
  title     = {Distributed Context-Aware Resource Allocation for Dynamic Sensor Fusion in Edge Inference},
  booktitle = {Proc. IEEE 22nd International Conference on Mobile Ad Hoc and Smart Systems (MASS)},
  year      = {2025},
  address   = {Chicago, USA},
  publisher = {IEEE},
  pages     = {1--9},
}




