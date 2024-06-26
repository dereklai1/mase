# Machine-Learning Accelerator System Exploration Tools

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Doc][doc-shield]][doc-url]

[contributors-shield]: https://img.shields.io/github/contributors/DeepWok/mase.svg?style=flat
[contributors-url]: https://github.com/DeepWok/mase/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/DeepWok/mase.svg?style=flat
[forks-url]: https://github.com/DeepWok/mase/network/members
[stars-shield]: https://img.shields.io/github/stars/DeepWok/mase.svg?style=flat
[stars-url]: https://github.com/DeepWok/mase/stargazers
[issues-shield]: https://img.shields.io/github/issues/DeepWok/mase.svg?style=flat
[issues-url]: https://github.com/DeepWok/mase/issues
[license-shield]: https://img.shields.io/github/license/DeepWok/mase.svg?style=flat
[license-url]: https://github.com/DeepWok/mase/blob/master/LICENSE.txt
[issues-shield]: https://img.shields.io/github/issues/DeepWok/mase.svg?style=flat
[issues-url]: https://github.com/DeepWok/mase/issues
[doc-shield]: https://readthedocs.org/projects/pytorch-geometric/badge/?version=latest
[doc-url]: https://deepwok.github.io/mase/

## Overview

Machine learning accelerators have been used extensively to compute models with high performance and low power. Unfortunately, the development pace of ML models is much faster than the accelerator design cycle, leading to frequent changes in the hardware architecture requirements, rendering many accelerators obsolete. Existing design tools and frameworks can provide quick accelerator prototyping, but only for a limited range of models that fit into a single hardware device. With the emergence of large language models such as GPT-3, there is an increased need for hardware prototyping of large models within a many-accelerator system to ensure the hardware can scale with ever-growing model sizes.

MASE provides an efficient and scalable approach for exploring accelerator systems to compute large ML models by directly mapping onto an efficient streaming accelerator system. Over a set of ML models, MASE can achieve better energy efficiency to GPUs when computing inference for recent transformer models.

![Alt text](./docs/imgs/overview.png)


## MASE Publications

* Fast Prototyping Next-Generation Accelerators for New ML Models using MASE: ML Accelerator System Exploration, [link](https://arxiv.org/abs/2307.15517)
  ```
  @article{cheng2023fast,
  title={Fast prototyping next-generation accelerators for new ml models using mase: Ml accelerator system exploration},
  author={Cheng, Jianyi and Zhang, Cheng and Yu, Zhewen and Montgomerie-Corcoran, Alex and Xiao, Can and Bouganis, Christos-Savvas and Zhao, Yiren},
  journal={arXiv preprint arXiv:2307.15517},
  year={2023}}
  ```
* MASE: An Efficient Representation for Software-Defined ML Hardware System Exploration, [link](https://openreview.net/forum?id=Z7v6mxNVdU)
  ```
  @article{zhangmase,
  title={MASE: An Efficient Representation for Software-Defined ML Hardware System Exploration},
  author={Zhang, Cheng and Cheng, Jianyi and Yu, Zhewen and Zhao, Yiren}}
  ```
### Repository structure

This repo contains the following directories:
* `components` - Internal hardware library
* `scripts` - Installation scripts  
* `machop` - MASE's software stack
* `hls` - HLS component of MASE
* `mlir-air` - MLIR AIR for ACAP devices
* `docs` - Documentation
* `Docker` - Docker container configurations

## MASE Dev Meetings

* Subscribe [Mase Weekly Dev Meeting (Wednesday 4:30 UK time)](https://calendar.google.com/calendar/event?action=TEMPLATE&tmeid=N2lpc25mN3VoamE5NXVmdmY5ZW1tOWpmMGdfMjAyMzExMDFUMTYzMDAwWiBqYzI0ODlAY2FtLmFjLnVr&tmsrc=jc2489%40cam.ac.uk&scp=ALL). Everyone is welcomed!
* Direct [Google Meet link](meet.google.com/fke-zvii-tgv)
* Join the [Mase Slack](https://join.slack.com/t/mase-tools/shared_invite/zt-2gl60pvur-pktLLLAsYEJTxvYFgffCog)
* If you want to discuss anything in future meetings, please add them as comments in the [meeting agenda](https://docs.google.com/document/d/12m96h7gOhhmikniXIu44FJ0sZ2mSxg9SqyX-Uu3s-tc/edit?usp=sharing) so we can review and add them.

## Donation  

If you think MASE is helpful, please [donate](https://www.buymeacoffee.com/mase_tools) for our work, we appreciate your support!

<img src='./docs/imgs/bmc_qr.png' width='250'>
