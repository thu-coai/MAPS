# (ICLR'2025) MAPS: Advancing Multi-modal Reasoning in Expert-level Physical Science

This repository is the official implementation of paper: [MAPS: Advancing Multi-modal Reasoning in Expert-level Physical Science](https://arxiv.org/abs/2501.10768). 


## PPM Construction Phase: Data Synthesis

Requirements:
```
# install latex / ngspice
yes | sudo apt install texlive-full
sudo apt-get install -y libngspice0-dev ngspice

# python libraries
pip install PyMuPDF PySpice readchar httpx
```

Run the follwing script to generate the data for PPM training. 
```
bash ./ppm_construction/data_syn/scripts/run_gen.sh
```

## PPM Construction Phase: PPM Training

To run the script of fine-tuning Physics Perception Model (PPM), please refer the official repository of [CogVLM](https://github.com/THUDM/CogVLM). 

Process the data:

```
# get dataset
python ./ppm_construction/ft_vlm/data_process/get_datasets_from_json_data.py --note grid_v11_240831

# split the dataset
python ./ppm_construction/ft_vlm/data_process/split_dataset_circ.py --note grid_v11_240831
```

Train CogAgent-17B:
```
bash ./ppm_construction/ft_vlm/scripts/run_ft_cogagent_lora.sh
```

Inference:
```
bash ./ppm_construction/ft_vlm/scripts/eval_cogagent_lora.sh
```

## Inference Phase

To run the inference phase of MAPS, please refer to `./inference/scripts/run_maps.sh`.

```shell
pip install PyMuPDF PySpice readchar httpx

```

## Citation
```
@misc{zhu2025mapsadvancingmultimodalreasoning,
      title={MAPS: Advancing Multi-Modal Reasoning in Expert-Level Physical Science}, 
      author={Erle Zhu and Yadi Liu and Zhe Zhang and Xujun Li and Jin Zhou and Xinjie Yu and Minlie Huang and Hongning Wang},
      year={2025},
      eprint={2501.10768},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2501.10768}, 
}
```