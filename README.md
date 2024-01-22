# CASF
This is the Repo for the paper: Better than Random: Reliable NLG Human Evaluation with Constrained Active Sampling

## Motivation
<div align=center>
<img src="https://github.com/EnablerRx/CASF/blob/main/Fig/motivation.png" width="350" height="315">
</div>

Random sampling is the most vital sampling method in natural language language gold-standarded human evaluation to save labor and costs, which is widely used in human evaluation sampling for its simplicity. However, random sampling can be risky. 

On the one hand, random sampling can lead to clustered selection, a phenomenon in which randomly selected samples are uncommonly close together in a population. 

On the other hand, random sampling may have the risk of data manipulation. Researchers can choose samples at will or conduct multiple random sampling to select a favorite subset, which will lead to unfair evaluation results. 

Since different sampling subsets may result in different inter-system rankings in human judgment, it is difficult to reliably select the best system. We urgently need a better sampling method to deliver reliable human evaluation with low labor and cost.

## Contributions
1) We investigate and experimentally analyze the sampling problem for the gold standard human evaluation in natural language generation.

2) We propose a Constrained Active Sampling Framework (CASF) for the sampling problem in manual evaluation. The proposed CASF can solve the problem of clustered selection and data manipulation for human evaluation sampling.

3) We re-evaluate 137 real NLG evaluation setups on 44 human evaluation metrics across 16 datasets and 5 NLG tasks. Experiment results demonstrate the proposed method ranks first or ranks second on 90.91% of the human metrics and receives 93.18% top-ranked system recognition accuracy. To ease the adoption of reliable sampling, we release a constrained active sampling tool. We strongly recommend using Constrained Active Sampling to sample test instances for human evaluation. 

## Direct Use
Run `CASF_tool.py` to select a sample subset for human evaluation. 



## Reproduce
Run `CASF.py` to reproduce the result in our paper.

# Citation
Please cite our work if you find it useful.
> To be released

