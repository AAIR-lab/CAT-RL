# CAT-RL

This repository contains the code for the [paper]([https://openreview.net/pdf?id=Lb8ZnWW_In6](https://proceedings.mlr.press/v216/dadvar23a.html)):

Conditional Abstraction Trees for Sample-efficient Reinforcement Learning.<br/>
[Mehdi Dadvar](https://www.mdadvar.net/), 
[Rashmeet Kaur Nayyar](https://rashmeetnayyar.com), and
[Siddharth Srivastava](http://siddharthsrivastava.net/). <br/>
39th Conference on Uncertainty in Artificial Intelligence, 2023. <br/>

<!-- [Paper](https://openreview.net/pdf?id=Lb8ZnWW_In6) -->
<!-- | [Extended Version]() | [Talk]() | [Slides]() | [Poster]() -->
<br />

## Directory Structure

```
|-- baselines/
|-- final_plots/
|-- results/
|-- src/
|   |-- envs/
|   |-- maps/
|   |-- abs_tree.py
|   |-- abstraction.py
|   |-- analyze.py
|   |-- hyper_param.py
|   |-- learning.py
|   |-- log.py
|   |-- method_hrl.py
|   |-- method_q.py
|   |-- results.py
|-- README.md
|-- method_catrl.py
|-- requirements.txt
```

- src/method_catrl.py: This file contains the code for CAT-RL. 
- src/: This directory contains the code for CAT-RL, and HRL and Q-learning baselines.
- final_plots/: This directory contains the plots for the final paper included in the paper.
- results/: This directory contains results in the form of pickle and csv files.
- baselines/: This directory contains the code for JIRP and deep learning baselines.
<br />

## Instructions to run the code

1. Install all the dependencies by executing the following command.
```
pip install -r requirements.txt
```

2. Uncomment the domain and hyper-parameters in the src/hyper_param.py. Make sure to uncomment only one domain.

3. Execute the following command to run CAT-RL algorithm for the domain. This will generate the output files within the results/ directory.
```
python3 method_catrl.py
```
<br />

Please note that this is a research code and not yet ready for public delivery,
hence most parts are not documented.

In case of any queries, please contact [mdadvar@asu.edu](mailto:mdadvar@asu.edu)
or [rmnayyar@asu.edu](mailto:rmnayyar@asu.edu).
<br />

## Contributors

[Mehdi Dadvar](https://www.mdadvar.net/)<br/>
[Rashmeet Kaur Nayyar](https://rashmeetnayyar.com)<br/>
[Siddharth Srivastava](https://siddharthsrivastava.net/)<br/>
<br/>

## Citation
```
@inproceedings{dadvar2023conditional,
  title={Conditional abstraction trees for sample-efficient reinforcement learning},
  author={Dadvar, Mehdi and Nayyar, Rashmeet Kaur and Srivastava, Siddharth},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={485--495},
  year={2023},
  organization={PMLR}
}
```
