# CS5494 - Assignment 1

This project is CS5494 - Assignment 1, completed by a team of three: Chen Jiaqi, Zhu Pengyu, and Li Shuo. The project involved implementing three models—FVSBN, NADE, and GAN—based on the MNIST dataset and completing the corresponding lab reports. Chen Jiaqi implemented the FVSBN model, Zhu Pengyu implemented the NADE model, and Li Shuo implemented the GAN model and compiled the report.

## Models and documentation

Task1.1, Task1.2, and Task1.3 are the project directories for the three models, respectively, with their corresponding reports located in each directory as task1.x_report.pdf.

| Task | Model | Project documentation | Report |
| --- | --- | --- | --- |
| 1.1 | Fully Visible Sigmoid Belief Network (FVSBN) | [task1.1_jchen3785/README.md](./task1.1_jchen3785/README.md) | [Task 1.1 report](./task1.1_jchen3785/task1.1_report.pdf) |
| 1.2 | Neural Autoregressive Distribution Estimator (NADE) | [task1.2_pengyuzhu3/README_task2.md](./task1.2_pengyuzhu3/README_task2.md) | [Task 1.2 report](./task1.2_pengyuzhu3/task1.2_report.pdf) |
| 1.3 | Generative Adversarial Network (GAN) | [task1.3_sli986/README.md](./task1.3_sli986/README.md) | [Task 1.3 report](./task1.3_sli986/task1.3_report.pdf) |

## Combined report

[61127749_60873400_61111113.pdf](./61127749_60873400_61111113.pdf) is a merged version of the three reports; pages 1–5 cover Task 1.1, pages 6–29 cover Task 1.2, and the remaining pages cover Task 1.3.

## Running the models

Open the documentation linked above for the model you want to run. Each subproject provides its own environment requirements and execution instructions; run its commands or notebooks from that subproject directory.

Install dependencies using the corresponding subproject's `requirements.txt`. Data preparation, training, evaluation, and saved results are described in the individual project documentation.

MNIST data files are excluded from this repository. Download or regenerate them by following the data preparation instructions in each subproject.

The repository includes the final GAN checkpoints (E0: epoch 2; E1-E3: epoch 20), all saved sample figures, and experiment logs. Intermediate GAN checkpoints are excluded; rerun training to regenerate them.
