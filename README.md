## About The Project

We plan to re-implement the two holistic semi-supervised learning approaches: [FixMatch](https://arxiv.org/abs/2001.07685) and [MixMatch](https://arxiv.org/abs/1905.02249) in Pytorch.


Use the `README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

This section lists all major frameworks/libraries used to bootstrap our project.

* python 3
* pytorch 1.7
* torchvision 0.8
* numpy 1.22
* pyyaml 6.0
* matlabplot 3.5
* tensorboard

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

These give you instructions on setting up your project locally or on Google Colab.
To execute our Project follow these simple steps.

### Installation & Execution (Local Machine)

_Instructions on installing and setting up our app on your local machine._

1. In this project, we work with Python 3. We recommend installing Anaconda or miniconda with Python 3.
2. Provided environment.yaml which contains a list of libraries needed to set the environment 
      ```sh
      $ conda env create -f environment.yaml
      ```
3. To install PyTorch, please refer to the Official Link: https://pytorch.org/
4. To execute our code locally, navigate to ".\7643_final_project\" folder and execute the following command.
      ```sh
      $ python main.py
      ```
 5. To run a different configuration, update line#35 in main.py to point to a differnt .yaml file.
      ```sh
      parser.add_argument('--config', default='./config/config_fixmatch_cifar10red40_1000_regular.yaml')
      ```
    The yaml files we ran for our project are available in ".\7643_final_project\config\"

### Installation & Execution (Google Colab)

_Instructions on installing and setting up our app on your Google Colab._

1. To execute the project on Google Colab, upload the "notebook.ipynb" file to Google Colab.
2. Review the Google Drive project directory and update it as needed.
      ```sh
      %cd '/content/drive/MyDrive/final_project/'
      ```
3. To run a different configuration, update line#35 in main.py to point to a differnt .yaml file.
      ```sh
      parser.add_argument('--config', default='./config/config_fixmatch_cifar10red40_1000_regular.yaml')
      ```
    The yaml files we ran for our project are available in ".\7643_final_project\config\"

## Results

### CIFAR10
| #Labels | 40 | 70 | 250 | 1000 | 4000 | 50000 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Paper (RA) | 86.19 ± 3.37 | - |94.93 ± 0.65 | - |95.74 ± 0.05| - |
| This code | - |92.11 | 94.67 | 95.14 | 95.53 | 96.67
<!-- Authors -->
## Authors

* Vimal Venugopal - vimalkumar.engr@gmail.com

Project Link: [https://github.com/vimvenu-rgb/Fix_Mix_Match-Project)

<p align="right">(<a href="#readme-top">back to top</a>)</p>




