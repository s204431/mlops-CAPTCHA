# CAPTCHA solving CNN

In this project we want to develop a deep learning model based on Convolutional Neural Networks to solve CAPTCHAs, as they are often used to differentiate between humans and machines. In addition to this we want to create a structure in this project that can be used as a template for further projects in machine learning.

We are using the *[PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)* framework, which is a library containing for example different models applicable for image classification. We make use of the pretrained *resnet18* as the model of choice for our project. This model is not too complex but it should be enough for this project.

For our project we use the [*CAPTCHA Characters Dataset*](https://www.kaggle.com/datasets/tahabakhtari/captcha-characters-dataset-118k-images) published by Taha Bakhtari on kaggle. It covers over 118.000 black-and-white images of letters and digits from CAPTCHAs with an image size of 52x32 pixel. The dataset contains images of 20 different CAPTCHA characters being various digits and letters. This gives rise to an image classification problem with 20 classes.

Here’s an example of a CAPTCHA image from our dataset:

![Example CAPTCH.](data/raw/2_10067.png)


## Model
Here we talk about our model that we use.

## Frameworks and Tools (maybe)
Here we list the frameworks and tools that we use and maybe explain some important ones.

## Setup and installation (important)
Here we describe how to setup the project to reproduce our results.

## Progress (maybe)
Maybe a section to keep track of our progress.

## Future steps (maybe)
The next steps.

## Project structure
The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
