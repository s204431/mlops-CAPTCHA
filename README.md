# Detecting CAPTCHA characters using TIMM Resnet

## Project Description
In this project we want to develop a deep learning model based on Convolutional Neural Networks (CNN) to solve CAPTCHAs, as they are often used to differentiate between humans and machines. This is a good measure of how well the model can adapt to a method, intended to make it fail. In addition to this we want to create a structure in this project that can be used as a template for further projects in machine learning. Not only will this make it easier to set up a new project, but it will also help in the future to understand the general structure and the meaning behind it in other peoples work.

We are using the *[PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)* framework, which is a library containing for example different models applicable for image classification. We make use of the pretrained *resnet18* with a final layer to match the output size as the model of choice for our project. This model is not too complex but it should be enough for this project. Additionally we make use of pytorch lightning to train the model as it handles boilerplate code cleanly.

For our project we use the [*CAPTCHA Characters Dataset*](https://www.kaggle.com/datasets/tahabakhtari/captcha-characters-dataset-118k-images) published by Taha Bakhtari on kaggle. It covers over 118.000 black-and-white images of letters and digits from CAPTCHAs with an image size of 52x32 pixel. The dataset contains images of 20 different CAPTCHA characters being various digits and letters. This gives rise to an image classification problem with 20 classes.

Here’s an example of a CAPTCHA image from our dataset:

![Example CAPTCH.](reports/figures/2_10067.png)

For now the data was loaded manually, using less than one gigabyte, but we will later implement the downloading process in the [data.py](src/captcha/data.py) and save the data using version control. We plan to start by randomly sampling 10.000 images at the start of development. We will then consider training on the entire dataset later on.

## Project structure
The directory structure of the project looks like this:
```txt
├── .dvc/                       #DVC configuration
│   ├── .gitignore
│   └── config
├── .github/                    # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       ├── cml_data.yaml
│       ├── codecheck.yaml
│       ├── deploy_docs.yaml
│       ├── pre_commit.yaml
│       ├── stage_model.yaml
│       └── tests.yaml
├── configs/                    # Configuration files
│   ├── gcloud/
│   │   ├── config_cpu.yaml
│   │   └── config_gpu.yaml
│   ├── model/
│   │   └── model.yaml
│   ├── optimizer/
│   │   ├── Adam_opt.yaml
│   │   └── Adam_opt_sweep.yaml
│   ├── config.yaml
│   └── default_config.yaml
├── dockerfiles/                # Dockerfiles
│   ├── backend.dockerfile
│   ├── evaluate.dockerfile
│   ├── frontend.dockerfile
│   ├── monitoring.dockerfile
│   └── train.dockerfile
├── docs/                       # Documentation
│   ├── mkdocs.yml
│   ├── README.md
│   └── source/
│       ├── backend_monitoring.md
│       ├── bentoml_service.md
│       ├── data.md
│       ├── dataset.md
│       ├── evaluate.md
│       ├── index.md
│       ├── model.md
│       ├── train.md
│       └── utils.md
├── models/                     # Trained models
│   └── model_fully_trained.pth # Example of a fully trained model
├── notebooks/                  # Jupyter notebooks
├── reports/                    # Reports
│   ├── README.md
│   ├── report.py
│   └── figures/
├── src/                        # Source code
│   ├── captcha/
│   │   ├── __init__.py
│   │   ├── backend_monitoring.py
│   │   ├── bentoml_client.py
│   │   ├── bentoml_service.py
│   │   ├── data.py
│   │   ├── dataset.py
│   │   ├── evaluate.py
│   │   ├── frontend.py
│   │   ├── logger.py
│   │   ├── model.py
│   │   ├── onnx_export.py
│   │   ├── train.py
│   │   └── utils.py
├── tests/                    # Tests
│   ├── integrationtests/
│   │   └── test_api.py
│   ├── performancetests/
│   │   ├── locustfile_backend.py
│   │   ├── locustfile_frontend.py
│   │   └── test_model.py
│   ├── test_images/
│   └── unittests/
│       ├── __init__.py
│       ├── test_data.py
│       ├── test_dataset.py
│       ├── test_model.py
│       └── test_train.py
├── .dockerignore
├── .dvcignore
├── .gcloudignore
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── README.md                 # Project README
├── backend_monitoring.yaml
├── cloudbuild.yaml
├── config.yaml
├── data.dvc
├── link_model.py
├── pyproject.toml            # Python project file
├── requirements.txt          # Project requirements
├── requirements_backend.txt  # Backend requirements
├── requirements_dev.txt      # Development requirements
├── requirements_frontend.txt # Frontend requirements
├── tasks.py                  # Project tasks
└── vertex_ai_train.yaml
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
