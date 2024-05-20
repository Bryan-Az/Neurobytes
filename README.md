# Neurobytes: A Music Recommendation App

The 4 main non-code artifacts created to demonstrate this project include a slide deck, project artifact screenshots, and a video presentation presented by all members of our team!


All code artifacts for this project are also included within this repo. 


## Project Architecture
The timeline for this project is 2 weeks. As such, our database is small which allows for quick model testing, training, and evaluation. It is also available on the cloud for persistent storage.

Link to the Project Demo: https://huggingface.co/spaces/Neurobytes/Neurobytes_Music_Recommender/tree/main

### UX Design using Gradio
Gradio is a cloud-based web application development platform. In this project, we used Gradio to create the demo application.

The user interface consists of:

1. Pages related to the application itself.
2. A model performance metric dashboard.


### ML Ops on Google Colab and Google Drive

The Large-Language Recommendation Model is built to support: 

1. Training, and Inference
2. ML Prod. Cloud Pipeline
3. Re-training & CI/CD
4. Evaluation using Tensorboard

### Cloud Database - AWS RDBS, MySQL

The cloud database chosen will have data related to:

1. User-to-Song rating/playback information
2. Song-to-Song similarity information based on musicality.


## Model Architecture

The model architecture currently decided is a Large-Language model trained on text and tabular data. As the scope of this project is restricted, pre-processed .wav file data will be included that captures the essence of the music in the database for model training, as well as sample information about users.


# Roles & Contributions
The team consists of 4  master students in software engineering specializing in data science at San Jose State University.
## Front-End
### Kelly Nguyen
UX Design & Gradio

### Joash Muganda
Platforms, CI/CD, MLOps

## Back-End 
### Nick Kornienko
MLOps

### Alexis Ambriz
MLOps, AWS Cloud Database

# Resources

https://github.com/visenger/awesome-mlops
