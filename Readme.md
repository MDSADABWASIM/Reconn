# Reconn [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/mdsadabwasim/reconn/main.py)

![Alt text](data/Reconn.png?raw=true)

Web app to do Exploratory data analysis and data preprocessing.

## Pre-requisites

The project was developed using python 3.6.7 with the following packages.
- Pandas
- Numpy
- Scikit-learn
- Pandas-profiling
- Streamlit
- Matplotlib
- Seaborn
- Pillow
- Plotly


Installation with pip:

```bash
pip install -r requirements.txt
```

## Getting Started
Open the terminal in you machine and run the following command to access the web application in your localhost.
```bash
streamlit run main.py
```

## Run on Docker
Alternatively you can build the Docker container and access the application at `localhost:8051` on your browser.
```bash
docker build --tag app:1.0 .
docker run --publish 8051:8051 -it app:1.0
```
## Files
- main.py : Streamlit App script
- requirements.txt : pre-requiste libraries for the project

## Summary
This repository shows the codebase of a web based application Reconn for quick Exploratory data analysis and Data preprocessing.

## Acknowledgements

[Kaggle](https://kaggle.com/), for providing the data for the machine learning pipeline.  
[Streamlit](https://www.streamlit.io/), for the open-source library for rapid prototyping.



