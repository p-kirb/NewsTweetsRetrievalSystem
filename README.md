# NewsTweetsRetrievalSystem
#### Repository for a search engine that retrieves breaking news tweets from the official accounts of CNN, BBC, and The Economist between 2010-2021.
#

## How to run the app
### 1. Pull the docker image
- For arm architecture:
```
docker pull dukekush/news_retrieval:latest-arm
```
- For amd architecture:
```
docker pull dukekush/news_retrieval:latest-amd
```
### 2. Run the container

- For arm architecture:
```
docker run -dp 8080:5000 dukekush/news_retrieval:latest-arm
```
- For amd architecture:
```
docker run -dp 8080:5000 dukekush/news_retrieval:latest-amd
```
### 3. Click:

- [http://localhost:8080](http://localhost:8080)

## Project Structure:
```
NewsTweetsRetrievalSystem
  ├── README.md
  ├── docker-compose.yml
  ├── docker_build
  │   ├── Dockerfile
  │   └── setup.py
  └── retrieval_system
      ├── data
      ├── functions
      ├── requirements.txt
      ├── scripts
      ├── tmp
      └── web_app
```
## Data Folder Structure
```
data
  ├── 01_input_data
  │   ├── tweets_bbc.csv
  │   ├── tweets_cnn.csv
  │   └── tweets_eco.csv
  ├── 02_joined_data
  │   └── joined_data.csv
  ├── 03_processed_data
  │   └── processed_data.csv
  ├── 04_model_components
  │   ├── custom_components
  │   └── lucene_components
  └── 05_evaluation
      ├── kappa_test
      ├── plots
      ├── relevant_queries.csv
      └── rocchio_results.json
```