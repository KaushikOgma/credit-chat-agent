# Credit Chat Assistant

## Introduction
This project is a real-time speech processing WebSocket API built using FastAPI. It allows clients to send 

## Basic / Prerequisites Setup:
- The [setupREADME.md](./setup/setupREADME.md) file contains the initial setup steps that needs to followed.

## Directory Structure

```bash
credit_chat_agent/
├── Dockerfile
├── README.md
├── app
│   ├── attribute_selector
│   │   ├── evaluation_attributes.py
│   │   ├── finetune_attributes.py
│   │   ├── log_attributes.py
│   │   ├── metadata_attributes.py
│   │   └── user_attributes.py
│   ├── controllers
│   │   ├── __init__.py
│   │   ├── data_ingestion_controller.py
│   │   ├── evaluation_controller.py
│   │   ├── finetune_controller.py
│   │   ├── log_controller.py
│   │   ├── metadata_controller.py
│   │   └── user_controller.py
│   ├── db
│   │   ├── __init__.py
│   │   └── seeder.py
│   ├── dependencies
│   │   ├── data_ingestion_dependencies.py
│   │   ├── evaluation_dependencies.py
│   │   ├── finetune_dependencies.py
│   │   ├── log_dependencies.py
│   │   ├── metadata_dependencies.py
│   │   └── user_dependencies.py
│   ├── middleware
│   │   └── auth_middleware.py
│   ├── repositories
│   │   ├── evaluation_repositories.py
│   │   ├── finetune_repositories.py
│   │   ├── log_repositories.py
│   │   ├── metadata_repositories.py
│   │   └── user_repositories.py
│   ├── routes
│   │   ├── __init__.py
│   │   ├── data_ingestion_route.py
│   │   ├── evaluation_route.py
│   │   ├── finetune_route.py
│   │   ├── log_route.py
│   │   ├── metadata_route.py
│   │   └── user_route.py
│   ├── schemas
│   │   ├── __init__.py
│   │   ├── auth_schema.py
│   │   ├── data_ingestion_schema.py
│   │   ├── evaluation_schema.py
│   │   ├── finetune_schema.py
│   │   ├── log_schema.py
│   │   ├── metadata_schema.py
│   │   └── user_schema.py
│   ├── services
│   │   ├── data_ingestor.py
│   │   ├── llm_finetune.py
│   │   ├── pinecone_vectorizer.py
│   │   ├── qa_evaluator.py
│   │   └── qa_generator.py
│   └── utils
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── exceptions.py
│       ├── helpers
│       │   ├── auth_helper.py
│       │   ├── common_helper.py
│       │   ├── date_helper.py
│       │   ├── file_helper.py
│       │   ├── openai_token_counter.py
│       │   ├── password_helper.py
│       │   └── prompt_helper.py
│       ├── logger.py
│       └── prompts.yaml
├── app.log
├── docker-compose.yml
├── docs
│   └── pull_request_template.md
├── main.py
├── requirements.txt
└── uploads         
```


## Data Model:


## Database Migration Steps:


## Flow Chart:

## Initiate Server Script:


