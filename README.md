# Speech Automation

## Introduction
This project is a real-time speech processing WebSocket API built using FastAPI. It allows clients to send audio files over a WebSocket connection, which are then processed to generate text transcriptions, correct the grammar, and return a synthesized audio response.

The core services of the project include:

Speech-to-Text: Converts audio files into text using OpenAI's Whisper API.
Grammar Correction: Leverages Google's Generative AI to correct the grammar of the transcribed text.
Text-to-Speech: Converts the corrected text back into speech using OpenAI's Text-to-Speech API.
The project is designed to handle client-server communication over WebSockets, ensuring real-time interaction, and makes use of a clean, modular architecture that separates concerns into controllers, services, and routes. The application also integrates with external APIs like OpenAI and Google to provide state-of-the-art AI-driven speech and text processing functionalities.

## Basic / Prerequisites Setup:
- The [setupREADME.md](./setup/setupREADME.md) file contains the initial setup steps that needs to followed.

## Directory Structure

```bash
OgmaConceptions_Speech_Automation/
│
├── app/
│   ├── attribute_selectors/        
│   ├── controllers/                
│   ├── db/                        
│   │   ├── seeder.py               
│   │   ├── __init__.py             
│   │   └── models/                 
│   ├── dependencies/               
│   ├── repositories/               
│   ├── routes/                     
│   │   ├── __init__.py           
│   │   └── Chat_routes.py                      
│   ├── schemas/
│   │   ├── chat_schema.py                     
│   ├── services/                   
│   │   ├── grammar_correction.py  
│   │   ├── speech_to_text.py      
│   │   ├── text_to_speech.py      
│   │   ├── websocket_service.py      
│   └── utils/                      
│       ├── config.py               
│       ├── constants.py           
│       ├── exceptions.py           
│       ├── logger.py               
│       ├── helpers/                
│       │   ├── password_helper.py  
│       │   ├── datetime_helper.py  
│       │   ├── token_helper.py     
│       │   ├── file_helper.py      
├── setup/                          
│   ├── setupREADME.md              
│   └── setup.sh                    
│   ├── db_model_versions/          
├── .env.example                                    
├── main.py                         
├── README.md                       
└── requirements.txt                
```


## Data Model:


## Database Migration Steps:


## Flow Chart:

![alt text](image.png)

## Initiate Server Script:


