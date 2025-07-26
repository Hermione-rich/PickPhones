from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def initialize_message(self):
        pass
    
    @abstractmethod
    def ai_message(self, payload):
        pass
    
    @abstractmethod
    def system_message(self, payload):
        pass
    
    @abstractmethod 
    def user_message(self, payload):
        pass
    
    @abstractmethod
    def get_response(self):
        pass
    

class DeepSeekLLM(BaseLLM):
    def __init__(self, model = "deepseek-chat"):
        super().__init__()
        self.client = OpenAI(
            api_key = os.getenv("DEEPSEEK_KEY"),
            base_url = os.getenv("DEEPSEEK_URL")
        )
        
        self.model = model
        self.messages = []
    
    def __str__(self):
        for message in self.messages:
            print(message)
    
    def initialize_message(self):
        self.messages = []

    def ai_message(self, payload):
        self.messages.append({"role": "ai", "content": payload})

    def system_message(self, payload):
        self.messages.append({"role": "system", "content": payload})

    def user_message(self, payload):
        self.messages.append({"role": "user", "content": payload})
        
    def get_response(self,temperature = 0.8):
    
        response = self.client.chat.completions.create(
        model="deepseek-chat",
        messages=self.messages,
        stream=False
)
        return response.choices[0].message.content
    
    def chat(self,text):
        self.initialize_message()
        self.user_message(text)
        response = self.get_response()
        return response
    
    