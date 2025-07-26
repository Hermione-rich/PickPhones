from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, name: str, source: str, model_name: str):
        self.name = name
        self.source = source
        self.model_name = model_name
        self.memory = []
    
    def remember(self, message: str):
        self.memory.append(message)
    
    def get_memory(self):
        return "\n".join(self.memory[-10:])
    
    
    @abstractmethod
    def get_respond(self, messages: str) -> str:
        ...