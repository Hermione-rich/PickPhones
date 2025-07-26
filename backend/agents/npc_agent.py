import sys
sys.path.append("./")
from BaseAgent import BaseAgent
from llms.LanggraphGPT import LanggraphGPT
from langchain.prompts import PromptTemplate

class NPCAgent(BaseAgent):
    def __init__(self, 
                 name: str, 
                 source: str, 
                 model_name: str):
        super().__init__(name, source, model_name)
        self.source = source
        self.llm = LanggraphGPT("gpt-4o")
        
    def get_respond(self, messages: str="你是谁？") -> str:
        prompt = PromptTemplate.from_template(
            "你是来自小说《{source}》的{name},以下是你们的对话片段: \n{messages}\n, 请回应对方的对话,回应风格要满足你自身角色特点"
        )
        my_prompt = prompt.format(source = self.source,name = self.name, memory = self.get_memory(), messages = messages)
        reply = self.llm.chat(my_prompt)
        self.remember(f"{self.name}: {reply}")
        return reply

    def __str__(self):
        return self.get_respond()

if __name__ == "__main__":
    npc = NPCAgent(name="蓝忘机", source = "魔道祖师", model_name="gpt-4o")
    print(npc)