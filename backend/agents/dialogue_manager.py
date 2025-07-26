import sys
sys.path.append("../")
from BaseAgent import BaseAgent
from chat_agent import RoleAgent
from npc_agent import NPCAgent


class DialogueManager:
    def __init__(self, agents: list[BaseAgent]):
        self.agents = agents
        self.dialogue_history = []
        
    def run_dialogue(self, rounds: int = 5, seed_message: str = "你怎么看？"):
        last_message = seed_message
        for i in range(rounds):
            agent = self.agents[i % len(self.agents)]
            print(f"\n【{agent.name}】:")
            response = agent.get_respond(messages=last_message)
            print(response)
            self.dialogue_history.append((agent.name, response))
            agent.remember(f"{agent.name}: {response}")
            last_message = response  # 传给下一个角色

if __name__ == "__main__":
    a = NPCAgent(name="魏无羡", source="魔道祖师", model_name="gpt-4o")
    b = NPCAgent(name="蓝忘机", source="魔道祖师", model_name="gpt-4o")
    c = NPCAgent(name="江澄", source="魔道祖师", model_name="gpt-4o")
    

    manager = DialogueManager(agents=[a, b, c])
    manager.run_dialogue(rounds=6, seed_message="江澄怎么一脸死给的样子？")