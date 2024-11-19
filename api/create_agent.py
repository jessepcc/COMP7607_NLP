from os.path import join, dirname
from typing import Optional, List
from letta import LLMConfig, EmbeddingConfig, create_client, LocalClient, ChatMemory, Block
from letta.schemas.block import Block 
import json
from tools import all_tools
from dotenv import load_dotenv

# dotenv_path = join(dirname(__file__), '.env')
load_dotenv()

# Initialize the client and create tools
client = create_client()

base_llm_config = LLMConfig(
    model="gpt-4o-mini",
    model_endpoint_type="openai",
    # model_endpoint="https://ark.cn-beijing.volces.com/api/v3",
    # model_endpoint="https://api2.aigcbest.top/v1",
    model_endpoint="https://inference.memgpt.ai", # memgpt free service
    context_window=16384
    # model = "memgpt-openai",
    # model_endpoint = "https://inference.memgpt.ai",
    # model_endpoint_type = "openai",
    # context_window = 8192,
)

client.set_default_llm_config(
    base_llm_config         
)
client.set_default_embedding_config( 
    EmbeddingConfig(
        embedding_endpoint_type="openai",
        embedding_endpoint="https://api.openai.com/v1",
        # embedding_endpoint="https://ark.cn-beijing.volces.com/api/v3",
        # embedding_endpoint="https://api2.aigcbest.top/v1",
        embedding_model="text-embedding-ada-002",
        embedding_dim=1536,
        embedding_chunk_size=300
        # embedding_endpoint_type = "hugging-face",
        # embedding_endpoint = "https://embeddings.memgpt.ai",
        # embedding_model = "BAAI/bge-large-en-v1.5",
        # embedding_dim = 1024,
        # embedding_chunk_size = 600
    )
)


class TaskMemory(ChatMemory):
    def __init__(self, human: str, persona: str, tasks: List[str]): 
        super().__init__(human=human, persona=persona) 
        self.link_block( 
            Block(
                limit=2000, 
                value=json.dumps(tasks), 
                label="tasks"
            )
        )

    def task_queue_push(self, task_description: str) -> Optional[str]:
        """
        Push to a task queue stored in core memory. 

        Args:
            task_description (str): A description of the next task you must accomplish. 
            
        Returns:
            Optional[str]: None is always returned as this function does not produce a response.
        """
        import json
        tasks = json.loads(self.memory.get_block("tasks").value)
        tasks.append(task_description)
        self.memory.update_block_value("tasks", json.dumps(tasks))
        return None

    def task_queue_pop(self) -> Optional[str]:
        """
        Get the next task from the task queue 
 
        Returns:
            Optional[str]: The description of the task popped from the queue, 
            if there are still tasks in queue. Otherwise, returns None (the 
            task queue is empty)
        """
        import json
        tasks = json.loads(self.memory.get_block("tasks").value)
        if len(tasks) == 0: 
            return None
        task = tasks[0]
        print("CURRENT TASKS: ", tasks)
        self.memory.update_block_value("tasks", json.dumps(tasks[1:]))
        return task

with open('persona.txt', 'r') as file:
    persona = file.read()

with open('human.txt', 'r') as file:
    human = file.read()

# Initialize the agent with loaded tasks
agent_memory = TaskMemory(human=human, persona=persona, tasks=[])
agent_state = client.create_agent(
    name="PG Copilot", 
    memory=agent_memory,
    tools=[tool.name for tool in all_tools]
)

print(f"Created agent: {agent_state.name} with ID {str(agent_state.id)}")

# Created agent: PG Copilot with ID agent-d7da8047-00ca-4010-ae97-6bae5c7ecb97