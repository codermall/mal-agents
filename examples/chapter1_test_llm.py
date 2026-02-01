import dotenv
from hi_agents import MalAgentsLLM

dotenv.load_dotenv()

def main():
	llm = MalAgentsLLM(model="doubao-seed-1-8-251228")
	response = llm.think([
		{
			"role": "user",
			"content": "Hello, how are you?"
		}
	])
	content = ""
	for chunk in response:
		content += chunk
	print("\n")
	print(content)

if __name__ == "__main__":
	main()
