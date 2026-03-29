from agent_with_retriever_tool import agent

if __name__ == "__main__" :
  print ("What would you like to ask the coding agent today? Type 'bye' to exit. \n")

  config = {"configurable": {"thread_id": "rag-session-1"}}
  while True :
    question = input ("Your input: ").strip()
    if question.lower() == "bye" :
      break
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    print(f"\nAgent: {result['messages'][-1].content}. \nType bye to exit.")