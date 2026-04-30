from multi_tool_agent import agent
eval_set = [
    {
        "input": "[User ID: 3] What is my employment status?",
        "expected_tools": ["lookup_user"],
        "expected_keywords": ["retiree", "finance"]
    },
    {
        "input": "[User ID: 2] I have not received my pay check for the past 3 months. I would like to see what is going on.",
        "expected_tools": ["create_ticket", "classify_ticket", "escalate_ticket", "lookup_user", "data_retriever"],
        "expected_keywords": ["hr", "finance"]
    },
    {
        "input": "[User ID: 3] My laptop has been very slow to the point that it takes more than 10 minutes to just turn it on. Can I get a replacement?",
        "expected_tools": ["create_ticket", "classify_ticket", "data_retriever", "lookup_user"],
        "expected_keywords": ["it", "loaner device", "one business day"]
    },
    {
        "input": "[User ID: 1] I would like to see what the maternity leave process would be like. I am due in about a month and my manager had forgotten to apply in my behalf so I need it to be processed ASAP.",
        "expected_tools": ["create_ticket", "classify_ticket", "data_retriever", "escalate_ticket", "lookup_user"],
        "expected_keywords": ["fmla", "primary caregiver", "30 days", "birth", "escalated"]
    },
    {
        "input": "[User ID: 3] I would like to learn about who our main competitors are and what is their strategy to try to beat us in the market.",
        "expected_tools": ["lookup_user", "data_retriever"],
        "expected_keywords": ["out of scope"]
    },
    {
        "input": "[User ID: 2] I would like to learn about my employee status. I was told that I should be promoted to a full time employee but I’m not sure if it’s been processed.",
        "expected_tools": ["create_ticket", "classify_ticket", "lookup_user"],
        "expected_keywords": ["part time", "hr", "escalated"]
    },
    {
        "input": "[User ID: 1] I am unable to install git on my computer. How should I install it?",
        "expected_tools": ["create_ticket", "classify_ticket"],
        "expected_keywords": ["it", "ticket"]
    },
    {
        "input": "[User ID: 4] I would like to learn about what employee 2 works on. He seems to always claim that he is very busy but I think he might be lying.",
        "expected_tools": ["lookup_user", "data_retriever"],
        "expected_keywords": ["privacy", "unable", "policy"]
    },
    {
        "input": "[User ID: 1] I had asked you earlier about a paycheck I had not received and you answered that a ticket was created and raised. I have not received any support in the past three months. Did you really create a ticket?",
        "expected_tools": ["data_retriever", "create_ticket", "classify_ticket", "escalate_ticket", "lookup_user"],
        "expected_keywords": ["hr", "escalated"]
    },
    {
        "input": "[User ID: 1] You said that my laptop was due for a renewal but I have not received one. Why did I not receive my new laptop when you promised it?",
        "expected_tools": ["data_retriever", "create_ticket", "classify_ticket", "escalate_ticket", "lookup_user"],
        "expected_keywords": ["raised", "guideline"]
    }
]

keyword_results = []
tool_use_results = []
for eval in eval_set:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": eval["input"]}]},
        config={"configurable": {"thread_id": "rag-session-1"}}
    )
    tools_used = [                                                                                    
        tc['name']  
        for msg in result['messages']
        if hasattr(msg, 'tool_calls')
        for tc in (msg.tool_calls or [])
    ]       
    response_text = result["messages"][-1].content.lower()

    keywordFound = 0
    for keyword in eval["expected_keywords"] :
        if keyword in response_text:
            keywordFound += 1
    toolUsed = 0
    for tool in eval["expected_tools"]:
        if tool in tools_used :
            toolUsed += 1

    keyword_results.append(keywordFound/len(eval["expected_keywords"]))
    tool_use_results.append(toolUsed/len(eval["expected_tools"]))

keyword_pass = sum(keyword_results) / len(keyword_results)
tool_use_pass = sum(tool_use_results) / len(tool_use_results)
print(f"keyword pass rate: {keyword_pass * 100}%")
print(f"too use pass rate: {tool_use_pass * 100}%")