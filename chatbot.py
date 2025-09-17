import os 
from dotenv import load_dotenv  
load_dotenv()

client=OpenAI(api_key= os.environ["API_KEY"])
system_prompt=" A am very proud of you using the system ,whats on your ,mind today"
user_prompt=input("write your question here ")
chat_completion=client.chat.completions.create(
    messages=[{
        "role":"system","content":system_prompt
    },{
        "role":"user","content":user_prompt
    }],
    model="gpt-4o"
)
response_text=chat_completion.choices[0].message.content
print(response_text)