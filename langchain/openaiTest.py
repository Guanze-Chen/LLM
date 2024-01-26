import openai

api_key = "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"

openai.api_key = api_key

import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key= api_key,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion)