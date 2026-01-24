import pandas as pd

import json

df = pd.read_csv('billable_codes.csv')
sft_data = []

for index, row in df.iterrows():
    _id = ("0" * 12)[:12-len(str(index))] + str(index)
    code = row['code']
    # add period after 3rd character in case len(code) > 3
    if len(code) > 3 and '.' not in code:
        code = code[:3] + '.' + code[3:]
    text = row.get('description', row.get('text', ''))
    assert text, f"Text is empty for code {code}"
    conversations = [
        {
            "from": "human",
            "value": "What is the ICD-10CM code for '{}'?".format(text)
        },
        {
            "from": "gpt",
            "value": f"{code}"
        }
    ]
    sft_data.append({
        "id": _id,
        "conversations": conversations
    })
import random
# shuffle the conversations
random.shuffle(sft_data)

with open('train_set_specific_grpo.json', 'w') as f:
    json.dump(sft_data, f, indent=4)