from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
from transformers import BitsAndBytesConfig #!!

import pandas as pd
import re
from tqdm import tqdm
import time

dataset = 'ML1M'
item_type = 'movie'

# import item names
names = pd.read_csv(f"in/mapping_items.tsv", sep='\t', names=['id','name'])
total_items = set(names['id'])

# load model
model_name = "nvidia/Llama3-ChatQA-1.5-8B"

bnb_config = BitsAndBytesConfig( 
  load_in_4bit=True,
  bnb_4bit_use_double_quant=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16
)


tokenizer_name = model_name
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation='eager',
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config #!!
    )
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True,  use_fast=False, use_cache=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side='right'
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, device_map="auto",\
                 return_full_text=False, max_new_tokens= 512)


sysprompt_item = """
Considering the movie name in the user prompt, write a list of properties for each of these elements: 
the name of the writer(s); 
the name of the director(s);
the subject(s) of the movie; 
the topic(s) of the movie; 
the genre(s) of the movie; 
actors and actrees starring in the movie; 
the kind of soundtrack of the movie; 
the kind of plot of the movie; 
the music composer of the soundtrack of the movie; 
the country in which the movie has been produced. 
Infer these information for the given movie. 
Furthermore, strictly follow this format in the output without adding anything else:
<genres: <genre_1>, ..., <genre_n>. plot: <plot_1>, ..., <plot_n>.> and so on. 
Make sure that every <genre_1>, <genre_n>, <plot_1>, <plot_n> etc does not contain multiple values (like "dark and gritty"), but rather separates them with commas ("dark", "gritty").
If you are unsure about something, avoid providing an answer.
"""

# start inferences
for count, item_id in enumerate(tqdm(total_items, total=len(total_items))):
    
  start_time = time.time()
  title = names[names['id'] == item_id]['name'].values[0]

  prompt = f"The {item_type} you have to infer information is: "
  if dataset == 'DBbook':
    title = title.split(';')[0]
  else:
    title = title.split('/')[-1]
  title = title.replace("_"," ")
  title = re.sub("[\(\[].*?[\)\]]", "", title).strip()
  prompt = prompt + title + "."

  print("Starting ID", item_id, " (title: ", title,")...")
  input_text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}, {'role': 'system', 'content': sysprompt_item}], tokenize=False)

  result = pipe(input_text)[0]['generated_text']
  print(prompt)
  print(result)

  mydict = {"prompt": prompt,
    "plain": result}

  result = result.replace('<','')
  result = result.replace('>','')
  result = result.split('.')

  for line in result:
    if (len(line.strip().split(': ')) >= 2):
      feat = line.replace("Favourite ","").strip().split(': ')[0].lower()
      vals = line.strip().split(': ')[1].lower()
      valsList = vals.split(', ')
      mydict[feat] = valsList


  json_object = json.dumps(mydict)

  print("JSON file created correctly.")
  filename = f"items/item" + str(item_id) + ".json"
  with open(filename, "w") as outfile:
      outfile.write(json_object)


  end_time = time.time()
  print("Row ", item_id, " finished in ", round(end_time-start_time,2), " seconds.\n")