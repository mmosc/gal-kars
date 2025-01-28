from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
from transformers import BitsAndBytesConfig #!!

import pandas as pd
import re
from tqdm import tqdm
import time
import os

# import ratings
ratings = pd.read_csv("in/train_sorted.tsv", sep='\t', names=['user','item','rating'])
names = pd.read_csv("in/mapping_items.tsv", sep='\t', names=['id','name'])
pos_ratings = ratings[ratings['rating'] == 1]
total_users = set(ratings['user'])

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

# define system prompt
sysprompt_user = """Considering the books given in the user prompt, write a list of these elements:
the user's favourite genre;
the user's favourite kind of plot;
the user's favourite writing style;
the user's favourite subjects;
the user's favourite topics;
the user's favourite kind of books (novel, essay, poetry).
Infer this information from the books the user likes.
Furthermore, strictly follow this format in the output without adding anything else:
<Favourite genre: <genre_1>, <genre_5>. Favourite kind of plot: <plot_1>, ..., <plot_n>.> and so on.
Make sure that every <genre_1>, ..., <genre_n>, ..., <plot_1>, <plot_n> etc does not contain multiple values (like "polar and thriller"),
but rather separates them with commas ("polar, thriller").
If you are unsure about something, avoid providing an answer.
"""

# start inferences
for user_i in tqdm(total_users, total=len(total_users)):

  if os.path.exists(f'users/user{user_i}_sysprompt1.json'):
    print(f'skipping user {user_i}, already existing')
    continue

  # keep track of the time
  start_time = time.time()

  user_i_pos_ratings = pos_ratings.loc[pos_ratings['user']==user_i]
  user_i_pos_ratings_size = int(user_i_pos_ratings.size/3)
  print("Starting user ", user_i, "... (", user_i_pos_ratings_size, " positive ratings )")

  prompt = "My favourite books are: "
  for i in range(user_i_pos_ratings_size):

    book_name = (names.loc[names['id']==user_i_pos_ratings['item'].values[i]])['name'].values[0].split(';')[0]
    book_name = book_name.replace("_"," ")
    book_name = re.sub("[\(\[].*?[\)\]]", "", book_name).strip()
    prompt = prompt + book_name + "; "
  prompt = prompt[:-2] + "."

  print(prompt)


  input_text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}, {'role': 'system', 'content': sysprompt_user}], tokenize=False)

  result = pipe(input_text)[0]['generated_text']

  mydict = {"prompt": input_text,
          "plain": result}

  result = result.replace('<','')
  result = result.replace('>','')
  result = result.split('.')

  for line in result:

    if (len(line.strip().split(': ')) >= 2):
      feat = line.replace("Favourite ","").strip().split(': ')[0].lower()
      vals = line.strip().split(': ')[1].lower()
      valsList = vals.split(', ')

      # save current results
      mydict[feat] = valsList


  json_object = json.dumps(mydict)
  print(mydict)


  print("JSON file created correctly.")
  filename = "users/user" + str(user_i) + "_sysprompt1.json"
  with open(filename, "w") as outfile:
      outfile.write(json_object)



  end_time = time.time()
  print("User ", user_i, " finished in ", round(end_time-start_time,2), " seconds.\n")
