import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from eval_utils import modified_relaxed_accuracy

quant_config = BitsAndBytesConfig(load_in_8bit = True)

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-ul2", device_map="cuda:0", quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

chart_type = "simple"
question_type = "simple"
question_set = "a"

if question_set == "h":
    question_set = "test_human"
else:
    question_set = "test_augmented"

df = pd.read_json('../dataset/{}/{}_{}.json'.format(chart_type, question_set, question_type))

questions = df['query'].tolist()
gold = df['label'].tolist()

tables = pd.read_json('../tables/{}/{}_{}.json'.format(chart_type, question_set, question_type))
df['table'] = tables[0].tolist()

"""DePlot Prompts."""

import enum


class TemplateKey(enum.Enum):
  QA = 'qa'
  POT = 'pot'

_INSTRUCTION = 'Read the table below to answer the following questions.'
_POT_INSTRUCTION = ('Read the table below to write code to answer the following'
                    ' questions using the variable ans.')
_TABLE = """Year | Democrats | Republicans | Independents
2004 | 68.1% | 45.0% | 53.0%
2006 | 58.0% | 42.0% | 53.0%
2007 | 59.0% | 38.0% | 45.0%
2009 | 72.0% | 49.0% | 60.0%
2011 | 71.0% | 51.2% | 58.0%
2012 | 70.0% | 48.0% | 53.0%
2013 | 72.0% | 41.0% | 60.0%"""


def _add_markup(table):
  parts = [p.strip() for p in table.split("<0x0A>")]
  if parts[0].startswith('TITLE'):
    result = f"Title: {parts[0].split(' | ')[1].strip()}\n"
    rows = parts[1:]
  else:
    result = ''
    rows = parts
  prefixes = ['Header: '] + [f'Row {i+1}: ' for i in range(len(rows) - 1)]
  return result + '\n'.join(prefix + row for prefix, row in zip(prefixes, rows))


def _skip_title(table):
  return '\n'.join(part for part in table.splitlines(keepends=False)
                   if not part.startswith('TITLE'))


_TEMPLATE = f"""{_INSTRUCTION}

{_add_markup(_TABLE)}

Q: In which year republicans have the lowest favor rate?
A: Let's find the column of republicans. Then let's extract the favor rates, they [45.0, 42.0, 38.0, 49.0, 51.2, 48.0, 41.0]. The smallest number is 38.0, that's Row 3.  Row 3 is year 2007. The answer is 2007.

Q: What is the sum of Democrats' favor rates of 2004, 2012, and 2013?
A: Let's find the rows of years 2004, 2012, and 2013. We find Row 1, 6, 7. The favor dates of Demoncrats on that 3 rows are 68.1, 70.0, and 72.0. 68.1+70.0+72=210.1. The answer is 210.1.

Q: By how many points do Independents surpass Republicans in the year of 2011?
A: Let's find the row with year = 2011. We find Row 5. We extract Independents and Republicans' numbers. They are 58.0 and 51.2. 58.0-51.2=6.8. The answer is 6.8.

Q: Which group has the overall worst performance?
A: Let's sample a couple of years. In Row 1, year 2004, we find Republicans having the lowest favor rate 45.0 (since 45.0<68.1, 45.0<53.0). In year 2006, Row 2, we find Republicans having the lowest favor rate 42.0 (42.0<58.0, 42.0<53.0). The trend continues to other years. The answer is Republicans.

Q: Which party has the second highest favor rates in 2007?
A: Let's find the row of year 2007, that's Row 3. Let's extract the numbers on Row 3: [59.0, 38.0, 45.0]. 45.0 is the second highest. 45.0 is the number of Independents. The answer is Independents.


{_INSTRUCTION}"""


_POT_TEMPLATE = f"""{_POT_INSTRUCTION}

{_add_markup(_TABLE)}

Q: What was the average difference in approval rates between democrats and republicans in 2006 and 2007?
#Python
democrats_2006 = 58.0
republicans_2006 = 42.0
# The difference between A and B is A - B which may be negative
difference_2006 = democrats_2006 - republicans_2006
democrats_2007 = 59.0
republicans_2007 = 38.0
difference_2007 = democrats_2007 - republicans_2007
ans = (difference_2006 + difference_2007) / 2

Q: What is the average of Democrats' favor rates of 2004, 2012, and 2013?
#Python
# Years 2004, 2012, and 2013  correspond to rows 1, 6 and 7.
democrats_2004 = 68.1
democrats_2012 = 70.0
democrats_2013 = 72.0
ans = (democrats_2004 + democrats_2012 + democrats_2013) / 3

Q: Which party had less than 50% approval rate in 2013?
#Python
# year 2013 corresponds to row 7. Numbers on row 7 are [72.0, 41.0, 60.0]
# Republicans are the only with less than 50.
ans = "Republicans"

Q: What is percentage of relative increase in approval rate for democrats from 2012 to 2013?
#Python
# year 2012 and 2013 correspond to rows 6 and 7.
# Numbers of democrats on row 6 are 70.0
democrats_2012 = 70.0
# Numbers of democrats on row 7 are 72.0
democrats_2013 = 72.0
ans = 100 * (democrats_2013 - democrats_2012) / democrats_2012

Q: What is the difference between republicans in 2011 and democrats in 2006?
#Python
# year = 2011 corresponds to row 5 and the republicans had a 51.2 rate
republicans_2011 = 51.2
# year = 2006 corresponds to row 2 and the democrats had a 58.0 rate
democrats_2006 = 58.0
# The difference between A and B is A - B which may be negative
ans = republicans_2011 - democrats_2006


{_POT_INSTRUCTION}"""


def get_template(template_key):
  """Returns a template given the key which identifies it."""
  if template_key == TemplateKey.QA:
    return _TEMPLATE
  if template_key == TemplateKey.POT:
    return _POT_TEMPLATE
  else:
    raise ValueError(f'Invalid template key {template_key}')


def build_prompt(template_key, table, question):
  """Builds a prompt given a table, question and a template identifier."""
  template = get_template(template_key)
  if template_key == TemplateKey.QA: # one if condition here 
    return f"""{template}

{_add_markup(table)}

Q: {question}
A: """
  elif template_key == TemplateKey.POT: # another if condition here 
    return f"""{template}

{_add_markup(table)}

Q: {question}
"""

tokenizer.pad_token = '<|endoftext|>'
tokenizer.padding_side = 'left'

queries = []

for i, question in enumerate(questions):
    prompt = build_prompt(TemplateKey.QA, tables[0][i], question)
    queries.append(prompt)

batch_size = 6

batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]

model_responses = []

for i, batch in enumerate(batches):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    inputs.to(model.device)
    outputs = model.generate(**inputs, max_new_tokens = 1024)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(i, responses)
    model_responses.extend(responses)

final_responses = []

for response in model_responses:
    if 'The answer is' in response:
        response = response.split('The answer is')[1].strip()
        
    if response.endswith('.'):
        response = response[:-1]
    
    final_responses.append(response)
  
model_performance = []

for i, ans in enumerate(final_responses):
    model_score = modified_relaxed_accuracy(questions[i], gold[i], ans)
    print('Response:', ans)
    print('Gold:', gold[i])
    print('Model score:', model_score)
    print()
    model_performance.append(model_score)

print('Model accuracy:', sum(model_performance) / len(model_performance))
print(sum(model_performance), len(model_performance))