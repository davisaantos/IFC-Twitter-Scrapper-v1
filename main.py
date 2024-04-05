# %%
import asyncio
from twscrape import API as twAPI
from pyairtable import Api
import pandas as pd
from datetime import datetime
from pytz import timezone
import dotenv, os
import re
from emoji import demojize
import transformers, torch
from urllib.parse import quote


# %%
# local configs
dotenv.load_dotenv()
WAIT_TIME_MIN = 5

# Airtable 
AIRTABLE_LIMIT = 1_100
airtable = Api(os.getenv('AIRTOKEN'))
base = airtable.base('')
broad_table = base.table('')
config = base.table('')

# Transformer
pretrained_model="belisards/albertina_gun"
tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)
model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=2)

def get_existing_ids(table):
    records = table.all()
    ids = []
    for record in records:
        try:
            ids.append(record['fields']['tw_id'])
        except:
            pass
    return ids

def replace_urls(text):
    url_pattern = re.compile(
        r'(http[s]?://|www\.)'
        r'(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    return url_pattern.sub('URL', text)

def cleantxt(text, lower=False):
    text = str(text)
    
    if lower:
        text = text.lower() # convert to lowercase
    text = replace_urls(text) # replace urls by token
    text = re.sub(r'\s+', ' ', text) # trim extra spaces
    text = re.sub(r'@\w+', 'USR', text) # remove user handles by token
    text = re.sub(r'(.)\1{3,}', r'\1\1\1\1', text) # remove repetition of char
    text = re.sub(r'[^\w\s]', ' ', text) # remove commas and other punctuation

    text = demojize(text)
    
    return text

def predict_sentence(sentence):
    sentence = cleantxt(sentence)
    inputs = tokenizer(sentence, truncation=True, max_length=512, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    probs = probs[0].tolist()
    if probs[0] > probs[1]:
        return 0
    else:
        return 1
    

def explode_dict(df,cols=None):
    # Identify columns containing dictionaries
    if cols:
        dict_columns = cols
    else:
        dict_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, dict)).any()]

    # Explode the identified columns
    for col in dict_columns:
        # Explode the dictionary column
        exploded_col = df[col].apply(pd.Series)
        
        # Add a prefix to the new created columns with the name of the original dictionary column
        exploded_col = exploded_col.add_prefix(f"{col}_")
        
        # Concatenate the exploded column with the original DataFrame, dropping the original dictionary column
        df = pd.concat([df.drop(col, axis=1), exploded_col], axis=1)
        
    return df


def process_tweets(results, message="O​​lá! Poderia informar o local e horário aproximado dos tiros? Pode ser via DM."):
    def generate_reply_url(tweet, text=message):
        base_url = "https://twitter.com/intent/tweet"
        text = quote(text)
        url = f"{base_url}?in_reply_to={tweet['id']}&text={text}"
        return url

    results['label'] = results['rawContent'].apply(lambda x: predict_sentence(x))
    results['reply_url'] = results.apply(generate_reply_url, axis=1)
    df = explode_dict(results)
    df['dia'] = df.date.apply(lambda x: x.date)
    df['hora'] = df.date.apply(lambda x: x.time)

    cols_dict = {
        'user_rawDescription': 'usr_bio',
        'user_location': 'usr_localizacao',
        'id_str': 'tw_id',
        'rawContent': 'texto',
        'link_original': 'url'
    }
    df.rename(columns=cols_dict, inplace=True)
    cols = ['tw_id', 'url', 'texto', 'dia', 'hora', 'reply_url', 'usr_localizacao', 'usr_bio','label']
    df['dia'] = df['dia'].astype(str)
    df['hora'] = df['hora'].astype(str)
    reports = df[cols][df.label == 1].copy()
    reports.drop('label', axis=1, inplace=True)
    print(f"Found {len(reports)} new reports!")
    print(reports.texto)

    return reports

async def collect_and_filter_tweets(myquery,current_ids):
    api = twAPI()
    results = []
    async for tweet in api.search(myquery,limit=100):
        if tweet.id not in current_ids:
            results.append(tweet.dict())
    if len(results) > 0:
        print(f"Found {len(results)} new tweets!")
        return pd.DataFrame(results)
    else:
        return print("No new tweets found!")

def update_backup(airtable,backupfile="backup.csv"):
    airtable = explode_dict(airtable, cols=['fields'])
    if not os.path.isfile(backupfile):
        print('Backup file does not exist. Creating new backup file.')
        airtable.to_csv(backupfile, index=False)
    else:
        print('Backup file exists. Loading backup file.')
        backup = pd.read_csv(backupfile)
        updated_backup = pd.concat([backup, airtable])
        updated_backup.drop_duplicates(subset=['id'], inplace=True) 
        updated_backup.to_csv(backupfile, index=False)
        print('Backup file updated.')

async def main():
    while True:
        try:
            current_ids = get_existing_ids(broad_table)
            current_records = pd.DataFrame(broad_table.all())
            last_id = config.first()['fields']['Último tweet']

            query = config.first()['fields']['Busca geral']
            myquery = f'-from:fogocruzadorj -from:fogocruzadope -from:fogocruzadoba lang:pt since_id:{last_id} {query}'

            if 'Busca extra' in config.first()['fields']:
                query_extra = config.first()['fields']['Busca extra']
                myquery = f'{myquery} OR {query_extra}'
            
            print(f'Buscando por : {myquery}')
        
            results = await collect_and_filter_tweets(myquery, current_ids)
            if results is not None:
                
                df = process_tweets(results, message=config.first()['fields']['Resposta'])
                broad_table.batch_create(df.to_dict(orient='records'))
                
                now_brazil = datetime.now().astimezone(timezone('America/Sao_Paulo'))
                config.update(fields={"Execução": now_brazil.strftime("%Y-%m-%d %H:%M:%S"),
                                    "Último tweet": str(results.id.max())}, record_id="recq3l3FwiSwMv9Pp")

                # Update local backup
                update_backup(current_records)
                
                if len(current_ids) > AIRTABLE_LIMIT:
                    print(f"Current number of records: {len(current_ids)}  is greater than Airtable limit: {AIRTABLE_LIMIT}")
                    # diff = len(current_ids) - AIRTABLE_LIMIT
                    # delete_ids = current_ids[-diff:]
                    # print(f"Deleting {diff} records from Airtable...")
                    # broad_table.batch_delete(delete_ids)                
            else:
                print("Nothing to process!")
            
            print(f"Waiting {WAIT_TIME_MIN} minutes...")

            await asyncio.sleep(WAIT_TIME_MIN*60)
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
     asyncio.run(main())
    ## To run in Jupyter Notebook:
    #     import nest_asyncio
    #     nest_asyncio.apply()
    #     await main()
    