from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    HammingDiversityLogitsProcessor,
    BeamSearchScorer,
    StoppingCriteriaList,
    MaxLengthCriteria,
    AdamW,
    get_scheduler
)
from transformers import DataCollatorForLanguageModeling

import selfies as sf
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import pandas as pd
from datasets import load_dataset, Dataset 
import torch
import numpy as np
from torch.utils.data import DataLoader
#from genslm import GenSLM, SequenceDataset
from tqdm import tqdm
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments

def _tokenize_function(smi):
    return tokenizer(smi, padding="max_length", truncation=True)

def tokenize_function(data):
    token_data = tokenizer(data['smiles'], padding="max_length", max_length=45, truncation=True)#, return_tensors="pt")
    return token_data

def convert_selfies(smi):
    try:
        self = sf.encoder(smi['smiles'])
    except:
        self = 0
    return 0

def preprocess(input_data, tokenizer):
    max_seq_len = 45
    #batch_size = 256
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenized_dataset = tokenizer(                                  
        #[group_by_kmer(seq, kmer_length) for seq in sequences],
        input_data,
        padding=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors="pt",
    )
    dataset = torch.utils.data.TensorDataset(
        tokenized_dataset["input_ids"], tokenized_dataset["attention_mask"]
    )
    dataloader = DataLoader(dataset, batch_size=32)
    return dataloader#tokenized_dataset#dataset


tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-1.2B")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
model = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-1.2B")
optimizer = AdamW(model.parameters(), lr = 2e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

smiles_origin_path_train = 'data/train-00000-of-00001-e9b227f8c7259c8b.parquet'
smiles_origin_path_vali = 'data/validation-00000-of-00001-9368b7243ba1bff8.parquet'
smiles_new_path = 'data/Top_hits.csv'

smiles_original_train = pd.read_parquet(smiles_origin_path_train, columns=['smiles']).sample(frac=0.01, random_state=1)
smiles_original_vali = pd.read_parquet(smiles_origin_path_vali, columns=['smiles']).sample(frac=0.01, random_state=2)
#smiles_original_vali = pd.read_parquet(smiles_origin_path_vali)['smiles'].sample(frac=0.01, random_state=2)
smiles_new = pd.read_csv(smiles_new_path, usecols= ['smiles'])#['smiles']
smiles_new_train, smiles_new_vali = train_test_split(smiles_new, test_size=0.1)

smiles_data_train = pd.concat([smiles_original_train, smiles_new_train]).sample(frac = 1)
#print(smiles_data_train)
smiles_data_vali = pd.concat([smiles_original_vali, smiles_new_vali]).sample(frac = 1)

selfies_data_train = smiles_data_train.apply(lambda x: convert_selfies(x)) 
print(selfies_data_train)
selfies_data_vali = smiles_data_vali.apply(lambda x: convert_selfies(x)) 
print(selfies_data_train)


dataset_train = Dataset.from_pandas(smiles_data_train)
dataset_vali = Dataset.from_pandas(smiles_data_vali)
dataset_train_tok = dataset_train.map(tokenize_function, batched=True).shuffle(seed=42)
dataset_vali_tok = dataset_vali.map(tokenize_function, batched = True).shuffle(seed=42)
print(dataset_train_tok['input_ids'][:2])
#smiles_data_train_seqs = list(smiles_data_train.to_numpy())
#smiles_data_vali_seqs = list(smiles_data_vali.to_numpy())

#selfies_train =smiles_data_train#.apply(lambda x: sf.encoder(x))# [sf.encoder(smi) for smi in smiles_data_train_seqs]
#selfies_vali = smiles_data_vali#.apply(lambda x: sf.encoder(x))#[sf.encoder(smi) for smi in smiles_data_vali_seqs]
#
#Dataset_train = preprocess(selfies_train, tokenizer)
#Dataset_vali = preprocess(selfies_vali, tokenizer)

#training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
training_args = TrainingArguments(
    output_dir="my_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs = 6.0,
    #push_to_hub=True,
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train_tok,
    eval_dataset=dataset_vali_tok,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
