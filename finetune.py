from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import pandas as pd
from datasets import load_dataset 
import torch
import numpy as np
from torch.utils.data import DataLoader
#from genslm import GenSLM, SequenceDataset
from tqdm import tqdm
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments

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
    #dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataset


tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-1.2B")
model = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-1.2B")

smiles_origin_path_train = 'origin_train_placeholder'
smiles_origin_path_vali = 'origin_vali_placeholder'
smiles_new_path = 'new_placeholder'

smiles_original_train = pd.read_parquet(smiles_origin_path_train)['smiles'].sample(frac=0.01, random_state=1)
smiles_original_vali = pd.read_parquet(smiles_origin_path_vali)['smiles'].sample(frac=0.01, random_state=2)
smiles_new = pd.read_csv(smiles_new_path)['smiles']
smiles_new_train, smiles_new_vali = train_test_split(smiles_new, test_size=0.1)

smiles_data_train = pd.concat([smiles_original_train, smiles_new_train]).sample(frac = 1)
smiles_data_vali = pd.concat([smiles_original_vali, smiles_new_vali]).sample(frac = 1)

Dataset_train = preprocess(smiles_data_train)
Dataset_vali = preprocess(smiles_data_vali)


training_args = TrainingArguments(output_dir="test_trainer")


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=Dataset_train,
    eval_dataset=Dataset_vali,
    compute_metrics=compute_metrics,
)

trainer.train()