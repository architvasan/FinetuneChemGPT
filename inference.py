import torch
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast#, TFGPT2LMHeadModel, AutoConfig
from mpi4py import MPI

#tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json", bos_token='[BOS]', eos_token='[END]')
#tokenizer.add_special_tokens({'pad_token': 'eos_token_id'})


from transformers import AutoModelForCausalLM

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

model = AutoModelForCausalLM.from_pretrained("fine_tune_chemgpt_1p7m/checkpoint-18500")
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-4.7M")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

FIL = f"SMILES_Generate.{rank}.dat"

prompt = "[C]"
inputs = tokenizer(prompt, return_tensors="pt").to(device)#.input_ids

#if True:
#    from transformers import set_seed
#    set_seed(42)
#    
#    sample_output = model.generate(
#                        **inputs,




if True:
    for i in range(100):
        outputs = model.generate(**inputs,
                                    max_new_tokens=40,
                                    num_beams=1000,
                                    #num_beam_groups=5,
                                    temperature=0.5,
                                    num_return_sequences=100,
                                    no_repeat_ngram_size=10,
                                    remove_invalid_values=True
                                    )
    
                                    #max_new_tokens=44,
                                    #diversity_penalty=1.0,
                                    #do_sample=True,
                                    #temperature=0.5) #max_new_tokens=45, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=500, temperature=1)
                                   # max_length = 45,
                                   # min_length=10,
                                   # do_sample=True,
                                   # top_k=50,
                                   # top_p=0.95,
                                   # num_return_sequences=2,
                                   # remove_invalid_values=True,
                                   # #pad_token_id = tokenizer.encode("[PAD]")[0],
                                   # temperature =1.0,
                                   # )
                                    #num_beams=5, num_beam_groups=5, max_new_tokens=44, diversity_penalty=1.0, do_sample=True, temperature=0.5) #max_new_tokens=45, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=500, temperature=1)
        for o in outputs:
            tokenized_out = tokenizer.decode(o, skip_special_tokens=True)
            with open(FIL, 'a') as f:
                    f.write(f"{tokenized_out}\n")

#for v in vocab_list[16:]:#range(0,100000):
#    prompt = v
#    inputs = tokenizer(prompt, return_tensors="pt").input_ids
#    for i in range(100):
#        outputs = model.generate(inputs, max_new_tokens=45, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=500, temperature=1)
#        tokenized_out = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#        with open(FIL, 'a') as f:
#            for line in tokenized_out:
#                f.write(f"{line}\n")
#
        # num_beams=1000, do_sample=True, num_return_sequences=1000, max_new_tokens=45)#, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=51)
        #print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

#outfile.close()

#generator = pipeline("text-generation", model = "my_model/checkpoint-9500")
#print(generator(prompt))
