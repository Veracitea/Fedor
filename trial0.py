# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
#pipe = pipeline("text-generation", model="tiiuae/falcon-180B")

#xgen-7b-8k-base Salesforce shafiq_joty
tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen-7b-8k-base")
model = AutoModelForCausalLM.from_pretrained("Salesforce/xgen-7b-8k-base")

#gemini google
#https://www.datacamp.com/tutorial/introducing-gemini-api

#llama2
#purple llama

#bloom bigscience
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom")

#mpt-7b-8k mosaicml
tokenizer = AutoTokenizer.from_pretrained("mosaicml/mpt-7b-8k", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("mosaicml/mpt-7b-8k", trust_remote_code=True)

#falcon 180B
#https://falconllm.tii.ae/falcon-models.html
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-180B")
model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-180B")

#vicuna 13b
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.3")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-13b-v1.3")

generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
generator("Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone")
[{'generated_text': 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Dragon-lords (for them to rule in a world ruled by their rulers, and all who live within the realm'}]
