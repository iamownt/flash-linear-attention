import fla
from transformers import AutoModelForCausalLM, AutoTokenizer
name = 'fla-hub/gla-340M-15B' # 341707776
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).cuda()
input_prompt = "Power goes with permanence. Impermanence is impotence. And rotation is castration."
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids, max_length=64)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
trainable_params, all_param = model.num_parameters(only_trainable=True), model.num_parameters()
print(f"% of trainable params: {trainable_params:d} / {all_param:d} = {trainable_params / all_param:.2%}")
