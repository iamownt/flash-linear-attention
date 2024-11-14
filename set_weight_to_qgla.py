import fla
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig



def load_and_transfer_the_weight():
    model_path1 = 'fla-hub/gla-340M-15B'
    model1 = AutoModelForCausalLM.from_pretrained(model_path1)
    trainable_params, all_param = model1.num_parameters(only_trainable=True), model1.num_parameters()
    print(f"% of trainable params: {trainable_params:d} / {all_param:d} = {trainable_params / all_param:.2%}")

    model_path2 = "training/configs/qgla_340M.json"
    model2 = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_path2))

    state_dict1 = model1.state_dict()
    model2.load_state_dict(state_dict1, strict=False)

    save_path = "/home/user/sngp/fla_models/qgla_340M"
    model2.save_pretrained(save_path)

# load_and_transfer_the_weight()

def text_generation():
    name = 'fla-hub/gla-340M-15B'
    # model_name = name
    model_name = "/home/user/sngp/fla_models/qgla_340M"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    input_prompt = "Hello everyone, I'm Songlin Yang"
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.cuda()
    outputs = model.generate(input_ids, max_length=32)
    print(model)
    print(outputs)
    print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])


text_generation()
