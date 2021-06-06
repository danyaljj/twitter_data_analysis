from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_t5 import load_tf_weights_in_t5

base_model = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration(T5Config.from_pretrained(base_model))

load_tf_weights_in_t5(model, None, "/Users/danielk/Desktop/drive-download-20210606T034942Z-001/")
model.eval()


def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


# just to verify that it works
run_model("Just like the Jussie hoax, people wake up and realize they’ve been fooled.")
run_model("Great <user> piece on limits of <hashtag> in moving dialogue forward on <hashtag> &amp;")
run_model("The truth. <hashtag> <hashtag>. Rock your <hashtag> everyday.")
run_model("More platitudes!! We need an <hashtag> - <hashtag>")

input_ids = tokenizer.encode("Just like the Jussie hoax, people wake up and realize they’ve been fooled.",
                             return_tensors="pt")
out = model(input_ids=input_ids, labels=input_ids, return_dict=True)
repr = out['encoder_last_hidden_state'].tolist()[0][-1]