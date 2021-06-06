from transformers import T5Config, T5Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_t5 import load_tf_weights_in_t5, T5ForConditionalGeneration
from torch.nn import CrossEntropyLoss


def new_forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        labels=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
):
    r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

    Returns:

    Examples::

        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)
        >>> input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        >>> outputs = model(input_ids=input_ids, labels=input_ids)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
        >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)
        >>> input_ids = tokenizer.encode("summarize: Hello, my dog is cute", return_tensors="pt")  # Batch size 1
        >>> outputs = model.generate(input_ids)
    """

    # if "lm_labels" in kwargs:
    #     warnings.warn(
    #         "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
    #         FutureWarning,
    #     )
    #     labels = kwargs.pop("lm_labels")
    # if "decoder_past_key_value_states" in kwargs:
    #     warnings.warn(
    #         "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
    #         FutureWarning,
    #     )
    #     past_key_values = kwargs.pop("decoder_past_key_value_states")
    # if "decoder_past_key_values" in kwargs:
    #     warnings.warn(
    #         "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
    #         FutureWarning,
    #     )
    #     past_key_values = kwargs.pop("decoder_past_key_values")
    assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
    #     encoder_outputs = BaseModelOutput(
    #         last_hidden_state=encoder_outputs[0],
    #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
    #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
    #     )

    hidden_states = encoder_outputs[0]

    if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(labels)

    # If decoding with past key value states, only the last tokens
    # should be given as an input
    if past_key_values is not None:
        assert labels is None, "Decoder should not use cached key value states when training."
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        if decoder_inputs_embeds is not None:
            decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_value_states=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]
    # Rescale output before projecting on vocab
    # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
    sequence_output = sequence_output * (self.model_dim ** -0.5)
    lm_logits = self.lm_head(sequence_output)

    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

    if not return_dict:
        output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        return ((loss,) + output) if loss is not None else output

    out = Seq2SeqLMOutput(
        loss=loss,
        logits=lm_logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )
    out.sequence_output = sequence_output
    return out


# monkey patch
T5ForConditionalGeneration.forward = new_forward

base_model = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration(T5Config.from_pretrained(base_model))

load_tf_weights_in_t5(model, None, "/Users/danielk/Desktop/drive-download-20210606T034942Z-001/")
model.eval()


def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    print(tokenizer.batch_decode(res, skip_special_tokens=True))


# just to verify that it works
run_model("Just like the Jussie hoax, people wake up and realize they’ve been fooled.")
run_model("Great <user> piece on limits of <hashtag> in moving dialogue forward on <hashtag> &amp;")

dummy_label_id = tokenizer.encode(".", return_tensors="pt")
# input_ids = tokenizer.encode("Just like the Jussie hoax, people wake up and realize they’ve been fooled.", return_tensors="pt")
# out = model(input_ids=input_ids, labels=dummy_label_id, return_dict=True)

X = []
y = []
with open('/Users/danielk/Desktop/drive-download-20210606T175138Z-001/train.tsv') as f:
    for idx, line in enumerate(f.readlines()):
        if idx > 500:
            break
        line_split = line.replace("\n", "").split("\t")
        input_ids = tokenizer.encode(line_split[1], return_tensors="pt")
        out = model(input_ids=input_ids, labels=dummy_label_id, return_dict=True)
        X.append(out.sequence_output.tolist()[0][0])
        y.append(line_split[2])

print(len(X))
print(len(y))


from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(X)

df = pd.DataFrame()
df["y"] = y
num_labels = len(set(y))
df["comp-1"] = z[:, 0]
df["comp-2"] = z[:, 1]

print(f" * num_labels: {num_labels}")
sns.scatterplot(
    x="comp-1", y="comp-2",
    hue=df.y.tolist(),
    palette=sns.color_palette("hls", num_labels),
    data=df
).set(title="Twitter data T-SNE projection")

plt.legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)

plt.show()
