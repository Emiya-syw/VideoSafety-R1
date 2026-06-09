from collections.abc import Iterable

from videollama3.constants import TEXTUAL_ALARM_TOKEN, VISUAL_ALARM_TOKEN


def configure_alarm_tokens(tokenizer, models: Iterable = ()):
    """Register dedicated alarm placeholders and synchronize model configs.

    The tokenizer IDs are resolved at runtime instead of being hard-coded. The
    placeholders remain in ``input_ids`` so their positions survive tokenization;
    the model replaces their ordinary token embeddings with the trainable visual
    and textual alarm embeddings before entering the LLM.
    """
    tokenizer.add_tokens(
        [VISUAL_ALARM_TOKEN, TEXTUAL_ALARM_TOKEN],
        special_tokens=True,
    )

    visual_alarm_token_id = tokenizer.convert_tokens_to_ids(VISUAL_ALARM_TOKEN)
    textual_alarm_token_id = tokenizer.convert_tokens_to_ids(TEXTUAL_ALARM_TOKEN)

    for model in models:
        if model is None:
            continue
        # Adding special tokens changes the tokenizer vocabulary. Resize every
        # policy/reference model before storing the resolved IDs in its config.
        if model.get_input_embeddings().num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        model.config.visual_alarm_token_id = visual_alarm_token_id
        model.config.textual_alarm_token_id = textual_alarm_token_id

    return visual_alarm_token_id, textual_alarm_token_id
