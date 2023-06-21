# export DEBUG_PORT=5679
# REMOTE_DEBUG=1 python -m tests.hf.smart_special_tokens smart_tokenizer_and_embedding_resize --model_path './logs/chatflow_7b_stage1_v1.1.2'
import transformers
from tests import logger
from .test_model import get_tokenizer, load_model


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict:dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def test_smart_tokenizer_and_embedding_resize(model_name=None, model_path=None):
    tokenizer = get_tokenizer(model_name=model_name, model_path=model_path)
    model = load_model(model_name=model_name, model_path=model_path)
    special_tokens_dict = {
        'eob_token': '<|EndOfBlock|>'
    }
    smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model)
    logger.info('test_smart_tokenizer_and_embedding_resize done')


if __name__ == "__main__":
    _d, component = dict(globals()).items(), {}
    for k, v in _d:
        if k.startswith('test_'):
            component[k] = v
            component[k[5:]] = v

    import fire
    fire.Fire(component)