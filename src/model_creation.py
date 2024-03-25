from utils.model_utils import Chat_LLM
from utils.openai_utils import is_openai_model, build_chat_model, build_openai_embeddings_model
from utils.hf_utils import create_hf_pipeline, create_hf_embeddings
import argparse
from dotenv import dotenv_values
from utils.utils import init_argument_parser

#Creates the model based on the model_name parameter
def build_model(opt,env):

    use_openai_api = is_openai_model(opt.model_name)

    #load model 
    model = build_chat_model(opt, env) if use_openai_api else create_hf_pipeline(opt, env)

    return Chat_LLM(model)

#Creates embeddings based on the openai embeddings
def build_embeddings(opt,env):
    
    if is_openai_model(opt.embeddings_name):
        embeddings = build_openai_embeddings_model(opt,env)
        return embeddings
    else:
        embeddings = create_hf_embeddings(opt,env)
        return embeddings


def add_parse_arguments(parser):
    #model parameters
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo-0613', help='name of the model')
    parser.add_argument('--temperature', type=float, default=1.0, help='model temperature')
    parser.add_argument('--embeddings_name', type=str, default='text-embedding-ada-002-v2', help='name of the embeddings')


    return parser

def main():
    opt = init_argument_parser(add_parse_arguments)
    env = dotenv_values()
    model = build_model(opt,env)
    embeddings = build_embeddings(opt,env)
    return model, embeddings

    
if __name__ == '__main__':
    main()




