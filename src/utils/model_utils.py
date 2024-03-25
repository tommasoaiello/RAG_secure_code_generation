
from langchain.prompts import (
    SystemMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import PydanticOutputParser
from utils.synthetic_dataset_utils import XSS_dataset, fill_df, save_subset_of_df
from langchain_core.output_parsers import StrOutputParser
from utils.utils import load_yaml, init_argument_parser, sanitize_output, fill_default_parameters, save_parameters_file, save_input_prompt_file, is_valid_url
from langchain import LLMChain, PromptTemplate
import os, json, requests


#class LLM to manage non chat models
class LargeLanguageModel():
    
    def __init__(self, model) -> None:
        self.model = model
    

    def chain_building(self, template, opt):
        return None
    
    def synthetic_chain_building(self, template, opt):
        return None

    #Given a chain it runs the experiment to generate a .py file
    def run_experiment(self, chain, opt, prompt_parameters, experiment_folder):

        print(f"Running experiment with {opt.model_name}")

        #save the input prompt and the parameters in the experiment folder
        input_prompt = chain.first.format(**prompt_parameters)
        print(input_prompt)
        save_input_prompt_file(os.path.join(experiment_folder, opt.input_prompt_file_name), input_prompt)
        save_parameters_file(os.path.join(experiment_folder, opt.parameters_file_name), opt)
        
        # response = chain.invoke(prompt_parameters)

        #cycle for n experiments
        i = 0
        failures = 0
        while i < opt.experiments:
            print(f"Experiment {i}")
            #try to invoke the chain
            try:
                response = chain.invoke(prompt_parameters)
                save_dir = os.path.join(experiment_folder, f"exp_{i}")
                os.makedirs(save_dir, exist_ok=True)
                #write the generated .py file in the experiment folder
                with open(os.path.join(save_dir, 'generated.py'), 'w') as f:
                    f.write(response)
                i = i + 1
                failures = 0
            except Exception as e:
                print(e)
                print(f"Experiment failed Error:{e}, try again")
                failures = failures + 1
                if failures > 10:
                    print("Too many failures, moving to next experiment")
                    i = i + 1
                    continue
                continue
    
    #Given a chain it creates the synthetic dataset
    def create_synthetic_dataset(self, chain, opt, prompt_parameters, experiment_folder):
        print(f"{opt.model_name} is invoked to create the dataset")

        #save input prompt and parameters file
        input_prompt = chain.first.format(**prompt_parameters)
        print(input_prompt)
        save_input_prompt_file(os.path.join(experiment_folder, opt.input_prompt_file_name), input_prompt)
        save_parameters_file(os.path.join(experiment_folder, opt.parameters_file_name), opt)

        #cycle for n experiments
        i = 0
        failures = 0
        while i < opt.experiments:
            print(f"Experiment {i}")
            try:
                df= fill_df(chain, prompt_parameters)
                save_file = os.path.join(experiment_folder, f"exp_{i}.csv")
                df.to_csv(save_file, index=False)
                for s in opt.subset:
                    save_subset_of_df(save_file, s)
                i = i + 1
                failures = 0
            except Exception as e:
                print(e)
                print("Experiment failed, try again")
                failures = failures + 1
                if failures > 10:
                    print("Too many failures, moving to next experiment")
                    i = i + 1
                    continue
                continue
     
class Chat_LLM(LargeLanguageModel):

    def __init__(self, model)->None:
        super().__init__(model)

    #Builds a chain for the experiment
    def chain_building(self, template, opt):
        print(f"Building Model {opt.model_name}")
        prompt = ChatPromptTemplate.from_messages([("system", template["input"]),
                                                    ("human", "{input}")])

        chain = prompt | self.model | StrOutputParser() | sanitize_output
        return chain
    
    #Builds a chain for the synthetic dataset
    def synthetic_chain_building(self,template,opt):
        print(f"Creating the prompt for synthetic dataset creation using {opt.model_name}")
        output_parser = PydanticOutputParser(pydantic_object=XSS_dataset)

        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(template['input']+ "\n{format_instructions}"),
                HumanMessagePromptTemplate.from_template("{input}")  
            ],
            input_variables=["input"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )

        chain = prompt | self.model | output_parser
        return chain
    


     