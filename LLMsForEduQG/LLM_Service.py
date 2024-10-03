import os

from openai import OpenAI
import requests
from langchain.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from MultipleChoiceQuestion import MultipleChoiceQuestion

class LLM_Service:

    TIMEOUT = 300 # in seconds

    # key : model
    supported_models = {
        # HuggingFace Llama
        'Llama31405Instruct': 'meta-llama/Meta-Llama-3.1-405B-Instruct',
        'Llama31405': 'meta-llama/Meta-Llama-3.1-405B',
        'Llama3170Instruct': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
        'Llama3170': 'meta-llama/Meta-Llama-3.1-70B',
        'Llama318Instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'Llama318': 'meta-llama/Meta-Llama-3.1-8B',
        'Llama370Instruct': 'meta-llama/Meta-Llama-3-70B-Instruct',
        'Llama370': 'meta-llama/Meta-Llama-3-70B',
        'Llama38Instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'Llama38': 'meta-llama/Meta-Llama-3-8B',
        'Llama27': 'meta-llama/Llama-2-7b',
        'Llama270': 'meta-llama/Llama-2-70b',
        # HuggingFace FlanT5
        'FlanT5Base': 'google/flan-t5-base',
        'FlanT5Large': 'google/flan-t5-large',
        'FlanT5Small': 'google/flan-t5-small',
        'FlanT5XXL': 'google/flan-t5-xxl',
        # HuggingFace MT5
        'MT5Base': 'google/mt5-base',
        'MT5Large': 'google/mt5-large',
        'MT5Small': 'google/mt5-small',
        'MT5XL': 'google/mt5-xl',
        'MT5XXL': 'google/mt5-xxl',
        # HuggingFace Gemma
        'Gemma2BIT': 'google/gemma-2b-it',
        # HuggingFace Mistral
        'Mistral7B03Instruct': 'mistralai/Mistral-7B-Instruct-v0.3',
        'Mistral7B03': 'mistralai/Mistral-7B-v0.3',
        'Mistral7B02Instruct': 'mistralai/Mistral-7B-Instruct-v0.2',
        'MistralSmallInstruct2409': 'mistralai/Mistral-Small-Instruct-2409',
        'MistralLargeInstruct2407': 'mistralai/Mistral-Large-Instruct-2407',
        'MistralNemoInstruct2407': 'mistralai/Mistral-Nemo-Instruct-2407',
        'Mistral7B01': 'mistralai/Mistral-7B-v0.1',
        'Mixtral8x7B01Instruct': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'Mixtral8x22B01Instruct': 'mistralai/Mixtral-8x22B-Instruct-v0.1',
        'Mixtral8x7B01': 'mistralai/Mixtral-8x7B-v0.1',
        'Mixtral8x22B01': 'mistralai/Mixtral-8x22B-v0.1',
        # HuggingFace Falcon
        'Falcon7BInstruct': 'tiiuae/falcon-7b-instruct',
        'Falcon40BInstruct': 'tiiuae/falcon-40b-instruct',
        'FalconMamba7BInstruct': 'tiiuae/falcon-mamba-7b-instruct',
        'FalconMamba7B': 'tiiuae/falcon-mamba-7b',
        # OpenAI's chat models
        'GPT4o': 'gpt-4o',
        'GPT4oMini': 'gpt-4o-mini',
        # 'GPTo1Mini': 'o1-mini', # Tier 5 is required
        'GPT4Turbo': 'gpt-4-turbo',
        'GPT4': 'gpt-4',
        'GPT35Turbo': 'gpt-3.5-turbo',
        'GPT35TurboInstruct': 'gpt-3.5-turbo-instruct',
        # 'GPT35Turbo0613': 'gpt-3.5-turbo-0613',  # snapshot June 13th 2023, deprecated June 13th 2024
        'GPT35Turbo1106': 'gpt-3.5-turbo-1106',  # snapshot November 6th 2023
        'GPT40613': 'gpt-4-0613',  # snapshot June 13th 2023
    } # ["gpt-4o-mini", "meta-llama/Meta-Llama-3.1-70B-Instruct"]

    def print_supported_llms(self):
        print("List of supported LLMs:")
        for llm, url in self.supported_models.items():
            print("{} : {}".format(llm, url))

    cold_models = []
    # ['Llama31405Instruct', 'Llama27', 'Llama270', 'FlanT5Base', 'FlanT5XXL', 'MT5Base', 'MT5Large', 'MT5Small',
    #  'Gemma2BIT', 'Mistral7B02Instruct']
    error_models = []
     # ['Llama31405', 'Llama3170', 'Llama318', 'Llama370', 'Llama38', 'MT5XL', 'MT5XXL', 'Mistral7B03',
     #         'MistralLargeInstruct2407', 'Mixtral8x22B01Instruct', 'Mixtral8x7B01', 'Mixtral8x22B01',
     #         'Falcon40BInstruct', 'FalconMamba7BInstruct', 'FalconMamba7B']
    timeout_models = []

    gpt_client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    hf_api_key = os.environ.get("API_KEY_HUGGINGFACE")

    def get_all_models(self):
        return list(self.supported_models.keys())


    def get_model_url(self,model_id:str):
        for key in self.supported_models.keys():
            if key.upper() == model_id.upper():
                return self.supported_models.get(key)
        return ""

    def get_model_ids_startswith(self, prefix: str):
        model_ids = []
        for key in self.supported_models.keys():
            if key.upper().startswith(prefix.upper()):
                model_ids.append(key)
        return model_ids

    def execute_prompt(self, model_id, prompt:str):
        response = None
        # avoid calling models currently cold/unsupported in HF or OPENAI
        # if model_id not in self.cold_models and model_id not in self.unsupported_models:
            # print("Running Model:{}".format(model_id))
        if model_id == "GPT35TurboInstruct":
            response = self.gpt_old_execute_prompt(model_id, prompt)
        elif model_id.startswith('GPT'):
            response = self.gpt_execute_prompt(model_id,prompt)
        else:  # use model from HF
            response = self.hf_execute_prompt(model_id,prompt)
        # else:
        #     print("Model Skipped:{}".format(model_id))
        return response

    def gpt_execute_prompt(self, model_id="GPT4oMini",prompt=""):
        model_url = self.get_model_url(model_id)
        if model_url == "":
            model_url = self.get_model_url("GPT4oMini")

        parser = PydanticOutputParser(pydantic_object=MultipleChoiceQuestion)
        format_instructions = parser.get_format_instructions()
        try:
            messages = [{"role": "user", "content": prompt + format_instructions}]
            completion = self.gpt_client.chat.completions.create(model=model_url, messages=messages)
            gpt_response = completion.choices[0].message
            if gpt_response.refusal:
                print("gpt_execute_prompt:gpt_response.refusal: ", gpt_response.refusal)
                return None
            else:
                parsed_mc_question = parser.invoke(gpt_response.content)
                return parsed_mc_question
        except ValidationError as err:
            print("gpt_execute_prompt:ValidationError: ", err)
            return None
        except Exception as exc:
            print("gpt_execute_prompt: general exception: ", exc)
            return None

    def gpt_old_execute_prompt(self, model_id="GPT35TurboInstruct",prompt=""):
        model_url = self.get_model_url(model_id)
        if model_url == "":
            model_url = self.get_model_url("GPT35TurboInstruct")

        parser = PydanticOutputParser(pydantic_object=MultipleChoiceQuestion)
        format_instructions = parser.get_format_instructions()
        try:
            messages = prompt + format_instructions
            completion = self.gpt_client.completions.create(model=model_url, prompt=messages)
            gpt_response = completion.choices[0].text

            if 'error' in gpt_response:
                print("gpt_execute_prompt:gpt_response.refusal: ", gpt_response)
                return None
            else:
                parsed_mc_question = parser.invoke(gpt_response)
                return parsed_mc_question
        except ValidationError as err:
            print("gpt_execute_prompt:ValidationError: ", err)
            return None
        except Exception as exc:
            print("gpt_execute_prompt: general exception: ", exc)
            return None

    def hf_execute_prompt(self, model_id, prompt: str):
        model_url = self.get_model_url(model_id)
        if model_url == "":
            model_url = self.get_model_url("Llama3170Instruct")

        headers = {"Authorization": f"Bearer {self.hf_api_key}"} # "x-wait-for-model": "true"
        API_URL = f"https://api-inference.huggingface.co/models/{model_url}"
        parameters = {}
        if (not model_id.startswith("Flan")) and (not model_id.startswith("MT5")):
            parameters['return_full_text'] = False
        parser = PydanticOutputParser(pydantic_object=MultipleChoiceQuestion)
        format_instructions = parser.get_format_instructions()

        payload = {"inputs": prompt + format_instructions, "parameters": parameters}
        try:
            response = requests.post(API_URL, headers=headers, json=payload)

            # if not response.ok and response.status_code == 503: # model is cold. wait until model is loaded
            #     print("Cold Model: {}".format(response.text))
            #     print("Re-trying...")
            #     headers = {"Authorization": f"Bearer {self.hf_api_key}", "x-wait-for-model": "true"}
            #     response = requests.post(API_URL, headers=headers, json=payload, timeout=self.TIMEOUT)

            if not response.ok:
                if response.status_code == 503: # model remains cold
                    self.cold_models.append(model_id)
                elif response.status_code == 504: # model timeout
                    self.timeout_models.append(model_id)
                elif response.status_code == 429: # Rate limit reached.
                    print('hf_execute_prompt: Rate limit reached' + response.text)
                    return "error=429"
                else: # response.status_code == 403 or 400: # error model
                    self.error_models.append(model_id)
                print('hf_execute_prompt: not response.ok' + response.text)
                return None

            generated_text = response.json()[0]['generated_text']
            # if 'error' in generated_text:
            #     print('hf_execute_prompt: ERROR: ' + generated_text['error'])
            #     return None

            parsed_mc_question = parser.invoke(generated_text)
            return parsed_mc_question
        except ValidationError as err:
            print ("hf_execute_prompt:ValidationError: ",err)
            return None
        except Exception as exc:
            print("hf_execute_prompt: general exception: ",exc)
            return None
