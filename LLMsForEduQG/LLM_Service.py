import os

from openai import OpenAI
import requests
from langchain.output_parsers import PydanticOutputParser
import pandas as pd
from pydantic import ValidationError

from MultipleChoiceQuestion import MultipleChoiceQuestion

class LLM_Service:

    supported_models = ["gpt-4o-mini",
                        "meta-llama/Meta-Llama-3.1-70B-Instruct"]

    gpt_client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    hf_api_key = os.environ.get("API_KEY_HUGGINGFACE")

    def execute_prompt(self, model_id, prompt:str):
        response = None
        if model_id.startswith('gpt'):
            response = self.gpt_execute_prompt(model_id,prompt)
        else:  # use model from HF
            response = self.hf_execute_prompt(model_id,prompt)
        return response

    def gpt_execute_prompt(self, model_id="gpt-4o-mini",prompt=""):
        parser = PydanticOutputParser(pydantic_object=MultipleChoiceQuestion)
        format_instructions = parser.get_format_instructions()
        messages = [{"role": "user", "content": prompt + format_instructions}]
        # completion = client.chat.completions.create(
        try:
            completion = self.gpt_client.beta.chat.completions.parse(
                model=model_id,
                messages = messages
            )
            # print(gpt_response.parsed)
            # return gpt_response.parsed
            # parse text to MultipleChoiceQuestion

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


    def hf_execute_prompt(self, model_id, prompt: str):
        # __context = ("You are an expert educational assessment assistant designed to generate education assessment items."
        #            "Please answer in json format with the following attributes: "
        #            "{question: str, correct_answer: str, distractor1: str, distractor2: str, distractor3: str, support: str}.")
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        API_URL = f"https://api-inference.huggingface.co/models/{model_id}"
        parameters = {'return_full_text' : False}
        parser = PydanticOutputParser(pydantic_object=MultipleChoiceQuestion)
        format_instructions = parser.get_format_instructions()

        payload = {"inputs": prompt + format_instructions, "parameters": parameters}
        try:
            response = requests.post(API_URL, headers=headers, json=payload)

        # parse text to MultipleChoiceQuestion:
            if not response.ok:
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
