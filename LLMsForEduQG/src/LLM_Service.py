import os

from langchain_core.output_parsers import PydanticOutputParser
from src.MultipleChoiceQuestion import MultipleChoiceQuestion
from openai import OpenAI
from pydantic import ValidationError
from config import NEBIUS_BASE_URL, LLM_TIMEOUT


class LLM_Service:
    TIMEOUT = LLM_TIMEOUT  

    # key : model
    supported_models = {
        "NEB_Llama3370Instruct": "meta-llama/Llama-3.3-70B-Instruct",
        "NEB_Qwen3235BInstruct": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "GPT54Mini": "gpt-5.4-mini-2026-03-17"
        #"NEB_DeepSeekV32": "deepseek-ai/DeepSeek-V3.2",
        #"GPT5Mini": "gpt-5-mini-2025-08-07",
        #"GPT54Nano": "gpt-5.4-nano-2026-03-17",
    }

    def print_supported_llms(self):
        print("List of supported LLMs:")
        for llm, url in self.supported_models.items():
            print("{} : {}".format(llm, url))

    cold_models = []
    error_models = []
    timeout_models = []

    gpt_client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    nebius_client = OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=os.environ.get("NEBIUS_API_KEY"),
    )

    def get_all_models(self):
        return list(self.supported_models.keys())

    def get_model_url(self, model_id: str):
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

    def execute_prompt(self, model_id, prompt: str):
        response = None

        if model_id.startswith("GPT"):
            response = self.gpt_execute_prompt(model_id, prompt)
        elif model_id.startswith("NEB_"):
            response = self.nebius_execute_prompt(model_id, prompt)
        else:
            print("Model Skipped:{}".format(model_id))
        return response

    def gpt_execute_prompt(self, model_id="GPT54Mini", prompt=""):
        model_url = self.get_model_url(model_id)
        if model_url == "":
            model_url = self.get_model_url("GPT54Mini")

        parser = PydanticOutputParser(pydantic_object=MultipleChoiceQuestion)
        format_instructions = parser.get_format_instructions()
        try:
            if hasattr(prompt, "messages") and prompt.messages:
                messages = prompt.messages[:]
                messages[-1]["content"] += format_instructions
            else:
                messages = [{"role": "user", "content": prompt.prompt + format_instructions}]
            completion = self.gpt_client.chat.completions.create(
                model=model_url, messages=messages
            )
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

    def nebius_execute_prompt(self, model_id, prompt: str):
        model_url = self.get_model_url(model_id)
        if model_url == "":
            return None

        parser = PydanticOutputParser(pydantic_object=MultipleChoiceQuestion)
        format_instructions = parser.get_format_instructions()
        try:
            if hasattr(prompt, "messages") and prompt.messages:
                messages = prompt.messages[:]
                messages[-1]["content"] += format_instructions
            else:
                messages = [{"role": "user", "content": prompt.prompt + format_instructions}]
            completion = self.nebius_client.chat.completions.create(
                model=model_url, messages=messages
            )
            response = completion.choices[0].message
            if hasattr(response, "refusal") and response.refusal:
                print("nebius_execute_prompt: refusal: ", response.refusal)
                return None

            content = response.content
            # stripping deepseek R1 thinking tokens
            if "<think>" in content:
                content = content.split("</think>")[-1].strip()

            parsed_mc_question = parser.invoke(content)
            return parsed_mc_question
        except ValidationError as err:
            print("nebius_execute_prompt: ValidationError: ", err)
            return None
        except Exception as exc:
            if "429" in str(exc):
                print("nebius_execute_prompt: rate limit reached")
                return "error=429"
            print("nebius_execute_prompt: general exception: ", exc)
            return None
