import argparse
import os.path

from LLMsForEduQG import LLMsForEduQG
from Prompt import Prompt, PromptID
import uuid

def list_of_strings(arg):
    return arg.split(',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='LLMsForEduQG',
        description='Testing framework to assess the performance of LLMs in generating multiple-choice education questions.',
        epilog='')
    parser.add_argument("-i", "--input", dest="input_filename", default="",
                        help="Input .csv file with the ground truth questions.", metavar="INPUT")

    parser.add_argument("-d", "--dir", "-o", "--out",  dest="results_dir",
                        default="results/{}".format(str(uuid.uuid4())), help="Output directory.", metavar="DIR")

    parser.add_argument("-N", "-n", "--max", dest="MAX", type=int,
                        default=0, help="Maximum number of questions analysed from the dataset.", metavar="MAX")

    parser.add_argument("-r", "--random", dest="random_choice", type=bool,
                        default=False, help="Random order of analysed questions.", metavar="RANDOM")

    parser.add_argument("-m", "--models", type=list_of_strings, dest="models_list",
                        default="", help="List the LLMs to run.", metavar="MODELS")

    parser.add_argument("-sw", '--starts-with', type=str, dest="models_prefix",
                        default=None, help="Selects all LLMs starting with the <prefix>.", metavar="PREFIX")

    parser.add_argument("-p", "--prompts", type=list_of_strings, dest="prompts_list",
                        default="", help="List the prompts to use.", metavar="PROMPTS")

    parser.add_argument('-ll', '--llms', "--llm-list",dest="list_llms",
                        action='store_true', help="List the supported LLMs.")

    parser.add_argument('-pl', "--prompt-list", dest="list_prompts",
                        action='store_true', help="List the available prompts.")


    args = parser.parse_args()

    print(args)
    if args.input_filename is None or args.input_filename == "":
        raise parser.error("input filename is mandatory.")
    elif not os.path.exists(args.input_filename):
        raise parser.error("Input filename does not exist.\n")
    else:
        #generate LLMrunner
        LLMrunner = LLMsForEduQG(args.input_filename, args.results_dir, args.MAX, args.random_choice)

        if args.list_llms:
            LLMrunner.llm_service.print_supported_llms()
        if args.list_prompts:
            PromptID.print_supported_prompts()

        # include only supported models
        models = []
        if args.models_prefix is not None:
            models = LLMrunner.llm_service.get_model_ids_startswith(args.models_prefix)
            if len(models) == 0:
                raise parser.error("Invalid models prefix.")
        else:
            if args.models_list is None or args.models_list == "" or args.models_list == [] or args.models_list == [""]:
                models = LLMrunner.llm_service.get_all_models()
            else:
                for m in LLMrunner.llm_service.get_all_models():
                     if m in args.models_list:
                         models.append(m)
            if len(models) == 0:
                raise parser.error("No model selected.")

        # include only supported prompts
        prompt_IDs = []
        if args.prompts_list is None or args.prompts_list == "" or args.prompts_list == [] or args.prompts_list == [""]:
            prompt_IDs = PromptID.all()
        else:
            for p in args.prompts_list:
                for p1 in PromptID.all():
                    if p == p1.name or "PromptID."+p == p1.name:
                        prompt_IDs.append(p1)

        if len(prompt_IDs) == 0:
            raise parser.error("No prompt selected.")

        LLMrunner.run_per_qid(prompt_ids=prompt_IDs, model_ids=models)

    # MAX = 5
    # random_choice = False
    # SciQdataset = r"D:\PycharmProjects\LLMsForEduQG\datasets\SciQ_test.csv"
    # results_dir = r"D:\PycharmProjects\LLMsForEduQG\results\\"+str(MAX)
    # # outname_without_answers = r"D:\PycharmProjects\EduQGdataAnalyzer\results\SciQ_test-generated-without-answers-"+str(MAX)+".csv"
    #
    # LLMrunner = LLMsForEduQG(SciQdataset,results_dir,MAX,random_choice)
    # # prompt_IDs = PromptID.all()
    # prompt_IDs = [PromptID.Simple_plus_Answer]
    # # models = ["meta-llama/Meta-Llama-3.1-70B-Instruct"]
    # # models = ["FlanT5Large"]
    # # models = LLMrunner.llm_service.get_all_models()
    # models = LLMrunner.llm_service.get_model_ids_startswith("Flan")
    # LLMrunner.run_per_qid(prompt_IDs,models)