from multiprocessing import process
from os import truncate
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, GenerationConfig
from models.model_base import PreTrainedModelWrapper
# from tlr import DPOTrainer

class AutoDPOModelForCausalLM(PreTrainedModelWrapper):
    """
    An autoregressive model with support for custom modules in addition to the language model.
    This class inherits from `PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained`, and `generate`. To call a method of the wrapped
    model, simply manipulate the `pretrained_model` attribute of this class.

    Class attributes:
        - **transformers_parent_class** (`transformers.PreTrainedModel`) -- The parent class of the wrapped model. This
            should be set to `transformers.AutoModelForCausalLM` for this class.
        - **lm_head_namings** (`tuple`) -- A tuple of strings that are used to identify the language model head of the
            wrapped model. This is set to `("lm_head", "embed_out")` for this class but can be changed for other models
            in the future
        - **supported_args** (`tuple`) -- A tuple of strings that are used to identify the arguments that are supported
            by the custom module class you designed. Currently, the supported args are: ______
    """

    transformers_parent_class = AutoModelForCausalLM
    lm_head_namings = ["lm_head", "embed_out"]

    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ()
    ####################################################################################

    def __init__(self, pretrained_model, **kwargs):
        r"""
        Initializes the model.

        Args:
            pretrained_model (`transformers.PreTrainedModel`):
                The model to wrap. It should be a causal language model such as GPT2.
                or any model mapped inside the `AutoModelForCausalLM` class.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to any `CustomModule` class.
        """
        super().__init__(pretrained_model, **kwargs)

        print("MODEL INIT")

        if not any(hasattr(self.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure CustomModule is repalced with the name of your custom module class
        # Remember that the below lines are just an example
        # You can reanme the class and the variabels to fit your custom module name,
        # just make sure they are consistent in the code
        # =========================================================================================
        # custom_module_kwargs, _, _ = self._split_kwargs(kwargs)
        # self.custom_module = CustomModule(self.pretrained_model.config, **custom_module_kwargs)
        # self._init_weights(**custom_module_kwargs)
        ###########################################################################################

    def _init_weights(self, **kwargs):
        """
        Initializes the weights of the custom module. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `custom_module_init_strategy`
        argument when calling `.from_pretrained`. Supported strategies are:
            - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `CustomModule` class.
        """
        ###############################################################
        # TODO (Optional): Please implement the initialization strategy for your custom module here
        pass
        ###############################################################

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the custom module
        to the state dictionary of the wrapped model by prepending the key with `custom_module.`.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure "custom_module" is repalced with the name of your custom module class
        # =========================================================================================
        # custom_module_state_dict = self.custom_module.state_dict(*args, **kwargs)
        # for k, v in custom_module_state_dict.items():
        #     pretrained_model_state_dict[f"custom_module.{k}"] = v
        ###########################################################################################
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        """
        We add the state dictionary of the custom module to the state dictionary of the wrapped model
        by prepending the key with `custom_module.`. This function removes the `custom_module.` prefix from the
        keys of the custom module state dictionary.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not hasattr(self, 'custom_module'):
            return

        for k in list(state_dict.keys()):
            if "custom_module." in k:
                state_dict[k.replace("custom_module.", "")] = state_dict.pop(k)
        self.custom_module.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for CustomModule models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put custom_module on the same device as the lm_head to avoid issues
            self.custom_module = self.custom_module.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def push_to_hub(self, *args, **kwargs):
        """Push the model to the Hugging Face hub."""
        ###########################################################################################
        # TODO (Optional): Please uncomment the following line to add the custom module to the hub model
        # Make sure custom_module is repalced with the name of your custom module class
        # =========================================================================================
        # self.pretrained_model.custom_module = self.custom_module
        ###########################################################################################

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        """
        Applies a forward pass to the wrapped model and returns the output from the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        Returns:
            output_dict (`dict`): A dictionary containing the output from the model.
        """
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        output_dict = {}

        ###############################################################
        # TODO: Please implement your customized forward pass here
        # =============================================================
        output_dict = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs)

        ###############################################################

        return output_dict

    def get_logprobs(self, batch, tokenizer):
        """
        Computes the log probabilities of a response using the model respectively.

        Args:
            batch (`dict` of `list`): A dictionary containing the input data for the DPO model.
                The data format is as follows:
                {
                    "prompt": List[str],
                    "chosen": List[str],
                    "rejected": List[str],
                    "chosen_logps": Optional(torch.FloatTensor)
                    "rejected_logps": Optional(torch.FloatTensor)
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input data.
        Returns:
            A tuple of two tensors: (chosen_logps, rejected_logps)
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the chosen responses. Shape: (batch_size,)
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the rejected responses. Shape: (batch_size,)
        """

        def get_logprobs_pair(questions: list, answers: list):
            """
            Compute the log probabilities of the answers given the questions.

            Args:
                questions (`list`): A list of questions.
                answers (`list`): A list of answers.
            Returns:
                logps (`torch.FloatTensor`): Log probabilities of the answers given the questions.
            """
            system  = {"role": "system", "content": "You are a helpful EPFL chatbot."}

            self.pretrained_model.to(torch.device("cuda"))

            strings = [tokenizer.apply_chat_template([system, {"role": "user", "content": question}], tokenize=False, add_generation_prompt=True) for question in questions]
            # Get the length of the tokenized prompt
            length_questions = [len(tokenizer.tokenize(string)) for string in strings]

            # Add the answers
            strings = [string + answer for string, answer in zip(strings, answers)]

            #TODO: pad to max_length of the model
            tokenized_strings = tokenizer(strings, return_tensors="pt", padding=True, truncation=True, max_length=2048, add_special_tokens=True)

            input_ids = tokenized_strings.input_ids
            attention_mask = tokenized_strings.attention_mask

            input_ids = input_ids.to(self.pretrained_model.device)
            attention_mask = attention_mask.to(self.pretrained_model.device)

            #Make sure there are no memory errors
            max_batch_size = 1
            logits = []
            # start = time.time()
            for i in range(0, len(input_ids), max_batch_size):
                end_idx = min(len(input_ids), i + max_batch_size)
                with torch.no_grad():
                        logits.append(self.pretrained_model(input_ids[i : end_idx], use_cache=False, attention_mask=attention_mask[i: end_idx]).logits)

            # end = time.time() 
            # print("Time taken to run model inference: ", end - start)

            if len(logits) >  1:
                logits = torch.cat(logits, dim=0)
            else:
                logits = logits[0]

            logps = []
            for i, logit in enumerate(logits):

                # get the input tokens and logits of the answer (not the whole prompt + answer)
                needed_input = input_ids[i][length_questions[i]:]
                needed_logits = logit[length_questions[i]:]

                #calculate a loss mask to use for the log probs
                loss_mask = needed_input != tokenizer.pad_token_id

                #calculate log softmax in the dimension of vocab_size
                needed_logits = F.log_softmax(needed_logits, dim=1)

                #use the token_ids to choose the log prob to choose from all of the logits
                selected_logits = torch.gather(needed_logits, dim=1, index=needed_input.unsqueeze(1)).squeeze(1)

                #sum the log probs
                sum_logits_answer = (selected_logits * loss_mask).sum().item()

                logps.append(sum_logits_answer) 
            
            return torch.tensor(logps)
            

        assert len(batch["prompt"]) == len(batch["chosen"]) == len(batch["rejected"]) 

        chosen_logps = get_logprobs_pair(batch["prompt"], batch["chosen"])
        rejected_logps = get_logprobs_pair(batch["prompt"], batch["rejected"])

        return chosen_logps, rejected_logps

    def prediction_step_reward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Computes the reward socres of the chosen and reject responses by implementing the DPO reward function
        Reference of the DPO reward function: https://arxiv.org/pdf/2305.18290.pdf

        Args:
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            output_dict (`dict`):
                A dictionary containing the reward scores of the chosen and rejected responses.
        """
        output_dict = {
            "chosen_rewards": [],
            "rejected_rewards": []
        }

        ########################################################################
        # TODO: Please implement the prediction step that computes the rewards
        # ======================================================================
        # You need to return one reward score for each chosen and rejected response.
        # ======================================================================

        # See Notion on how to compute the rewards
        # define beta
        #TODO: are we supposed to use the same beta we used for training or can we consider beta = 1 and just remove it 
        # (we only use this value to compare it with the reference model, so beta does not make a difference)
        # valid because we will use DPOTrainer and not this function to train the model
        beta = 1
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        output_dict["chosen_rewards"] = chosen_rewards.tolist()
        output_dict["rejected_rewards"] = rejected_rewards.tolist()
        
        ########################################################################

        return output_dict

    def prediction_step_mcqa(self, batch, tokenizer):
            """
            Computes the mcqa prediction of the given question.

            Args:
                batch (`dict` of `list`):
                    A dictionary containing the input mcqa data for the DPO model.
                    The data format is as follows:
                    {
                        "question": List[str], each <str> contains the question body and the choices
                        "answer": List[str], each <str> is a single letter representing the correct answer
                    }
                tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input questions.
            Returns:
                output_dict (`dict`): A dictionary containing the model predictions given input questions.
            """
            ########################################################################
            # TODO: Please implement the prediction step that generates the prediction of the given MCQA question
            # ======================================================================
            # You need to return one letter prediction for each question.
            # ======================================================================
            #TODO: make sure the template of the questions the teachers are going to provide is EXACTLY THE SAME as the one we are using here 
            # I am guiding myself by what our training datasets have, adapt  otherwise
            ########################################################################
            QUESTION_TEMPLATE = """### QUESTION
            <question>

            ###OPTIONS
            <options>"""

            #############################################################################################################################################

            def get_mcq_options(samples):
                """
                Returns a dataset in the format:
                {
                    "question": [str],
                    "options": [list[str]],
                    "correct_option": [str]
                }
                """
                questions_and_options = samples["question"]
                remove_prefix = len("Question: ")
                remove_suffix = len("\n\nAnswer")

                #remove the preffix and suffix from questions and options
                questions_and_options = [question_and_options[remove_prefix:-remove_suffix] for question_and_options in questions_and_options]

                #separate into questions and options
                questions_and_options = [question_and_options.split("\n\nOptions:\n") for question_and_options in questions_and_options]

                questions, options = zip(*questions_and_options)
                questions = list(questions)

                #split options into a list
                options = list(options)
                options = [opts.split("\n") for opts in options]
                options = [[opt for opt in opts if opt != "" and opt != " "] for opts in options]

                #remove the option identifier
                to_remove = ["A)", "B)", "C)", "D)", "E)", "F)", "A. ", "B. ", "C. ", "D. ", "E. ", "F. "]

                for remove in to_remove:
                    options = [[opt.replace(remove, "") if opt.startswith(remove) else opt for opt in opts] for opts in options]

                return {"question": questions, "options": options, "correct_option": samples["answer"]}
            
            #############################################################################################################################################

            def apply_template(samples):
                system = {"role": "system", "content": "You are a helpful EPFL chatbot."}
                messages = []

                #apply question template
                for i, question in enumerate(samples["question"]):
                    prompt = QUESTION_TEMPLATE.replace("<question>", question)
                    options=""
                    for j, opt in enumerate(samples["options"][i]):
                        options += f"{chr(ord('A') + j)}: {opt}\n"

                    prompt = prompt.replace("<options>", options)

                    conversation = [system]
                    conversation.append({"role": "user", "content": prompt})
                    messages.append(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))

                return messages
            #############################################################################################################################################

            def get_answers(input_ids, answers, processed_batch_opts, force_answer=False):
                """
                Extracts the answer from the output of the model.
                If the output is mal-formed, either returns the input_ids and index of the mal-formed output if force_answer is False, or "A" otherwise
                """
                bad_responses_input_ids = []
                results = []
                print(f"{answers[0] = }")
                for i, answer in answers:
                    trigger = "### ANSWER\n" 
                    start_idx = answer.find(trigger)


                    #if the template for the correct option was found, get the character after that
                    if start_idx != -1:
                        option = answer[start_idx + len(trigger): start_idx + len(trigger) + 1] 

                        #check if the model returned a lower case option
                        option = option.upper() if option.islower() else option

                        #check if option is among the possible options
                        length_options = len(processed_batch_opts[i])
                        option_idx = ord(option) - ord("A")

                        if option_idx < length_options:
                            results.append((i, option))
                        else:
                            print("Option not among the possible options!")
                            if not force_answer:
                                bad_responses_input_ids.append((i, input_ids[i]))
                            else:
                                results.append((i, "A"))
                
                    else:
                        print("Exception occured!")
                        if not force_answer:
                            bad_responses_input_ids.append((i, input_ids[i]))
                        else:
                            results.append((i, "A"))
                
                if not force_answer:
                    return results, bad_responses_input_ids

                else:
                    return results
            #############################################################################################################################################

            def pad2dtensors(tensors, pad_token_id):
                """
               Pads a list of tensors to the same length (the maximum length)
                """
                max_len = max([tensor.size(1) for tensor in tensors])
                padded_tensors = []
                for tensor in tensors:
                    if tensor.size(1) == max_len:
                        padded_tensors.append(tensor)
                        continue
                    pad_tensor = torch.tensor([pad_token_id] * (max_len - tensor.size(1))).unsqueeze(0).expand(tensor.shape[0], -1).to(tensor.device)
                    padded_tensor = torch.cat([tensor, pad_tensor], dim=1)
                    padded_tensors.append(padded_tensor)
                return padded_tensors
            #############################################################################################################################################

            processed_batch = get_mcq_options(batch)
            processed_batch_opts = processed_batch["options"]
            
            prompts = apply_template(processed_batch)

            print(f"{prompts = }")

            #When running, I get the following warning, soemthing along the lines of: Please set the padding side to left for correct generation as this is a decoder-only model
            tokenizer.padding_side = "left"
            all_input_ids = tokenizer(prompts, padding=True, truncation=True, max_length = 1024,  return_tensors="pt", return_attention_mask=False).input_ids

            
            #TODO: define best generation config
            generation_config = GenerationConfig(
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id,
                num_beams = 10,
                num_beam_groups = 5,
                max_new_tokens = 600,
                diversity_penalty = 1.0,
                repetition_penalty = 1.2,
                early_stopping=False,
                no_repeat_ngram_size = 5
            )

            #the most strict config to try to get an answer if the used config fails (greedy decoding)
            fallback_config = GenerationConfig(
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id,
                num_beams = 1,
                repetition_penalty = 1.5,
                max_new_tokens = 600,
                no_repeat_ngram_size = 5
            )
                
            #do the generation in smaller batches to avoid OOM
            batch_size = 1
            answer = []
            with torch.no_grad():
                for i in range(0, len(all_input_ids), batch_size):
                    max_idx = min(len(all_input_ids), i + batch_size)
                    answer.append(self.pretrained_model.generate(inputs=all_input_ids[i : max_idx].to(self.pretrained_model.device), generation_config=generation_config, tokenizer=tokenizer))
            
            all_input_ids = all_input_ids.cpu()
            #concatenate all tensors in one as if the generation was done in a full batch
            answer = [a.unsqueeze(0).detach().cpu() if a.dim() == 1 else a.detach().cpu() for a in answer]
            if len(answer) > 1:
                answer = torch.cat(pad2dtensors(answer, tokenizer.pad_token_id), dim=0)
            else:
                answer = answer[0]
                
            answers = tokenizer.batch_decode(answer, skip_special_tokens=False)

            answers = list(enumerate(answers))

            #extract answers
            results, bad_responses_input_ids = get_answers(all_input_ids, answers, processed_batch_opts)

            # print(f"{results = }")

            
            #if all the answers are well formed, we return
            if len(results) == len(answers):
                results = [result[1] for result in results]
                return {"preds": results}

            #else, we try the failed ones with the fallback config
            idxs_bad_responses, input_ids = zip(*bad_responses_input_ids)
            idxs_bad_responses = list(idxs_bad_responses)

            input_ids = [input_id.unsqueeze(0) if input_id.dim() == 1 else input_id for input_id in input_ids]

            #do not need to pad as they all have the same size (due to tokenizer padding)
            if len(input_ids) > 1:
                input_ids = torch.cat(input_ids, dim=0)
            else:
                input_ids = input_ids[0]

            #do the same to avoid OOM
            answer = []
            with torch.no_grad():
                for i in range(0, len(input_ids), batch_size):
                    max_idx = min(len(input_ids), i + batch_size)
                    answer.append(self.pretrained_model.generate(inputs=input_ids[i : max_idx].to(self.pretrained_model.device), generation_config=fallback_config, tokenizer=tokenizer))
            
            #concatenate all tensors in one as if the generation was done in a full batch
            answer = [a.unsqueeze(0).detach().cpu() if a.dim() == 1 else a.detach().cpu() for a in answer]

            if len(answer) > 1:
                answer = torch.cat(pad2dtensors(answer, tokenizer.pad_token_id), dim=0)
            else:
                answer = answer[0]

            answers = tokenizer.batch_decode(answer, skip_special_tokens=False)

            answers = list(zip(idxs_bad_responses, answers))

            #get all the missing results (if not found again, we return 'A')
            results_bad_responses = get_answers(all_input_ids, answers, processed_batch_opts, force_answer=True)
            results += results_bad_responses
            # print(f"{results_bad_responses = }")
            # print(f"{results = }")

            assert len(results) == len(batch["question"])

            
            #sort them back
            results.sort(key=lambda x: x[0])
            results = [result[1] for result in results]

            print(results)

            return {"preds": results}





















######################################################################################################## NOT USED ########################################################################################################
class AutoDPOModelForSeq2SeqLM(PreTrainedModelWrapper):
    r"""
    A seq2seq model with support for custom modules in addition to the transformer model.
    This class inherits from `~trl.PreTrainedModelWrapper` and wraps a
    `transformers.PreTrainedModel` class. The wrapper class supports classic functions
    such as `from_pretrained` and `push_to_hub` and also provides some additional
    functionalities such as `generate`.

    Args:
        pretrained_model (`transformers.PreTrainedModel`):
            The model to wrap. It should be a causal language model such as GPT2.
            or any model mapped inside the `AutoModelForSeq2SeqLM` class.
        kwargs:
            Additional keyword arguments passed along to any `CustomModule` classes.
    """

    transformers_parent_class = AutoModelForSeq2SeqLM
    lm_head_namings = ["lm_head", "embed_out", "output_projection"]
    ####################################################################################
    # TODO (Optional): Please put any required arguments for your custom module here
    supported_args = ()
    ####################################################################################

    def __init__(self, pretrained_model, **kwargs):
        super().__init__(pretrained_model, **kwargs)
        self.is_encoder_decoder = True
        if not self._has_lm_head():
            raise ValueError("The model does not have a language model head, please use a model that has one.")

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure CustomModule is repalced with the name of your custom module class
        # Remember that the below lines are just an example
        # You can reanme the class and the variabels to fit your custom module name,
        # just make sure they are consistent in the code
        # =========================================================================================
        # custom_module_kwargs, _, _ = self._split_kwargs(kwargs)
        # self.custom_module = CustomModule(self.pretrained_model.config, **custom_module_kwargs)
        # self._init_weights(**custom_module_kwargs)
        ###########################################################################################

    def _has_lm_head(self):
        # check module names of all modules inside `pretrained_model` to find the language model head
        for name, _module in self.pretrained_model.named_modules():
            if any(attribute in name for attribute in self.lm_head_namings):
                return True
        return False

    def _init_weights(self, **kwargs):
        """
        Initializes the weights of the custom module. The default initialization strategy is random.
        Users can pass a different initialization strategy by passing the `custom_module_init_strategy`
        argument when calling `.from_pretrained`. Supported strategies are:
            - `normal`: initializes the weights with a normal distribution.

        Args:
            **kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the `CustomModule` class.
        """
        ###############################################################
        # TODO (Optional): Please implement the initialization strategy for your custom module here
        pass
        ###############################################################

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the custom module
        to the state dictionary of the wrapped model by prepending the key with `custom_module.`.

        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not self.is_peft_model:
            pretrained_model_state_dict = self.pretrained_model.state_dict(*args, **kwargs)
        else:
            pretrained_model_state_dict = {}

        ###########################################################################################
        # TODO (Optional): Please uncomment the following lines to initialize your custom module
        # Make sure "custom_module" is repalced with the name of your custom module class
        # =========================================================================================
        # custom_module_state_dict = self.custom_module.state_dict(*args, **kwargs)
        # for k, v in custom_module_state_dict.items():
        #     pretrained_model_state_dict[f"custom_module.{k}"] = v
        ###########################################################################################
        return pretrained_model_state_dict

    def post_init(self, state_dict):
        r"""
        We add the state dictionary of the custom module to the state dictionary of the wrapped model
        by prepending the key with `custom_module.`. This function removes the `custom_module.` prefix from the
        keys of the custom module state dictionary.
chosen_labels
        IMPORTANT: Make sure to replace `custom_module` with the name of your custom module class name.
        """
        if not hasattr(self, 'custom_module'):
            return

        for k in list(state_dict.keys()):
            if "custom_module." in k:
                state_dict[k.replace("custom_module.", "")] = state_dict.pop(k)
        self.custom_module.load_state_dict(state_dict, strict=False)
        del state_dict

        if hasattr(self.pretrained_model, "hf_device_map"):
            if (
                "cpu" in self.pretrained_model.hf_device_map.values()
                or "disk" in self.pretrained_model.hf_device_map.values()
            ):
                raise ValueError(
                    "The model is offloaded on CPU or disk - CPU & disk offloading is not supported for CustomModule models."
                )

            # get the lm_head device
            for name, module in self.pretrained_model.named_modules():
                if any(attribute in name for attribute in self.lm_head_namings):
                    lm_head_device = module.weight.device
                    break

            # put custom_module on the same device as the lm_head to avoid issues
            self.custom_module = self.custom_module.to(lm_head_device)

            def set_device_hook(module, input, outputs):
                r"""
                A hook that sets the device of the output of the model to the device of the first
                parameter of the model.

                Args:
                    module (`nn.Module`):
                        The module to which the hook is attached.
                    input (`tuple`):
                        The input to the module.
                    outputs (`tuple`):
                        The output of the module.
                """
                new_output = ()
                for output in outputs:
                    if isinstance(output, torch.Tensor):
                        new_output += (output.to(lm_head_device),)
                    else:
                        new_output += (output,)
                return new_output

            self.register_forward_hook(set_device_hook)
            self.is_sequential_parallel = True

    def push_to_hub(self, *args, **kwargs):
        """Push the model to the Hugging Face hub."""
        ###########################################################################################
        # TODO (Optional): Please uncomment the following line to add the custom module to the hub model
        # Make sure custom_module is repalced with the name of your custom module class
        # =========================================================================================
        # self.pretrained_model.custom_module = self.custom_module
        ###########################################################################################

        return self.pretrained_model.push_to_hub(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs,
    ):
        r"""
        Applies a forward pass to the wrapped model and returns the output from the model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, `optional`):
                Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model
                (see `past_key_values` input) to speed up sequential decoding.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            kwargs (`dict`, `optional`):
                Additional keyword arguments, that are passed to the wrapped model.
        Returns:
            ouput_dict (`dict`): A dictionary containing the output from the model.
        """
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        if self.is_peft_model and self.pretrained_model.active_peft_config.peft_type == "PREFIX_TUNING":
            kwargs.pop("past_key_values")

        ouput_dict = {}

        ###############################################################
        # TODO: Please implement your customized forward pass here
        # =============================================================
        raise NotImplementedError
        ###############################################################

        return ouput_dict

    def get_logprobs(self, batch, tokenizer):
        """
        Computes the log probabilities of a response using the model respectively.

        Args:
            batch (`dict` of `list`): A dictionary containing the input data for the DPO model.
                The data format is as follows:
                {
                    "prompt": List[str],
                    "chosen": List[str],
                    "rejected": List[str],
                    "chosen_logps": Optional(torch.FloatTensor)
                    "rejected_logps": Optional(torch.FloatTensor)
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input data.
        Returns:
            A tuple of two tensors: (chosen_logps, rejected_logps)
            chosen_logps (`torch.FloatTensor`):
                Log probabilities of the chosen responses. Shape: (batch_size,)
            rejected_logps (`torch.FloatTensor`):
                Log probabilities of the rejected responses. Shape: (batch_size,)
        """
        ###############################################################
        # TODO: Please implement your customized logprob computation here
        # =============================================================
        raise NotImplementedError
        ###############################################################

        return chosen_logps, rejected_logps

    def prediction_step_reward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ):
        """
        Computes the reward socres of the chosen and reject responses by implementing the DPO reward function
        Reference of the DPO reward function: https://arxiv.org/pdf/2305.18290.pdf

        Args:
            policy_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps (`torch.FloatTensor`):
                Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        Returns:
            output_dict (`dict`):
                A dictionary containing the reward scores of the chosen and rejected responses.
        """
        output_dict = {
            "chosen_rewards": [],
            "rejected_rewards": []
        }

        ########################################################################
        # TODO: Please implement the dpo loss function to compute the rewards
        # You need to return one reward score for each chosen and rejected response.
        # ======================================================================
        raise NotImplementedError
        ########################################################################

        return output_dict

    def prediction_step_mcqa(self, batch, tokenizer):
        """
        Computes the mcqa prediction of the given question.

        Args:
            batch (`list` of `dict`):
                A list of dictionaries containing the input mcqa data for the DPO model.
                The data format is as follows:
                {
                    "question": str,
                    "choices": List[str],
                    "answer": str,
                }
            tokenizer (`PreTrainedTokenizerBase`): The tokenizer used to tokenize the input questions.
        Returns:
            output_dict (`dict`): A dictionary containing the model predictions given input questions.
        """
        output_dict = {"preds": []}

        ########################################################################
        # TODO: Please implement the prediction step that generates the prediction of the given MCQA question
        # ======================================================================
        # You need to return one letter prediction for each question.
        # ======================================================================
        raise NotImplementedError
        ########################################################################

        return output_dict
