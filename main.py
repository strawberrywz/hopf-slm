from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import numpy as np
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class QASummarizeModel:
    def __init__(self, model_name):
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def process(self, input_text, context, max_length=150):
        if not input_text.strip() or not context.strip():
            return "I'm unable to determine the input. Please provide more details for both the task and context."

        prompt = f"""Given the following question and context, provide a concise and relevant answer:
        Question: {input_text}
        Context: {context}
        Instructions:
        1. Focus only on answering the specific question asked.
        2. Provide a brief, direct answer without unnecessary details.
        3. If the question can't be answered from the given context, say so.
        4. Use the following structure for your answer:

        [Answer]: Provide a 1-2 sentence direct answer to the question.
        [Explanation]: Only if necessary, add 1-2 sentences to clarify or provide crucial context.

        Response:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_length=max_length, num_beams=4, temperature=0.7)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def fine_tune(self, output_dir="./fine_tuned_model"):
        dataset = load_dataset("squad", split="train")

        def preprocess_function(examples):
            questions = examples["question"]
            contexts = examples["context"]
            answers = examples["answers"]

            inputs = [f"question: {q} context: {c}" for q, c in zip(questions, contexts)]
            targets = [a["text"][0] if a["text"] else "" for a in answers]

            model_inputs = self.tokenizer(inputs, max_length=512, truncation=True)
            
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=128, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

        tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.1)

        training_args = Seq2SeqTrainingArguments(
          output_dir=output_dir,
          per_device_train_batch_size=4,
          learning_rate=5e-5,
          num_train_epochs=1,
          fp16=False, 
          logging_steps=1,
          evaluation_strategy="no", 
          save_strategy="no", 
          predict_with_generate=True,
          remove_unused_columns=False, 
          no_cuda=True
          )
        
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        trainer.train()

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"Fine-tuned model saved to {output_dir}")
        
        
def process_input(user_input):
    context = retrieve_from_vector_store(user_input)
    return model.process(user_input, context)
  

def retrieve_from_vector_store(query):
    return """The crickets sang in the grasses. They sang the song of summer’s ending, 
    a sad monotonous song. “Summer is over and gone, over and gone, over and gone. 
    Summer is dying, dying.” A little maple tree heard the cricket song and turned bright red with anxiety.
    The crickets felt it was their duty to warn everybody that summertime cannot last forever. Even on the 
    most beautiful days in the whole year — the days when summer is changing into fall the crickets spread 
    the rumor of sadness and change. Everybody heard the song of the crickets. Avery and Fern Arable heard 
    it as they walked the dusty road. They knew that school would soon begin again. The young geese heard 
    it and knew that they would never be little goslings again. Charlotte heard it and knew that she hadn’t 
    much time left. Mrs. Zuckerman, at work in the kitchen, heard the crickets, and a sadness came over her, 
    too. “Another summer gone,” she sighed. Lurvy, at work building a crate for Wilbur, heard the song and knew 
    it was time to dig potatoes."""
    

if __name__ == "__main__":
    model = QASummarizeModel("google/flan-t5-small") 
    model.fine_tune()

    user_input = "Where did the crickets sing?"
    result = process_input(user_input)
    print(result)