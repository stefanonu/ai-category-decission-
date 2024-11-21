from gpt4all import GPT4All
import json
import pyreadr
from model import CategoryScores
import re
import time
import pandas as pd
import asyncio
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    filename=f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PROMPT_TEMPLATE = """You have this abstract:
{abstract}

Your job is to find me the best match category for this abstract using this list [{categories}].

Maybe some articles are part of multiple categories, try to define ratio from 0-10 for each category where maximum is 10 and minum is 0.

It is the most important task to give score for all categories, if it's not mentioned you can assign 0, do not suppose things.

Also explain why you made that decission
"""

class ProcessingStats:
    def __init__(self):
        self.start_time = time.time()
        self.processed = 0
        self.errors = 0
        self.total_processing_time = 0

    def update(self, success=True, processing_time=0):
        self.processed += 1
        self.total_processing_time += processing_time
        if not success:
            self.errors += 1

    def get_stats(self):
        elapsed = time.time() - self.start_time
        return {
            "processed": self.processed,
            "errors": self.errors,
            "avg_time": self.total_processing_time / self.processed if self.processed else 0,
            "total_time": elapsed
        }

def extract_json(text, idx):
    try:
        logging.info(text)
        # Find the first '{' and last '}'
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and start < end:
            json_str = text[start:end + 1]
            json_data = json.loads(json_str)
            return json_data
        else:
            logging.info(f"Text: {text}")
            return None
    except json.JSONDecodeError as e:
        logging.info(f"Text: {text}")
        return None
    except Exception as e:
        logging.info(f"Text: {text}")
        return None

class AbstractProcessor:
    def __init__(self, model_path="Meta-Llama-3-8B-Instruct.Q4_0.gguf", num_models=2):
        self.model_path = model_path
        self.models = [GPT4All(model_path, n_threads=15) for _ in range(num_models)]
        self.current_model = 0
        self.lock = asyncio.Lock()
        self.stats = ProcessingStats()
        self.processed_count = 0
        self.reset_threshold = 50

    def get_next_model(self):
        self.current_model = (self.current_model + 1) % len(self.models)
        return self.models[self.current_model]

    async def process_batch(self, batch, categories, start_idx):
        batch_start_time = time.time()
        results = []
        
        try:
            async with self.lock:
                model = self.get_next_model()
                for idx, abstract in enumerate(batch):
                    absolute_idx = start_idx + idx
                    try:
                        # Create new chat session for each abstract
                        with model.chat_session():
                            response = model.generate(
                                PROMPT_TEMPLATE.format(
                                    abstract=abstract[:1000],  # Limit abstract length
                                    categories=categories
                                ),
                                max_tokens=1024
                            )
                            
                            data = extract_json(response, absolute_idx)
                            if data:
                                category_scores = CategoryScores(**data)
                                results.append({
                                    'abstract': abstract,
                                    'scores': {category: float(getattr(category_scores, category)) for category in categories}
                                })
                                self.stats.update(True, time.time() - batch_start_time)
                            else:
                                results.append({'abstract': abstract, 'scores': None})
                                self.stats.update(False, time.time() - batch_start_time)
                    except Exception as e:
                        logging.error(f"Error processing abstract {absolute_idx}: {e}")
                        results.append({'abstract': abstract, 'scores': None})
                        self.stats.update(False, time.time() - batch_start_time)

                    # Reset model more frequently
                    self.processed_count += 1
                    if self.processed_count >= self.reset_threshold:
                        logging.info(f"Resetting model after {self.processed_count} processes")
                        self.models[self.current_model] = GPT4All(self.model_path, n_threads=15)
                        self.processed_count = 0
            
        except Exception as e:
            logging.error(f"Batch processing error: {e}")
            results.extend([{'abstract': abstract, 'scores': None} for abstract in batch])
        
        return results

async def process_all_abstracts(abstracts, categories, batch_size=10):
    processor = AbstractProcessor()
    results = []
    pbar = tqdm(total=len(abstracts))
    
    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i:min(i + batch_size, len(abstracts))]
        batch_results = await processor.process_batch(batch, categories, i)
        results.extend(batch_results)
        pbar.update(len(batch))
        
        # Log progress
        stats = processor.stats.get_stats()
        logging.info(f"Progress: {stats}")
        
    pbar.close()
    return results, processor.stats

async def main():
    start_time = time.time()
    logging.info("Starting processing")

    # Load data
    result = pyreadr.read_r('all_art1_49_en.rds')
    df = result[None]

    categories = [
        "social_interaction", "language", "economy", "policy", "education", "art",
        "health", "family", "media", "philosophy", "age", "technology", "national",
        "culture", "environment", "law", "gender", "history", "psychology", "religion",
        "inequality", "urban", "food", "sport", "travel", "war", "cities_countries", "labour"
    ]

    # Take first 60 rows for testing
    sample_df = df.head(10)
    abstracts = sample_df['abstract'].tolist()

    # Process abstracts
    all_results, stats = await process_all_abstracts(abstracts, categories)

    # Update DataFrame with results
    new_df = df.copy()
    for idx, result in enumerate(all_results):
        if result['scores']:
            new_df.loc[idx, categories] = pd.Series(result['scores'])

    # Save the updated DataFrame
    output_file = f'updated_all_art1_49_en_{datetime.now().strftime("%Y%m%d_%H%M%S")}.rds'
    # pyreadr.write_rds(output_file, new_df)

    # Log final statistics
    end_time = time.time()
    execution_time = end_time - start_time
    final_stats = {
        "total_execution_time": execution_time,
        "average_time_per_abstract": execution_time/len(abstracts),
        "processed_abstracts": stats.processed,
        "errors": stats.errors,
        "average_processing_time": stats.total_processing_time/stats.processed
    }
    
    logging.info(f"Final Statistics: {final_stats}")
    print(f"Total execution time: {execution_time:.2f} seconds")
    print(f"Average time per abstract: {execution_time/len(abstracts):.2f} seconds")
    print(f"Processed abstracts: {stats.processed}")
    print(f"Errors: {stats.errors}")

if __name__ == '__main__':
    asyncio.run(main())
