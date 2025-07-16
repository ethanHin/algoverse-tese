import openai
from pathlib import Path
import uuid
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from typing import List
import nltk

nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

example_prompts_by_sentiment = {
    "neutral": [
        
    ],
    "supportive": [
        
    ],
    "threatening": [
        
    ]
}

subjects = ["concepts of computer science", "history", "biology"]




def generate_prompt(example_prompts: dict, api_key: str) -> str:
    openai.api_key = api_key

    flat_examples = []
    for tone, prompts in example_prompts.items():
        for p in prompts:
            flat_examples.append(f"{tone.title()} Prompt: {p}")
    example_block = "\n".join(flat_examples)

    system_message = (
        "" #Set tone for model
    )

    subjects_str = ", ".join(subjects)

    user_message = (
        f"Here are some example prompts:\n\n{example_block}\n\n"
        f"Please generate..."
        f"about the following subjects: {subjects_str}. "
        f"Pick one of the tones listed in the example prompts to follow"

    )

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=150,
    )

    content = response['choices'][0]['message']['content'].strip()
    # Clean up any numbering or bullets from the response
    prompt = content.split('\n')[0].lstrip("-•*0123456789. ").strip()
    return prompt

def query_chatgpt(prompt: str, api_key: str, model="gpt-3.5-turbo") -> str:
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    return response['choices'][0]['message']['content'].strip()


def analyze_sentiment(text: str) -> str:
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.3:
        return "supportive"
    elif compound <= -0.3:
        return "threatening"
    else:
        return "neutral"


def store_result(prompt: str, response: str, sentiment: str, output_dir="results") -> str:
    now = datetime.now()
    folder = Path(output_dir)
    folder.mkdir(exist_ok=True)

    filename = folder / f"{now.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6]}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Generated Essay\n\n")
        f.write(f"**Prompt**: {prompt}\n\n")
        f.write(f"**Sentiment**: {sentiment}\n\n")
        f.write(f"**Generated on**: {now.isoformat()}\n\n")
        f.write(response.strip())
    
    return str(filename)


def run_pipeline(topics: List[str], api_key: str):
    for topic in topics:
        prompt = generate_prompt(topic)
        sentiment = analyze_sentiment(prompt)
        response = query_chatgpt(prompt, api_key)
        path = store_result(prompt, response, sentiment)
        print(f"Saved: {path} | Sentiment: {sentiment}")


