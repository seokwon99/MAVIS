import pandas as pd
import argparse
import numpy as np
import re
from nltk.tokenize import sent_tokenize

def extract_references(line):
    text = line['answer']
    len_docs = len(line['documents_id'])
    pattern = r'(.*?)\s*\[([^\]]+)\]'
    matches = re.findall(pattern, text)

    results = [] 

    for full_sentence, tag in matches:
        tokenized = sent_tokenize(full_sentence.strip())
        if tokenized:
            last_sentence = tokenized[-1].strip()
            if len(last_sentence) == 0:
                last_sentence = results[-1][0]
            results.append((last_sentence, f'[{tag}]'))

    new_results = []
    for sentence, refs in results:
        if len(sentence) < 5 and len(new_results) > 0:
                new_results[-1] = (new_results[-1][0], new_results[-1][1] + "," + refs)
        else:
            new_results.append((sentence, refs))
    results = new_results
    subscript_map = {'₀':'0', '₁':'1', '₂': '2', '²':'2', '₃':'3', '₄':'4', '₅':'5', '₆':'6', '₇':'7', '₈':'8', '₉':'9'}

    def normalize_digits(s):
        return ''.join(subscript_map.get(ch, ch) for ch in s)

    total_refs = [
        int(normalize_digits(ref))
        for sentence, refs in results
        for ref in refs
        if normalize_digits(ref).isdigit() and int(normalize_digits(ref)) - 1 < len_docs
    ]
    total_refs = list(set(total_refs))  # Remove duplicates
    return total_refs

def read_df_with_references(path):
    df = pd.read_json(path, lines=True)

    if 'answer' in df.columns:
        df['reference'] = df.apply(extract_references, axis=1)
        df['used_reference'] = df['reference'].apply(len)
        df['given_image_reference'] = df['documents_id'].apply(lambda x: len([doc for doc in x if ("-" not in doc) and ("_" not in doc)]))
        df['given_text_reference'] = df['documents_id'].apply(lambda x: len([doc for doc in x if ("-" in doc) or ("_" in doc)]))
        df['used_image_reference'] = df.apply(lambda x: len([ref for ref in x['reference'] if ("-" not in x['documents_id'][ref-1]) and ("_" not in x['documents_id'][ref-1])]), axis=1)
        df['used_text_reference'] = df.apply(lambda x: len([ref for ref in x['reference'] if ("-" in x['documents_id'][ref-1]) or ("_" in x['documents_id'][ref-1])]), axis=1)
        
    df['pos_image_reference'] = df['pos_documents'].apply(lambda x: len([doc for doc in x if "-" not in str(doc)]))
    df['pos_text_reference'] = df['pos_documents'].apply(lambda x: len([doc for doc in x if "-" in str(doc)]))

    df = df[df['pos_image_reference'] > 0]
    df = df[df['pos_text_reference'] > 0]
    df = df[df.apply(lambda x: x['pos_image_reference'] - x['pos_text_reference'] >= 1, axis=1)]

    if 'completeness' in df.columns and 'relevance' in df.columns:
        df['mean'] = df.apply(lambda x: (x['completeness'] + x['relevance']) / 2, axis=1)
    if 'recall' in df.columns and 'precision' in df.columns:
        df['precision'] = df['groundedness'].apply(lambda x: np.mean(list(x.values())))
        df['f1'] = df.apply(lambda x: (x['recall'] + x['precision']) / 2, axis=1)
         
    return df




parser = argparse.ArgumentParser()
parser.add_argument("--eval-path", type=str)
parser.add_argument("--sample", type=int, default=-1, help="Sample fraction for evaluation")
args = parser.parse_args()


image_path_to_subreddit = pd.read_csv("experiment/image_path_to_subreddit.csv").to_dict(orient="records")
image_path_to_subreddit = {item["hash"]: item["subreddit"] for item in image_path_to_subreddit}
subreddit_to_domain = {
    # Art & Design
    "Art": "Art & Design",
    "Design": "Art & Design",
    "Filmmakers": "Art & Design",
    "GraphicDesign": "Art & Design",
    "Illustration": "Art & Design",
    "Music": "Art & Design",
    "architecture": "Art & Design",
    "femalefashionadvice": "Art & Design",
    "frugalmalefashion": "Art & Design",
    "malefashionadvice": "Art & Design",
    "musictheory": "Art & Design",
    "photocritique": "Art & Design",
    "vinyl": "Art & Design",
    "houseplants": "Art & Design",
    "gardening": "Art & Design",
    "HomeImprovement": "Tech & Engineering",
    "woodworking": "Tech & Engineering",

    # Business
    "personalfinance": "Business",
    "Daytrading": "Business",
    "StockMarket": "Business",
    "wallstreetbets": "Business",
    "Economics": "Business",
    "algotrading": "Business",

    # Science
    "science": "Science",
    "nasa": "Science",
    "math": "Science",
    "chemistry": "Science",
    "Physics": "Science",
    "biology": "Science",
    "neuroscience": "Science",
    "Astronomy": "Science",
    "AskPhysics": "Science",
    "AskChemistry": "Science",
    "askscience": "Science",
    "AskStatistics": "Science",
    "genetics": "Science",
    "space": "Science",
    "Futurology": "Science",
    "askmath": "Science",
    "learnmath": "Science",
    "matheducation": "Science",
    "datascience": "Science",
    "astrophotography": "Science",
    "environment": "Science",
    "natureismetal": "Science",
    "UFOs": "Science",
    "singularity": "Science",
    "theydidthemath": "Science",
    "astrology": "Science",

    # Health & Medicine
    "Health": "Health & Medicine",
    "medicine": "Health & Medicine",
    "medical_advice": "Health & Medicine",
    "AskMedical": "Health & Medicine",

    # Humanities & Social Science
    "sociology": "Humanities & Social Science",
    "AskHistorians": "Humanities & Social Science",
    "AskHistory": "Humanities & Social Science",
    "history": "Humanities & Social Science",
    "classics": "Humanities & Social Science",
    "politics": "Humanities & Social Science",
    "PoliticalScience": "Humanities & Social Science",
    "linguistics": "Humanities & Social Science",
    "religion": "Humanities & Social Science",

    # Tech & Engineering
    "SoftwareEngineering": "Tech & Engineering",
    "engineering": "Tech & Engineering",
    "ElectricalEngineering": "Tech & Engineering",
    "AskEngineers": "Tech & Engineering",
    "webdev": "Tech & Engineering",
    "hardware": "Tech & Engineering",
    "technology": "Tech & Engineering",
    "pcmasterrace": "Tech & Engineering",
    "gadgets": "Tech & Engineering",
    "javascript": "Tech & Engineering",
    "AskElectricians": "Tech & Engineering",
    "AskElectronics": "Tech & Engineering",
    "ChemicalEngineering": "Tech & Engineering",
    "industrialengineering": "Tech & Engineering",
    "buildapc": "Tech & Engineering",
    "AskComputerScience": "Tech & Engineering",
    "compsci": "Tech & Engineering",
    "programming": "Tech & Engineering",
    "techsupport": "Tech & Engineering",
    "3Dprinting": "Tech & Engineering",
    "raspberry_pi": "Tech & Engineering",
    "MechanicalEngineering": "Tech & Engineering",
    "civilengineering": "Tech & Engineering",
    "AerospaceEngineering": "Tech & Engineering",
    "DIY": "Tech & Engineering",
    "homeautomation": "Tech & Engineering",
    "mac": "Tech & Engineering",
    "hacking": "Tech & Engineering",
    "aviation": "Tech & Engineering",
    "dataisbeautiful": "Tech & Engineering",

    # Others
    "Survival": "Others",
    "Outdoors": "Others",
    "camping": "Others",
    "hiking": "Others",
    "homestead": "Others",
    "NationalPark": "Others",
    "backpacking": "Others",
    "roadtrip": "Others",
    "TravelHacks": "Others",
    "JapanTravel": "Others",
    "homeowners": "Others",
    "AdviceAnimals": "Others",
    "Weird": "Others",
    "boardgames": "Others",
    "Mommit": "Others",
    "sports": "Others",
    "PremierLeague": "Others",
    "soccer": "Others",
    "formula1": "Others",
    "MMA": "Others",
    "ufc": "Others",
    "CFB": "Others",
    "mlb": "Others",
    "cookingforbeginners": "Others",
    "foodhacks": "Others",
    "lifehacks": "Others",
    "howto": "Others",
    "snowboarding": "Others",
    "whatisthisthing": "Others",
    "confusing_perspective": "Others",
    "scifi": "Others",
    "worldbuilding": "Others",
    "CasualUK": "Others",
    "college": "Others",
    "Parenting": "Humanities & Social Science",
    "languagelearning": "Humanities & Social Science",
    "europe": "Humanities & Social Science",
    "GetStudying": "Humanities & Social Science",
}



if __name__ == "__main__":
    # df = read_df_with_references(args.eval_path)
    df = read_df_with_references(args.eval_path)
    df['subreddit'] = df['image'].apply(lambda x: image_path_to_subreddit.get(x.split("/")[-1].split(".")[0], "Unknown"))
    df['domain'] = df['subreddit'].apply(lambda x: subreddit_to_domain.get(x, "Unknown"))

    if args.sample > 0:
        # df = df[df['domain'] != "Others"]
        # df = df.groupby('subreddit').apply(lambda x: x.sample(n=min(len(x), args.sample * 2))).reset_index(drop=True)
        # df = df.groupby('domain').apply(lambda x: x.sample(n=min(len(x), args.sample))).reset_index(drop=True)
        df = df.sample(n=args.sample, random_state=42)

    print(df.groupby('subreddit').size())
    print(df.groupby('domain').size())
    df.to_json(args.eval_path.replace(".jsonl", "_sampled.jsonl"), orient="records", lines=True)