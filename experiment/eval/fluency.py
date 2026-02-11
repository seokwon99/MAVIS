from experiment.eval.mauve.examples import load_gpt2_dataset
import pandas as pd
import re
import argparse
from nltk.tokenize import sent_tokenize

def preprocess_text(text):
    # [숫자] 형태의 인용 제거
    text = re.sub(r'\[\d+\]', '', text)

    # 줄바꿈 문자(\n, \r 등) 제거
    text = text.replace('\n', ' ').replace('\r', ' ')

    # 중복된 공백을 하나로 줄이기
    text = re.sub(r'\s+', ' ', text)

    # 양쪽 공백 제거
    text = text.strip()

    text = text.replace(" .", ".")
    text = text.replace("**", "")
    return text

def read_df_with_references(path):
    df = pd.read_json(path, lines=True)
    
    df['pos_image_reference'] = df['pos_documents'].apply(lambda x: len([doc for doc in x if "-" not in str(doc)]))
    df['pos_text_reference'] = df['pos_documents'].apply(lambda x: len([doc for doc in x if "-" in str(doc)]))

    df = df[df['pos_image_reference'] > 0]
    df = df[df['pos_text_reference'] > 0]
    
    return df

parser = argparse.ArgumentParser()
parser.add_argument("--eval-path", type=str)
args = parser.parse_args()

path = args.eval_path
df = read_df_with_references(path)
print(f"Evaluating path: {path}")

df = df[df['documents_id'].apply(lambda x: len(x) > 0)]

df = df[df.A.apply(lambda x: len(x[0])) > 500]
p_text = df.apply(lambda x: " ".join(preprocess_text(x['Q'] + " " +  x['A'][0]).split()), axis=1).tolist() # human
q_text = df.apply(lambda x: " ".join(preprocess_text(x['Q'] + " " + x['answer']).split()), axis=1).tolist() # machine
q_text = [t for t in q_text if ("Error" not in t) and ("document" not in t.lower()) and ("information" not in t.lower())]

import mauve
out = mauve.compute_mauve(p_text=p_text, q_text=q_text, device_id=0, max_text_length=512, verbose=False, batch_size=8, featurize_model_name="gpt2-large")
print(out.mauve)