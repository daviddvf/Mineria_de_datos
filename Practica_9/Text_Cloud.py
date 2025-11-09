# Text Cloud
import os
import re
from collections import Counter
import pandas as pd
from unidecode import unidecode
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

CSV_Dir = "../Practica_1/ntsb.csv"
OUT_Dir = "wc_out"
os.makedirs(OUT_Dir, exist_ok=True)

# Carga
df = pd.read_csv(CSV_Dir, low_memory=False, dtype=str)
df["narr_cause"] = df.get("narr_cause", pd.Series()).fillna("missing").astype(str)

# limpieza y normalizacion
def clean_text(s: str) -> str:
    if s is None:
        return ""
    # Elimina acentos y caracteres no estandar
    s = unidecode(str(s))

    # Convierte a minusculas
    s = s.lower()

    # Elimina notas de metadatos, edits o firmas por ejemplo: "*This report was modified on ...*"
    s = re.sub(r'\*this report was modified on[^\n]*\*', ' ', s)
    s = re.sub(r'member [a-z0-9 \.\'\"\-]+ did not approve[^\n]*', ' ', s)

    # Normaliza y elimina texto repetitivo que carece de informacion util
    patterns = [
        r'contribut(?:ing|ed)? (?:to the|to|were|was|factors? (?:in)? the accident(?: were)?)',
        r'contributing factor was',
        r'contributing factors were',
        r'contributing to the incident',
        r'contributing to the accident',
        r'a factor (?:in|was)',
        r'also causal was',
        r'also causal (?:was|were)',
        r'the probable cause of this accident is [^\.\n]+',
        r'the reason for this occurrence was [^\.\n]+',
        r'causing the accident[^\.\n]*',
        r'contributing to the severity [^\.\n]*'
    ]
    for pat in patterns:
        s = re.sub(pat, ' ', s)

    # Reemplaza frases largas de "undetermined" por token unico
    s = re.sub(r'\b(the probable cause of this accident is undetermined|the reason for this occurrence was undetermined|for undetermined reasons)\b', ' undetermined ', s)

    # Quita posesivos ('s o ’s) ejemplo: convierte "pilot's" a "pilot"
    s = re.sub(r"(\w)[\'\u2019]s\b", r"\1", s)

    # Normaliza "no." a "no"
    s = re.sub(r'\bno\.\s*([0-9]+)\b', r'no \1', s)

    # Elimina urls / emails
    s = re.sub(r'http[s]?://\S+', ' ', s)
    s = re.sub(r'\S+@\S+', ' ', s)

    # Elimina cualquier comilla residual y asteriscos y caracteres extraños
    s = s.replace('"', ' ').replace("''", ' ').replace('``', ' ').replace('*', ' ')

    # Elimina todo lo que no sea letra, numero o espacio
    s = re.sub(r'[^a-z0-9\s]', ' ', s)

    # Recorta espacios
    s = re.sub(r'\s+', ' ', s).strip()

    if s == "":
        return ""
    return s

df["narr_clean"] = df["narr_cause"].map(clean_text)

# Stopwords
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords as nltk_stopwords

stop = set(STOPWORDS)
stop.update(nltk_stopwords.words("english"))

# Se añaden stopwords especificas
stop_add = {
    "accident","contributing","contribute","was","resulting","causing","contributor",
    "contributed","part","due","to","the","a","an","missing","contributingto"
}
stop.update(stop_add)

# Unigrama
# Concatena todos los textos excluyendo "missing"
corpus = df.loc[df["narr_clean"].str.strip().ne("missing"), "narr_clean"].dropna().tolist()
all_text = " ".join(corpus)
tokens = [t for t in all_text.split() if t not in stop and len(t) > 1]

# Calcula frecuencias
freq = Counter(tokens)
# Crea y guarda CSV con un top de 200 palabras mas comunes
top_unigrams = freq.most_common(200)
pd.DataFrame(top_unigrams, columns=["word","count"]).to_csv("wc_out/top_unigrams.csv", index=False)

# Genera y guarda unigrama
wc = WordCloud(width=1200, height=600, background_color="white",
               stopwords=stop, max_words=200, collocations=False).generate(" ".join(tokens))

plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("wc_out/top_unigrams.png", dpi=300)
plt.close()


# Biograma

stop_list = [str(w) for w in sorted(stop)]

vectorizer = CountVectorizer(ngram_range=(2,2), stop_words=stop_list, min_df=3)
texts_for_bigrams = df.loc[df["narr_clean"].str.strip().ne("missing"), "narr_clean"].fillna("").astype(str)
X = vectorizer.fit_transform(texts_for_bigrams)
sums = X.sum(axis=0).A1
terms = vectorizer.get_feature_names_out()
bigram_counts = dict(zip(terms, sums))

# Genera y guarda CSV con las top 200 bi palabras mas comunes
pd.DataFrame(sorted(bigram_counts.items(), key=lambda x: -x[1])[:200], columns=["bigram","count"]).to_csv("wc_out/top_bigrams.csv", index=False)

# Genera texto de bigrams
bigram_text = " ".join([ (t.replace(" ", "_") + " ") * int(c) for t, c in bigram_counts.items() if c>0 ])

wc2 = WordCloud(width=1200, height=600, background_color="white",
                    collocations=False, max_words=200).generate(bigram_text)
plt.figure(figsize=(12,6))
plt.imshow(wc2, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("wc_out/wordcloud_bigrams.png", dpi=300)
plt.close()


