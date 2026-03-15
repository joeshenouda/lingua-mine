#!/usr/bin/env python3
"""
generate_full_biographies_tokenized.py

Generates N full-sentence biographies, saves one per line in a text file,
prints 3 examples, and computes average token count using TikTokenTokenizer.
"""

import random
import datetime
import sys
import os
import json
from pathlib import Path

import sys, os
# Ensure parent directory (which contains 'lingua') is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Import TikTokenTokenizer from your local Lingua installation
from lingua.tokenizer import TikTokenTokenizer

# ----------------------------
# TEMPLATE DICTIONARY
# ----------------------------
# Import template dictionary
from templates import TEMPLATES


# ----------------------------
# LOAD DATA FILES
# ----------------------------

def load_list(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

FIRST_NAMES = load_list("random_first_names.txt")
MIDDLE_NAMES = load_list("random_middle_names.txt")
LAST_NAMES = load_list("random_last_names_unique.txt")
CITIES = load_list("cities_states.txt")
UNIVERSITIES = load_list("universities.txt")
MAJORS = load_list("majors.txt")
COMPANIES = load_list("companies.txt")

# ----------------------------
# HELPERS
# ----------------------------

def random_birthdate():
    start = datetime.date(1900, 1, 1)
    end = datetime.date(2025, 12, 31)
    delta = (end - start).days
    d = start + datetime.timedelta(days=random.randint(0, delta))
    return d.strftime("%B %-d, %Y") if sys.platform != "win32" else d.strftime("%B %#d, %Y")

# ----------------------------
# BIOGRAPHY GENERATOR
# ----------------------------

def generate_biographies(N=10, seed=42):
    if seed is not None:
        random.seed(seed)
    used_names, biographies = set(), []
    full_names = []
    birth_cities, current_cities =[], []
    birthdates, universities, majors, companies = [], [], [], []
    pronouns = []
    while len(biographies) < N:
        first, middle, last = random.choice(FIRST_NAMES), random.choice(MIDDLE_NAMES), random.choice(LAST_NAMES)
        full_name = f"{first} {middle} {last}"
        if full_name in used_names:
            continue
        used_names.add(full_name)
        full_names.append(full_name)

        birth_city, current_city = random.choice(CITIES), random.choice(CITIES)
        birth_cities.append(birth_city)
        current_cities.append(current_city)

        birthdate, university, major, company = random_birthdate(), random.choice(UNIVERSITIES), random.choice(MAJORS), random.choice(COMPANIES)
        birthdates.append(birthdate)
        universities.append(university)
        majors.append(major)
        companies.append(company)
        # Choose pronoun randomly for subsequent sentences
        pronoun = random.choice(["He", "She", "It"])
        pronouns.append(pronoun)
        
        sentences = [
            random.choice(TEMPLATES["city_of_birth"]).replace("[X]", full_name).replace("[Y]", birth_city),
            random.choice(TEMPLATES["birthdate"]).replace("[X]", pronoun).replace("[Y]", birthdate),
            random.choice(TEMPLATES["university"]).replace("[X]", pronoun).replace("[Y]", university),
            random.choice(TEMPLATES["major"]).replace("[X]", pronoun).replace("[Y]", major),
            random.choice(TEMPLATES["company"]).replace("[X]", pronoun).replace("[Y]", company),
            random.choice(TEMPLATES["current_location"]).replace("[X]", pronoun).replace("[Y]", current_city)
        ]
        biographies.append(" ".join(sentences))
    return biographies, full_names, birth_cities, current_cities, birthdates, universities, majors, companies, pronouns

# ----------------------------
# MAIN EXECUTION
# ----------------------------

if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    bios, full_names, birth_cities, current_cities, birthdates, universities, majors, companies, pronouns = generate_biographies(N)

    bios_paraphrased = []
    for i,bio in enumerate(bios):
        sentences = [
            random.choice(TEMPLATES["city_of_birth"]).replace("[X]", full_names[i]).replace("[Y]", birth_cities[i]),
            random.choice(TEMPLATES["birthdate"]).replace("[X]", pronouns[i]).replace("[Y]", birthdates[i]),
            random.choice(TEMPLATES["university"]).replace("[X]", pronouns[i]).replace("[Y]", universities[i]),
            random.choice(TEMPLATES["major"]).replace("[X]", pronouns[i]).replace("[Y]", majors[i]),
            random.choice(TEMPLATES["company"]).replace("[X]", pronouns[i]).replace("[Y]", companies[i]),
            random.choice(TEMPLATES["current_location"]).replace("[X]", pronouns[i]).replace("[Y]", current_cities[i])
        ]
        bios_paraphrased.append(" ".join(sentences))

    output_path = "biographies_full_sentences_test.txt"
    output_path_para = "biographies_paraphrased_test.txt"

    print("\n🧠 Example Biographies:\n")
    for bio in bios[:3]:
        print(f"- {bio}\n")

    print("\n🧠 Example Paraphrased Biographies:\n")
    for bio in bios_paraphrased[:3]:
        print(f"- {bios_paraphrased}\n")


    # Initialize TikToken tokenizer
    model_path = "/Users/jsheno/experiments/lingua/tiktoken-tokenizer/9b5ad71b2ce5302211f9c61530b329a4922fc6a4"  # update this path
    tokenizer = TikTokenTokenizer(model_path=model_path)

    token_counts = []
    with open(output_path, "w", encoding="utf-8") as f:
        for bio in bios:
            f.write(bio + "\n")
            tokens = tokenizer.encode(bio, add_bos=False, add_eos=False)
            token_counts.append(len(tokens))

    with open(output_path_para, "w", encoding="utf-8") as f:
        for bio in bios_paraphrased:
            f.write(bio + "\n")

    avg_tokens = sum(token_counts) / len(token_counts)
    print(f"✅ Generated {N} biographies and saved to {output_path}")
    print(f"✅ Generated {N} paraphrased biographies and saved to {output_path_para}")
    print(f"🔢 Average tokens per biography: {avg_tokens:.2f}")
