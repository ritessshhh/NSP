import os
import random
import csv
import re

# ==== CONFIGURATION ====
INPUT_TXT        = "txt-ha/all_books.txt"   # Path to input text file
OUTPUT_CSV       = "nsp_questions_ha.csv"    # Output CSV file path
MIN_CONTEXT      = 3
MAX_CONTEXT      = 10
MIN_DIST         = 2
MAX_DIST         = 10
# ========================

def clean_text(s: str) -> str:
    """
    Replace all whitespace (newlines, tabs, multiple spaces)
    with a single space, and strip leading/trailing spaces.
    """
    return ' '.join(s.replace('\n', ' ').replace('\r', ' ').split())

def load_stories(filepath):
    """
    Load raw text and split into individual stories by lines of dashes.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split on any line with 3 or more dashes
    raw_stories = re.split(r'(?m)^[ \t\-]{3,}\s*$', text)
    # Strip and filter out empty
    stories = [s.strip() for s in raw_stories if s.strip()]
    return stories

def split_sentences(text):
    """
    Split a story into sentences by punctuation delimiters (., !, ?).
    """
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if s.strip()]

def generate_nsp_items(sentences, story_id, story_length,
                       context_range=(MIN_CONTEXT, MAX_CONTEXT),
                       distractor_distances=(MIN_DIST, MAX_DIST)):
    """
    Generate NSP questions for one story, including four features:
      - story_length, context_length,
      - distractor_distance, distractor_length

    Returns a list of dicts per question.
    """
    questions = []
    n = len(sentences)

    for context_len in range(context_range[0], context_range[1] + 1):
        # slide window over sentences
        for i in range(0, n - context_len - distractor_distances[0]):
            raw_context = sentences[i:i + context_len]
            raw_true    = sentences[i + context_len]

            # valid distractor indices
            min_d, max_d = distractor_distances
            valid_idxs = list(range(i + context_len + min_d,
                                     min(i + context_len + max_d + 1, n)))
            if not valid_idxs:
                continue
            d_idx = random.choice(valid_idxs)
            raw_distractor = sentences[d_idx]

            # compute features
            dist_distance = d_idx - (i + context_len)
            dist_length   = len(raw_distractor.split())

            # clean all text fields
            context_str  = clean_text(' '.join(raw_context))
            true_str     = clean_text(raw_true)
            distract_str = clean_text(raw_distractor)

            # randomize A/B order
            if random.choice([True, False]):
                A, B, label = true_str, distract_str, 'A'
            else:
                A, B, label = distract_str, true_str, 'B'

            questions.append({
                'story_id': story_id,
                'story_length': story_length,
                'context': context_str,
                'context_length': context_len,
                'distractor_distance': dist_distance,
                'distractor_length': dist_length,
                'option_A': A,
                'option_B': B,
                'label': label
            })

    return questions

if __name__ == '__main__':
    # Load and split into stories
    stories = load_stories(INPUT_TXT)
    print(f"Total Stories = {len(stories)}")
    all_questions = []

    for story_id, story_text in enumerate(stories):
        sents = split_sentences(story_text)
        story_len = len(sents)
        # Skip too-short stories
        if story_len < MIN_CONTEXT + MIN_DIST + 1:
            continue
        qlist = generate_nsp_items(
            sentences=sents,
            story_id=story_id,
            story_length=story_len
        )
        all_questions.extend(qlist)

    # Write all questions to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
    fieldnames = ['story_id', 'story_length', 'context', 'context_length',
                  'distractor_distance', 'distractor_length',
                  'option_A', 'option_B', 'label']
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for q in all_questions:
            writer.writerow(q)

    print(f"Generated {len(all_questions)} NSP questions across {len(stories)} stories → {OUTPUT_CSV}")