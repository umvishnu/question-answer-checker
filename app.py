from flask import Flask, render_template, request, session
import spacy
import string
from nltk.corpus import wordnet
import nltk
from questions import questions
import random
import os
import jinja2
import logging 
logging.basicConfig(level=logging.DEBUG) 

# Download required NLTK data
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


# New Flask initialization code
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

print(f"Template Directory: {TEMPLATE_DIR}")
print(f"Available templates: {os.listdir(TEMPLATE_DIR)}")  # Debug line to show template files

app = Flask(__name__,
           template_folder=TEMPLATE_DIR)

# Enable debug mode and set secret key
app.config['DEBUG'] = True
app.secret_key = 'your_secret_key_here'
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = ''.join(char for char in text if char not in string.punctuation)
    return text.strip()

def are_words_similar(word1, word2, threshold=0.85):
    """Check if two words are semantically similar."""
    # Get synsets for both words
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    
    if not synsets1 or not synsets2:
        return False
    
    # Calculate maximum similarity between any pair of synsets
    max_similarity = max(
        s1.path_similarity(s2) if s1.path_similarity(s2) is not None else 0
        for s1 in synsets1
        for s2 in synsets2
    )
    
    return max_similarity >= threshold

def calculate_word_similarity(text1, text2):
    # Clean texts
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    # Split into words
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    # Find common words
    common_words = words1.intersection(words2)
    
    # Calculate similarity
    if len(words1) == 0:
        return 0
    
    return (len(common_words) / len(words1)) * 100

def calculate_meaning_similarity(text1, text2):
    # Clean texts
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    # Process with spaCy
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    # Get meaningful words (excluding stopwords and punctuation)
    words1 = [token for token in doc1 if not token.is_stop and token.is_alpha]
    words2 = [token for token in doc2 if not token.is_stop and token.is_alpha]
    
    if not words1 or not words2:
        return 0
    
    # Calculate semantic similarity for each word pair
    total_similarity = 0
    word_count = 0
    
    for word1 in words1:
        word_similarities = []
        
        # Get word vectors and check similarity
        for word2 in words2:
            # Use spaCy's word vectors for similarity
            similarity = word1.similarity(word2)
            word_similarities.append(similarity)
        
        if word_similarities:
            # Take the best match for this word
            total_similarity += max(word_similarities)
            word_count += 1
    
    if word_count == 0:
        return 0
    
    # Calculate average similarity and convert to percentage
    meaning_score = (total_similarity / word_count) * 100
    
    # Adjust score based on length difference penalty
    length_diff = abs(len(words1) - len(words2)) / max(len(words1), len(words2))
    length_penalty = 1 - (length_diff * 0.5)  # Less severe penalty
    
    final_score = meaning_score * length_penalty
    
    return min(100, final_score)  # Cap at 100%

def analyze_answer(user_answer, correct_answer):
    """Provide detailed analysis of the answer."""
    doc1 = nlp(clean_text(user_answer))
    doc2 = nlp(clean_text(correct_answer))
    
    analysis = {
        'missing_key_words': [],
        'extra_words': [],
        'similar_words': [],
        'highlighted_correct': correct_answer,
        'highlighted_user': user_answer,
        'key_concepts': []
    }
    
    # Find key words in correct answer
    key_words = [token.text for token in doc2 if not token.is_stop and token.is_alpha]
    user_words = [token.text for token in doc1 if not token.is_stop and token.is_alpha]
    
    # Track used words to avoid duplicates
    used_user_words = set()
    used_correct_words = set()
    
    # Find similar words and missing words
    for word in key_words:
        found = False
        for user_word in user_words:
            if user_word not in used_user_words and are_words_similar(word, user_word):
                analysis['similar_words'].append((word, user_word))
                used_user_words.add(user_word)
                used_correct_words.add(word)
                found = True
                break
        if not found:
            analysis['missing_key_words'].append(word)
    
    # Find extra words
    for word in user_words:
        if word not in used_user_words and not any(are_words_similar(word, key_word) for key_word in key_words):
            analysis['extra_words'].append(word)
    
    # Highlight missing words in correct answer
    highlighted_correct = correct_answer
    for word in analysis['missing_key_words']:
        highlighted_correct = highlighted_correct.replace(
            word,
            f'<span class="missing-word" title="Missing in your answer">{word}</span>'
        )
    analysis['highlighted_correct'] = highlighted_correct
    
    # Highlight extra words in user answer
    highlighted_user = user_answer
    for word in analysis['extra_words']:
        highlighted_user = highlighted_user.replace(
            word,
            f'<span class="extra-word" title="Not found in correct answer">{word}</span>'
        )
    analysis['highlighted_user'] = highlighted_user
    
    # Extract key concepts (nouns and verbs)
    key_concepts = [token.text for token in doc2 if token.pos_ in ['NOUN', 'VERB'] and not token.is_stop]
    analysis['key_concepts'] = key_concepts
    
    return analysis

def get_feedback(score):
    if score >= 90:
        return {
            "message": "Excellent! 🎉",
            "class": "excellent",
            "details": "Your answer matches perfectly with the correct answer!"
        }
    elif score >= 80:
        return {
            "message": "Great Job! 👏",
            "class": "good",
            "details": "Your answer shows good understanding of the concept!"
        }
    elif score >= 70:
        return {
            "message": "Good Progress! 💪",
            "class": "average",
            "details": "You're on the right track, but there's room for improvement."
        }
    else:
        return {
            "message": "Keep Learning! 📚",
            "class": "needs-work",
            "details": "Review the correct answer and try again."
        }

@app.route('/')
def home():
    if 'current_question' not in session:
        session['current_question'] = random.choice(questions)
    return render_template('index.html', question=session['current_question'])

@app.route('/check', methods=['POST'])
def check():
    user_answer = request.form.get('user_answer', '')
    
    current_question = session.get('current_question')
    if not current_question:
        current_question = random.choice(questions)
        session['current_question'] = current_question
    
    correct_answer = current_question['correct_answer']
    
    # Calculate similarities
    word_similarity = calculate_word_similarity(correct_answer, user_answer)
    meaning_similarity = calculate_meaning_similarity(correct_answer, user_answer)
    
    # Adjust weights based on similarity scores
    if meaning_similarity > word_similarity:
        total_score = (word_similarity * 0.3 + meaning_similarity * 0.7)
    else:
        total_score = (word_similarity * 0.4 + meaning_similarity * 0.6)
    
    # Get feedback
    word_feedback = get_feedback(word_similarity)
    meaning_feedback = get_feedback(meaning_similarity)
    total_feedback = get_feedback(total_score)
    
    # Analyze answer
    analysis = analyze_answer(user_answer, correct_answer)
    
    return render_template('index.html',
                         question=current_question,
                         user_answer=user_answer,
                         correct_answer=correct_answer,
                         word_score=round(word_similarity, 2),
                         meaning_score=round(meaning_similarity, 2),
                         total_score=round(total_score, 2),
                         word_feedback=word_feedback,
                         meaning_feedback=meaning_feedback,
                         total_feedback=total_feedback,
                         analysis=analysis,
                         show_results=True)

@app.route('/next-question', methods=['POST'])
def next_question():
    current = session.get('current_question')
    available_questions = [q for q in questions if q['id'] != current['id']]
    if available_questions:
        session['current_question'] = random.choice(available_questions)
    else:
        session['current_question'] = random.choice(questions)
    return render_template('index.html', question=session['current_question'])

# Add these error handlers here
@app.errorhandler(jinja2.exceptions.TemplateNotFound)
def template_not_found(e):
    return f"Template not found: {e.name}. Current template folder: {app.template_folder}", 500

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return f"Server Error: {error}. Template folder: {app.template_folder}", 500
    
if __name__ == '__main__':
    app.run(debug=True)
