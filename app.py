import requests
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Step 1: Fetch papers from arXiv
def fetch_arxiv_papers(query, max_results=100):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query={query}&start=0&max_results={max_results}"
    response = requests.get(base_url + search_query)
    return response.text

# Step 2: Parse arXiv data
def parse_arxiv_data(data):
    root = ET.fromstring(data)
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        paper = {
            'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
            'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text,
            'authors': [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')],
            'published': entry.find('{http://www.w3.org/2005/Atom}published').text,
            'link': entry.find('{http://www.w3.org/2005/Atom}id').text,
        }
        papers.append(paper)
    return papers

# Step 3: Clean text
nltk.download('stopwords', quiet=True)
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in text.split() if word not in stop_words])

# Step 4: Create TF-IDF matrix
def create_tfidf_matrix(papers):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([clean_text(paper['summary']) for paper in papers])
    return tfidf_matrix

# Step 5: Get recommendations
def get_recommendations(tfidf_matrix, index, top_n=5):
    cosine_sim = cosine_similarity(tfidf_matrix)
    similar_indices = cosine_sim[index].argsort()[-top_n-1:-1][::-1]
    return similar_indices

def recommend_papers(papers, tfidf_matrix, paper_index):
    similar_indices = get_recommendations(tfidf_matrix, paper_index)
    return [papers[i] for i in similar_indices]

# Keywords to fields mapping
# Keywords to fields mapping (expanded with more fields and broader categories)
keyword_field_map = {
    'artificial intelligence': ['deep learning', 'machine learning', 'natural language processing', 'reinforcement learning'],
    'deep learning': ['artificial intelligence', 'neural networks', 'computer vision', 'natural language processing'],
    'machine learning': ['artificial intelligence', 'data science', 'reinforcement learning', 'natural language processing'],
    'reinforcement learning': ['machine learning', 'deep learning', 'robotics'],
    'natural language processing': ['deep learning', 'artificial intelligence', 'speech recognition', 'computational linguistics'],
    'cybersecurity': ['network security', 'ethical hacking', 'encryption', 'blockchain security'],
    'data science': ['data mining', 'predictive analytics', 'big data', 'statistics', 'business intelligence'],
    'blockchain': ['cryptography', 'distributed systems', 'decentralization', 'smart contracts'],
    'computer vision': ['image recognition', 'deep learning', 'object detection', 'convolutional neural networks'],
    'quantum computing': ['quantum cryptography', 'quantum algorithms', 'superposition', 'quantum entanglement'],
    'robotics': ['robot motion planning', 'robot control', 'autonomous systems', 'sensor fusion'],
    'cloud computing': ['distributed systems', 'virtualization', 'containerization', 'cloud security'],
    'big data': ['hadoop', 'spark', 'data lakes', 'data mining'],
    'bioinformatics': ['genomics', 'proteomics', 'computational biology', 'biostatistics'],
    'renewable energy': ['solar energy', 'wind energy', 'sustainable energy', 'energy storage'],
    'iot': ['embedded systems', 'sensor networks', 'edge computing', 'smart devices'],
    'ethical hacking': ['penetration testing', 'vulnerability assessment', 'social engineering', 'cybersecurity'],
    # Add more fields as needed
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form.get('field')
    papers_data = fetch_arxiv_papers(query)
    papers = parse_arxiv_data(papers_data)

    recommendations = [{'index': i, 'paper': paper} for i, paper in enumerate(papers)]
    return jsonify(recommendations)

@app.route('/recommend_similar', methods=['POST'])
def recommend_similar():
    paper_index = int(request.form.get('paper_index'))
    query = request.form.get('field')
    papers_data = fetch_arxiv_papers(query)
    papers = parse_arxiv_data(papers_data)

    tfidf_matrix = create_tfidf_matrix(papers)
    similar_papers = recommend_papers(papers, tfidf_matrix, paper_index)

    similar_paper_recommendations = [{'index': i, 'paper': paper} for i, paper in enumerate(similar_papers)]
    return jsonify(similar_paper_recommendations)

@app.route('/suggest_fields', methods=['POST'])
def suggest_fields():
    user_input = request.form.get('field').lower()
    suggested_fields = set()

    for keyword, fields in keyword_field_map.items():
        if keyword in user_input or user_input in keyword:  # Partial match, both ways
            suggested_fields.update(fields)

    return jsonify(list(suggested_fields) if suggested_fields else ["No suggestions available"])

if __name__ == '__main__':
    app.run(debug=True)
