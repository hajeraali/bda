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
def fetch_arxiv_papers(query, max_results=10):
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
keyword_field_map = {
    # Artificial Intelligence & Machine Learning
    'artificial intelligence': ['deep learning', 'machine learning', 'natural language processing', 'reinforcement learning'],
    'deep learning': ['artificial intelligence', 'neural networks', 'computer vision', 'natural language processing'],
    'machine learning': ['artificial intelligence', 'data science', 'reinforcement learning', 'natural language processing'],
    'reinforcement learning': ['machine learning', 'deep learning', 'robotics'],
    'natural language processing': ['deep learning', 'artificial intelligence', 'speech recognition', 'computational linguistics'],
    'neural networks': ['deep learning', 'machine learning', 'artificial intelligence', 'computer vision'],

    # Data Science & Analytics
    'data science': ['data mining', 'predictive analytics', 'big data', 'statistics', 'business intelligence'],
    'big data': ['data science', 'data engineering', 'cloud computing', 'distributed systems'],
    'data mining': ['data science', 'machine learning', 'statistics', 'big data'],
    'predictive analytics': ['data science', 'machine learning', 'statistics', 'business intelligence'],
    'statistics': ['data science', 'predictive analytics', 'mathematics', 'data mining'],
    'business intelligence': ['data science', 'data visualization', 'data engineering', 'predictive analytics'],
    'data engineering': ['big data', 'cloud computing', 'data science', 'databases'],

    # Computer Vision
    'image processing': ['computer vision', 'object detection', 'image recognition', 'pattern recognition'],
    'object detection': ['computer vision', 'image recognition', 'facial recognition', 'pattern recognition'],
    'image recognition': ['object detection', 'facial recognition', 'augmented reality', 'pattern recognition'],
    '3D reconstruction': ['augmented reality', 'virtual reality', 'pattern recognition', 'computer vision'],
    'pattern recognition': ['image recognition', 'object detection', 'facial recognition', 'gesture recognition'],
    'augmented reality': ['virtual reality', '3D reconstruction', 'computer vision', 'gesture recognition'],
    'facial recognition': ['image recognition', 'object detection', 'pattern recognition', 'gesture recognition'],
    'gesture recognition': ['facial recognition', 'augmented reality', 'pattern recognition', 'computer vision'],

    # Cybersecurity
    'cybersecurity': ['network security', 'ethical hacking', 'encryption', 'blockchain security'],
    'network security': ['cybersecurity', 'firewalls', 'encryption', 'penetration testing'],
    'ethical hacking': ['cybersecurity', 'penetration testing', 'vulnerability assessment', 'malware analysis'],
    'encryption': ['cybersecurity', 'network security', 'blockchain security', 'data privacy'],
    'blockchain security': ['blockchain', 'cybersecurity', 'cryptography', 'encryption'],
    'vulnerability assessment': ['ethical hacking', 'penetration testing', 'cybersecurity', 'network security'],
    'penetration testing': ['ethical hacking', 'vulnerability assessment', 'cybersecurity'],
    'malware analysis': ['cybersecurity', 'vulnerability assessment', 'ethical hacking', 'network security'],

    # Robotics
    'robotics': ['autonomous systems', 'control systems', 'robot motion planning', 'reinforcement learning'],
    'autonomous systems': ['robotics', 'control systems', 'artificial intelligence', 'robot motion planning'],
    'control systems': ['robotics', 'autonomous systems', 'industrial automation'],
    'robot motion planning': ['robotics', 'autonomous systems', 'control systems'],
    
    # Natural Sciences
    'bioinformatics': ['genomics', 'biotechnology', 'systems biology', 'molecular biology'],
    'genomics': ['bioinformatics', 'molecular biology', 'genetics', 'biotechnology'],
    'molecular biology': ['genomics', 'bioinformatics', 'biochemistry', 'genetics'],
    'environmental science': ['climate change', 'sustainability', 'ecology', 'conservation'],
    'climate change': ['environmental science', 'sustainability', 'renewable energy', 'global warming'],

    # Social Sciences
    'psychology': ['cognitive science', 'behavioral science', 'neuroscience', 'social psychology'],
    'social media': ['communications', 'digital marketing', 'behavioral science', 'media studies'],
    'economics': ['finance', 'business analytics', 'data science', 'market research'],
    'finance': ['economics', 'investment', 'financial modeling', 'business analytics'],
    
    # Engineering
    'mechanical engineering': ['robotics', 'control systems', 'manufacturing', 'thermodynamics'],
    'electrical engineering': ['electronics', 'power systems', 'control systems', 'renewable energy'],
    'civil engineering': ['structural engineering', 'construction management', 'geotechnical engineering', 'environmental engineering'],
    'chemical engineering': ['process engineering', 'biotechnology', 'materials science', 'thermodynamics'],
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form.get('field')
    papers_data = fetch_arxiv_papers(query)
    papers = parse_arxiv_data(papers_data)
    
    # Return papers along with their indices
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
    
    # Return similar papers with their indices
    similar_paper_recommendations = [{'index': i, 'paper': paper} for i, paper in enumerate(similar_papers)]
    return jsonify(similar_paper_recommendations)

@app.route('/suggest_fields', methods=['POST'])
def suggest_fields():
    user_input = request.form.get('field').lower()
    suggested_fields = set()
    
    for keyword, fields in keyword_field_map.items():
        if keyword in user_input:
            suggested_fields.update(fields)  # Add all related fields

    return jsonify(list(suggested_fields))

if __name__ == '__main__':
    app.run(debug=True)
