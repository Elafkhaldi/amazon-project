from flask import Flask, render_template, request
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Initialize T5 model for summarization
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t5_model.to(device)

# Function to classify sentiment based on keywords in the review text
def classify_sentiment(review_text):
    # Simple sentiment analysis based on keywords (you can expand it with a proper model later)
    positive_keywords = ["good", "amazing", "excellent", "great", "love"]
    negative_keywords = ["bad", "terrible", "awful", "horrible", "disappointing"]
    
    review_text_lower = review_text.lower()
    
    # Check for positive keywords
    if any(keyword in review_text_lower for keyword in positive_keywords):
        return "Positive"
    # Check for negative keywords
    elif any(keyword in review_text_lower for keyword in negative_keywords):
        return "Negative"
    else:
        return "Neutral"

def cluster_reviews(reviews_list):
    clusters = {
        1: ["This is great", "Loved it!", "Amazing product"],
        2: ["Bad experience", "Terrible quality"],
    }
    formatted_clusters = []
    for cluster_id, reviews in clusters.items():
        cluster_text = f"Cluster {cluster_id}:\n"
        for review in reviews:
            cluster_text += f"{review}\n"  # Remove the dash and make each review on a new line
        formatted_clusters.append(cluster_text.strip())
    return formatted_clusters

# Function to summarize reviews with longer summary
def summarize_reviews(reviews_list):
    prompt = " ".join(reviews_list)
    inputs = t5_tokenizer("summarize: " + prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    summary_ids = t5_model.generate(inputs['input_ids'], max_length=350, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def index():
    classification_result = None
    clustering_result = None
    summary_result = None

    review_text = ''
    reviews_to_cluster = ''
    reviews_to_summarize = ''

    if request.method == "POST":
        # Sentiment classification
        if "classify" in request.form:
            review_text = request.form.get("review_text")
            if review_text:
                classification_result = classify_sentiment(review_text)

        # Review clustering
        elif "cluster" in request.form:
            reviews_to_cluster = request.form.get("reviews_to_cluster")
            if reviews_to_cluster:
                reviews_list = reviews_to_cluster.strip().split('\n')
                clustering_result = cluster_reviews(reviews_list)

        # Review summarization
        elif "summarize" in request.form:
            reviews_to_summarize = request.form.get("reviews_to_summarize")
            if reviews_to_summarize:
                reviews_list = reviews_to_summarize.strip().split('\n')
                summary_result = summarize_reviews(reviews_list)

    return render_template(
        "index.html",
        classification_result=classification_result,
        clustering_result=clustering_result,
        summary_result=summary_result,
        review_text=review_text,
        reviews_to_cluster=reviews_to_cluster,
        reviews_to_summarize=reviews_to_summarize,
    )

if __name__ == "__main__":
    app.run(debug=True)


