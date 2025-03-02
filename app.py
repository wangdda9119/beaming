from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scripts.preprocessing import fetch_news, process_news_data

# Flask 앱 생성
app = Flask(__name__)

# 모델 로드
MODEL_PATH = "zeropepsi/robert-ko-news-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 감정 분석 파이프라인 생성
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

@app.route('/')
def home():
    return render_template("index.html")  # 기본 웹 페이지 제공

@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form['query']  # 사용자가 입력한 검색어 가져오기
    if not query:
        return render_template("index.html", error="검색어를 입력하세요.")

    # 뉴스 데이터 가져오기 (네이버 뉴스 API 호출)
    news_items = fetch_news(query)
    processed_news = process_news_data(news_items)

    # 감정 분석 수행
    positive_results = []
    negative_results = []

    for news in processed_news.values():
        full_text = news["title"] + " " + news["summary"]  # 분석할 텍스트
        analysis = sentiment_analyzer(full_text)
        prediction = analysis[0]

        if prediction['label'].lower() == "label_0" and prediction['score'] > 0.8:
            negative_results.append({
                "title": news["title"],
                "summary": news["summary"],
                "link": news["link"]
            })
        elif prediction['label'].lower() == "label_1" and prediction['score'] > 0.95:
            positive_results.append({
                "title": news["title"],
                "summary": news["summary"],
                "link": news["link"]
            })
    
    # 감정 분석 결과를 results.html로 전달
    return render_template("results.html", positive=positive_results, negative=negative_results)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render의 환경 변수에서 포트 가져오기
    app.run(host="0.0.0.0", port=port)
