import json
import html
import re
import time

# 네이버 API 키 설정
CLIENT_ID = "L1c5IBoQZ2wioGa61pTX"
CLIENT_SECRET = "5JtlPeI4Yn"
MAX_DISPLAY = 100  # 한 번에 가져올 뉴스 개수


def fetch_news(query):
    """ 네이버 API에서 뉴스를 가져오는 함수 """
    import urllib.request
    import urllib.parse
    time.sleep(0.5)  # API 요청 제한 방지를 위한 대기 시간
    
    news_data = []
    encText = urllib.parse.quote(query)
    
    for i in range(5):  # 1페이지 100개, 10번 반복 (총 1000개)
        start = 1 + i * MAX_DISPLAY
        url = f"https://openapi.naver.com/v1/search/news?query={encText}&display={MAX_DISPLAY}&start={start}&sort=sim"
        
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id", CLIENT_ID)
        request.add_header("X-Naver-Client-Secret", CLIENT_SECRET)
        
        try:
            response = urllib.request.urlopen(request)
            if response.getcode() == 200:
                response_body = response.read()
                response_json = json.loads(response_body.decode("utf-8"))
                news_data.extend(response_json.get("items", []))
            else:
                print(f"Error Code: {response.getcode()}")
        except Exception as e:
            print(f"API 호출 중 오류 발생: {e}")
    
    return news_data


def process_news_data(news_items):
    """ 뉴스 제목과 요약을 따로 저장하고, HTML 태그 및 특수문자 제거 """
    processed_news = {}
    for idx, item in enumerate(news_items):
        title = html.unescape(item['title'])
        summary = html.unescape(item['description'])

        # HTML 태그 제거
        title = re.sub(r"<.*?>", "", title)
        summary = re.sub(r"<.*?>", "", summary)

        processed_news[idx] = {
            "title": title,
            "summary": summary,
            "link": item['link']
        }
    
    return processed_news



def analyze_sentiment(news_dict, model, tokenizer):
    """ 감정 분석을 수행하고 긍정/부정 결과를 반환 """
    from transformers import pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    positive_results = []
    negative_results = []
    
    for news in news_dict.values():
        analysis = sentiment_analyzer(news[0])
        prediction = analysis[0]
        
        if prediction['label'].lower() == "negative" and prediction['score'] > 0.8:
            negative_results.append({"text": news[0], "label": "부정"})
        elif prediction['label'].lower() == "positive" and prediction['score'] > 0.95:
            positive_results.append({"text": news[0], "label": "긍정"})
    
    return {"긍정": positive_results, "부정": negative_results}


def save_to_json(data, file_path):
    """ 데이터를 JSON 파일로 저장 """
    import json
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"JSON 파일이 '{file_path}'에 저장되었습니다.")


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    # 검색어 리스트
    search_keywords = ["범죄", "전쟁", "징역", "사고", "폭력", "재난", "사망", "파괴", "분쟁", "테러", "부도", "부패", "봉사", "구조", "산업", "발전", "혁신", "성공", "희망", "협력"]
    
    all_news = []
    for keyword in search_keywords:
        news_items = fetch_news(keyword)
        all_news.extend(news_items)
    
    # 뉴스 데이터 전처리
    processed_news = process_news_data(all_news)
    
    # 감정 분석 모델 및 토크나이저 로드
    model_name = "monologg/koelectra-base-finetuned-nsmc"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 감정 분석 수행
    sentiment_results = analyze_sentiment(processed_news, model, tokenizer)
    
    # JSON 저장
    save_to_json(sentiment_results, "beaming/data/train_over09.json")
