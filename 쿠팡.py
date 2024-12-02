### 네이버 주식사이트에서 종목명과 현재가 가져오기 ###

# 1단계: 필요 라이브러리 설치
import requests
from bs4 import BeautifulSoup

# 2단계: 네이버 금융 페이지에 GET 요청 보내기
response = requests.get('https://finance.naver.com/')

if response.status_code == 200:
    # 3단계: BeautifulSoup 객체 생성
    soup = BeautifulSoup(response.text, 'html.parser')

    # 4단계: <tr> 태그 여러 개 가져오기 위해 select() 사용
    종목정보_list = soup.select('#_topItems1 tr')

    print("거래상위 TOP 종목:")
    # 반복문 사용해서 여러 종목의 종목명과 현재가 가져오기
    for 종목정보 in 종목정보_list:
        # 종목명 가져오기
        종목명 = 종목정보.select_one('th a').text

        # 현재가 가져오기
        현재가 = 종목정보.select_one('td').text

        # 결과 출력
        print(f"종목명: {종목명}, 현재가: {현재가}")
else:
    print(f"페이지 요청 실패: 상태 코드 {response.status_code}")