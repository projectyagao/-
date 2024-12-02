# import requests #import = 가져온다는뜻 즉 HTML에 요청을 보내 정보를 가져온다는뜻
# from bs4 import BeautifulSoup # BeautifulSoup은 BS4 라는 패키지에 포함된 클래스 중 하나이고 그걸 가져오겠다는 뜻

# res = requests.get('https://finance.naver.com/')  #requests에 요청을 보내 가져온다 네이버경로를 그값을 res에 넣는다.res=respones 와 같은뜻
# html = res.text

# soup = BeautifulSoup(html,'html.parser') #태그 접근에 쉽게 하기
# #soup=BeautifulSoup #beatifulsoup을 처리 해주어야만  html안에 있는 태그들에 접근할수가있다, ex) <p>,<a>등등

# print('응답코드:',res.status_code) #출력 응답코드 : 는 200 정상이다 서버 코드의 종류는 404는 페이지를 찾을수 없음
# #500은 서버오류등 의미합니다

# top = soup.select('#_topitemsI > tr')

# # #class는 .뭐뭐를 써주고 id는 id#을 이용하여 불러온다
# # #필요한 데이터 가져오기 class는 여러개 id는 한개
# # for ]
# top_stocks = soup.select('#_topItems1 > tr')  # '거래상위 TOP 종목' 섹션 테이블의 ID

# print("거래상위 TOP 종목:")
# for stock in top_stocks:
#     name = stock.select_one('th > a').text.strip()  # 종목명
#     price = stock.select_one('td').text.strip()     # 현재가
#     print(f"종목명: {name}, 현재가: {price}")
    
#     import requests

# import requests

# # 2단계: GET 요청 보내기
# response = requests.get('https://finance.naver.com/')
# print(f"응답 상태 코드: {response.status_code}")  # 상태 코드 확인

# # 응답 성공 여부 확인
# if response.status_code == 200:
#     # 응답 내용 확인
#     print(response.text[:1000])  # HTML 일부 출력
# else:
#     print("요청 실패")

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

chrome_options = Options()
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument("--language=ko_KR")

service = Service('C:/Users/r2com/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe')
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.get('https://ev.or.kr/nportal/buySupprt/initSubsidyPaymentCheckAction.do')

try:
    
    driver.execute_script("window.scrollTo(1, document.body.scrollHeight);")
    time.sleep(2)
    
    element = WebDriverWait(driver, 10).until( 
        EC.presence_of_element_located((By.XPATH, '//*[@id="desired-element-xpath"]'))
    )

    # 페이지를 스크롤 (JavaScript 실행)
    driver.execute_script("window.scrollBy(0, 1000);")  # 1000 픽셀 아래로 스크롤

    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="desired-element-xpath"]'))  # 원하는 요소의 XPATH
    )
    time.sleep(2)
    
    query_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="btnLocalCarPrc"]'))
    )
    query_button.click()
    
    time.sleep(2)
    
    screenshot_path = 'element_screenshot.png'
    element.screenshot(screenshot_path)
    print(f'Screenshot saved as {screenshot_path}')

except Exception as e:
    print(f"Error: {e}")

finally:
    print("Press Ctrl+C to close the browser...")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("Closing browser...")
        driver.quit()
