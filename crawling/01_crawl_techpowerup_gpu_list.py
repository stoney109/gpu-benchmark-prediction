"""TechPowerUp GPU Database에서 NVIDIA GPU 목록(연도별)을 수집한다.

[수집 고지]
- 본 스크립트는 교육/연구 목적의 1회성 데이터 수집용으로 작성되었다.
- 요청 간 충분한 대기 시간을 두어 서버에 부담을 주지 않도록 했다.
- 수집한 원본 데이터는 이 저장소에 포함하지 않으며 재배포하지 않는다.
- 실행 전 대상 사이트의 robots.txt 및 이용약관을 확인할 것.

출력: data/nvidia_gpu_specs_by_year.csv
"""

import csv
import random
import time
from pathlib import Path

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

BASE_URL = "https://www.techpowerup.com"
URL_TEMPLATE = BASE_URL + "/gpu-specs/?mfgr=NVIDIA&released={year}&sort=name"
START_YEAR, END_YEAR = 1999, 2024

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_CSV = DATA_DIR / "nvidia_gpu_specs_by_year.csv"


def main():
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Selenium 4.6+ 는 Selenium Manager가 드라이버를 자동 관리한다.
    driver = webdriver.Chrome(options=options)

    gpu_list = []
    try:
        for year in range(START_YEAR, END_YEAR + 1):
            driver.get(URL_TEMPLATE.format(year=year))
            time.sleep(random.uniform(15, 30))  # 서버 부하 방지를 위한 대기

            soup = BeautifulSoup(driver.page_source, "html.parser")
            rows = soup.select("tbody tr")

            year_gpu_count = 0
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 8:
                    continue

                product_tag = cols[0].find("a")
                gpu_list.append([
                    product_tag.text.strip(),          # Product Name
                    cols[1].text.strip(),              # GPU Chip
                    str(year),                         # Released (요청 연도로 고정)
                    cols[3].text.strip(),              # Bus
                    cols[4].text.strip(),              # Memory
                    cols[5].text.strip(),              # GPU clock
                    cols[6].text.strip(),              # Memory clock
                    cols[7].text.strip(),              # Shaders / TMUs / ROPs
                    BASE_URL + product_tag["href"],    # detail_url
                ])
                year_gpu_count += 1

            print(f"크롤링 중: {year}년 - {year_gpu_count}개 항목")
    finally:
        driver.quit()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Product Name", "GPU Chip", "Released", "Bus", "Memory",
            "GPU clock", "Memory clock", "Shaders / TMUs / ROPs", "detail_url",
        ])
        writer.writerows(gpu_list)

    print(f"✅ CSV 저장 완료! 총 {len(gpu_list)}개 항목 수집됨 → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
