"""01번에서 수집한 목록의 detail_url을 순회하며 GPU 상세 스펙을 수집한다.

[수집 고지]
- 본 스크립트는 교육/연구 목적의 1회성 데이터 수집용으로 작성되었다.
- 요청 간 충분한 대기 시간을 두어 서버에 부담을 주지 않도록 했다.
- 수집한 원본 데이터는 이 저장소에 포함하지 않으며 재배포하지 않는다.
- 실행 전 대상 사이트의 robots.txt 및 이용약관을 확인할 것.

입력: data/nvidia_gpu_specs_by_year.csv
출력: data/nvidia_gpu_spec_techpowerup.csv (중간 저장: data/temp_details/)
"""

import os
import random
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = DATA_DIR / "nvidia_gpu_specs_by_year.csv"
OUTPUT_CSV = DATA_DIR / "nvidia_gpu_spec_techpowerup.csv"
FAILED_CSV = DATA_DIR / "failed_urls.csv"
TEMP_DIR = DATA_DIR / "temp_details"

SAVE_EVERY = 10  # 중간 저장 간격


def make_driver():
    options = Options()
    options.add_argument("--no-sandbox")
    # Selenium 4.6+ 는 Selenium Manager가 드라이버를 자동 관리한다.
    return webdriver.Chrome(options=options)


def crawl_detail_page(url, driver):
    driver.get(url)
    time.sleep(random.uniform(5, 10))  # 서버 부하 방지를 위한 대기

    soup = BeautifulSoup(driver.page_source, "html.parser")

    raw_name = soup.find("h1", class_="gpudb-name")
    model_name = raw_name.text.replace("NVIDIA ", "").strip() if raw_name else ""

    # section.details > div 안의 dl 목록들
    details = {}
    for div in soup.select("section.details div"):
        for dl in div.find_all("dl"):
            dt, dd = dl.find("dt"), dl.find("dd")
            if dt and dd:
                details[dt.text.strip()] = dd.text.strip()

    return {"Model Name": model_name, **details, "URL": url}


def retry_model_name_only(csv_path=OUTPUT_CSV):
    """Model Name이 비어 있는 행만 다시 크롤링해 채운다."""
    df = pd.read_csv(csv_path)

    missing = df[df["Model Name"].isna() | (df["Model Name"].str.strip() == "")]
    if missing.empty:
        print("🎉 빈 Model Name이 없습니다.")
        return

    print(f"🔍 빈 Model Name 재시도 대상: {len(missing)}개")
    driver = make_driver()

    for idx in tqdm(missing.index):
        try:
            url = df.at[idx, "URL"]
            detail_data = crawl_detail_page(url, driver)
            for key, value in detail_data.items():
                if key in df.columns and pd.api.types.is_numeric_dtype(df[key]):
                    value = pd.to_numeric(value, errors="coerce")
                df.at[idx, key] = value
            print(f"✅ {url} → {detail_data.get('Model Name', 'No name')}")
        except Exception as e:
            print(f"❌ 실패: {df.at[idx, 'URL']} → {e}")
            continue
    driver.quit()

    df.to_csv(csv_path, index=False)
    print(f"💾 수정된 결과 저장 완료: {csv_path}")


def main(retry_failed=False):
    if retry_failed and FAILED_CSV.exists():
        print("🔁 실패한 URL 재시도 모드입니다.")
        df = pd.read_csv(FAILED_CSV)
        url_list = df["url"].dropna().unique().tolist()
    else:
        df = pd.read_csv(INPUT_CSV)
        url_list = df["detail_url"].dropna().unique().tolist()

    driver = make_driver()
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    result_data = []
    failed_urls = []

    for idx, url in enumerate(tqdm(url_list)):
        try:
            result_data.append(crawl_detail_page(url, driver))

            if (idx + 1) % SAVE_EVERY == 0:
                pd.DataFrame(result_data).to_csv(
                    TEMP_DIR / f"temp_nvidia_gpu_details_{idx + 1}.csv", index=False
                )
                print(f"💾 {idx + 1}개 저장 완료.")
        except Exception as e:
            print(f"❌ Error at {url}: {e}")
            failed_urls.append(url)
            continue

    driver.quit()

    pd.DataFrame(result_data).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 전체 크롤링 완료: {OUTPUT_CSV}")

    if failed_urls:
        pd.DataFrame({"url": failed_urls}).to_csv(FAILED_CSV, index=False)
        print(f"⚠️ 실패한 URL {len(failed_urls)}개 → {FAILED_CSV} 저장 완료")
    else:
        if FAILED_CSV.exists():
            os.remove(FAILED_CSV)
        print("🎉 실패한 URL 없음!")


if __name__ == "__main__":
    main()
    # 실패한 URL만 재시도: main(retry_failed=True)
    # 빈 Model Name만 재수집: retry_model_name_only()
