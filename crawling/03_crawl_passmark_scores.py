"""PassMark GPU Compute Benchmark(DirectCompute) 점수 차트를 수집한다.

[수집 고지]
- 본 스크립트는 교육/연구 목적의 1회성 데이터 수집용으로 작성되었다.
- 단일 페이지 1회 요청만 수행한다.
- 수집한 원본 데이터는 이 저장소에 포함하지 않으며 재배포하지 않는다.
- 실행 전 대상 사이트의 robots.txt 및 이용약관을 확인할 것.

출력: data/nvidia_gpu_passmark_score.csv
"""

import csv
from collections import defaultdict
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = "https://www.videocardbenchmark.net/directCompute.html"
TOP_N = 1731  # 차트 상위 N개만 사용 (하위권은 노이즈가 커서 제외)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_CSV = DATA_DIR / "nvidia_gpu_passmark_score.csv"


def main():
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code != 200:
        print("요청 실패:", response.status_code)
        return

    soup = BeautifulSoup(response.content, "html.parser")
    raw_scores = []

    for li in soup.select("ul.chartlist > li"):
        a_tag = li.select_one("a")
        if not a_tag:
            continue

        name_tag = a_tag.select_one("span.prdname")
        score_tag = a_tag.select_one("span.count")
        if not (name_tag and score_tag):
            continue

        name = name_tag.get_text(strip=True)
        try:
            score = float(score_tag.get_text(strip=True).replace(",", ""))
        except ValueError:
            continue
        raw_scores.append((name, score))

    top_scores = raw_scores[:TOP_N]

    # 중복된 이름은 평균 처리 (순서 보존)
    name_scores = defaultdict(list)
    for name, score in top_scores:
        name_scores[name].append(score)

    seen = set()
    averaged_scores = []
    for name, _ in top_scores:
        if name not in seen:
            averaged_scores.append((name, sum(name_scores[name]) / len(name_scores[name])))
            seen.add(name)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Name", "Score"])
        writer.writerows(averaged_scores)

    print(f"✅ 저장 완료: {OUTPUT_CSV} ({len(averaged_scores)}개 행 저장됨)")


if __name__ == "__main__":
    main()
