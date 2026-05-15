"""
앰버 7대 플랫폼 통합 AI 지배인 (Streamlit) v4
v3 대비 추가:
- 🐛 차트 버그 수정 (NaN 플랫폼 빈 행 제거)
- 1. VOC → 액션 아이템 변환기 (부서 자동 분류 + 칸반)
- 2. 재방문/추천 의도 분석 (별점 ≠ 추천의도 갭 감지)
- 3. 연관 분석 (카테고리 동시 출현 히트맵)
- 4. 응대 SLA 추적 (작성일 → 답변일 시간)
- 5. Before/After 비교 (두 기간 만족도 비교)
- 6. 방치 알람 (5일 이상 미답변 부정 리뷰)
- 7. 직원 이름 언급 트래커
- 8. 요일/시즌 패턴 분석
"""

import streamlit as st
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import json
import traceback
import re
import difflib
from datetime import datetime, timedelta
from collections import Counter
import io

# ─────────────────────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="앰버 7대 플랫폼 통합 관리",
    layout="wide",
    page_icon="🏨",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# 1. Firebase 초기화
# ─────────────────────────────────────────────────────────
if not firebase_admin._apps:
    try:
        if "FIREBASE_JSON" in st.secrets:
            key_dict = json.loads(st.secrets["FIREBASE_JSON"])
            cred = credentials.Certificate(key_dict)
        else:
            cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Firebase 키 에러: {e}")
        st.stop()

db = firestore.client()

# ─────────────────────────────────────────────────────────
# 2. Gemini 설정
# ─────────────────────────────────────────────────────────
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        st.error(
            "⚠️ Gemini API 키가 등록되지 않았습니다.\n\n"
            "Streamlit Cloud → 앱 Settings → Secrets 에서 "
            "`GOOGLE_API_KEY = \"새_키_값\"` 형식으로 등록해 주세요."
        )
        st.stop()
except Exception as e:
    st.error(f"Gemini API 키 설정 실패: {e}")
    st.stop()

PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-flash-latest"]

ADMIN_URLS = {
    "네이버(Naver)": "https://new.smartplace.naver.com/bizes/place/6736761/reviews?bookingBusinessId=953025&menu=visitor",
    "아고다(Agoda)": "https://ycs.agoda.com/ko-kr/Reviews/Index",
    "부킹닷컴(Booking.com)": "https://admin.booking.com/hotel/hoteladmin/extranet_ng/manage/reviews.html",
    "익스피디아(Expedia)": "https://www.expediapartnercentral.com/",
    "야놀자": "https://partner.yanolja.com/review",
    "여기어때": "https://partner.goodchoice.kr/reservations/reservation-list",
    "트립닷컴(Trip.com)": "https://ebooking.trip.com/pro-web/review",
    "마이리얼트립(MyRealTrip)": "https://partner.myrealtrip.com/reviews/accommodation",
    "트립어드바이저(TripAdvisor)": "https://www.tripadvisor.com/Owners",
    "구글(Google)": "https://business.google.com/reviews",
}

# ─────────────────────────────────────────────────────────
# 3. 호텔 도메인 카테고리 + 부서 매핑
# ─────────────────────────────────────────────────────────
HOTEL_CATEGORIES = {
    "조식": ["조식", "아침", "뷔페", "조찬", "브런치", "breakfast"],
    "객실": ["객실", "방", "룸", "베드", "침대", "침구", "이불", "베개"],
    "직원/서비스": ["직원", "스태프", "서비스", "프론트", "안내", "응대", "친절", "매니저"],
    "청결": ["청결", "깨끗", "더럽", "청소", "위생", "먼지", "곰팡이", "냄새"],
    "수영장/풀": ["수영장", "풀", "야외풀", "실내풀", "인피니티"],
    "뷰/전망": ["뷰", "전망", "경치", "바다", "오션", "view"],
    "위치/접근성": ["위치", "교통", "공항", "버스", "지하철", "주차", "접근"],
    "시설": ["시설", "사우나", "스파", "헬스", "피트니스", "라운지", "바", "레스토랑"],
    "가성비/가격": ["가성비", "가격", "비싸", "저렴", "값어치", "비용", "할인"],
    "소음/방음": ["소음", "시끄럽", "조용", "방음", "층간"],
    "온수/욕실": ["온수", "샤워", "욕실", "화장실", "수압", "물"],
    "와이파이": ["와이파이", "wifi", "인터넷"],
}

CATEGORY_TO_DEPT = {
    "조식": "F&B팀",
    "객실": "하우스키핑",
    "직원/서비스": "프론트/서비스팀",
    "청결": "하우스키핑",
    "수영장/풀": "시설관리",
    "뷰/전망": "(고정 요소)",
    "위치/접근성": "(고정 요소)",
    "시설": "시설관리",
    "가성비/가격": "경영진",
    "소음/방음": "시설관리",
    "온수/욕실": "시설관리",
    "와이파이": "IT/시설관리",
}


# ─────────────────────────────────────────────────────────
# 4. 데이터 함수
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_reviews():
    try:
        docs = (
            db.collection("reviews")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .stream()
        )
        data = []
        for doc in docs:
            d = doc.to_dict()
            d["id"] = doc.id
            d.setdefault("platform", "기타")
            d.setdefault("content", "(내용 없음)")
            d.setdefault("status", "대기중")
            d.setdefault("date", "")
            d.setdefault("ai_reply", "")
            d.setdefault("title", "")
            d.setdefault("positive", "")
            d.setdefault("negative", "")
            d.setdefault("score", "")
            d.setdefault("user", "")
            d.setdefault("has_reply", False)
            d.setdefault("satisfaction_tags", "")
            d.setdefault("post_time", "")
            d.setdefault("post_date", "")
            d.setdefault("travel_date", "")
            d.setdefault("stay_period", "")
            d.setdefault("country", "")
            d.setdefault("room_type", "")
            d.setdefault("traveler_type", "")
            d.setdefault("booking_id", "")
            d.setdefault("review_id", "")
            d.setdefault("owner_reply", "")
            d.setdefault("ai_categories", [])
            d.setdefault("final_reply", "")
            d.setdefault("completed_at", "")
            d.setdefault("recommend_intent", "")
            data.append(d)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"데이터 로딩 실패: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_benchmark():
    try:
        doc = db.collection("config").document("benchmark").get()
        if doc.exists:
            return doc.to_dict()
        return {"jeju_avg": None, "competitor_avg": None, "note": ""}
    except:
        return {"jeju_avg": None, "competitor_avg": None, "note": ""}


@st.cache_data(ttl=300)
def get_templates():
    try:
        docs = db.collection("reply_templates").stream()
        return [{"id": d.id, **d.to_dict()} for d in docs]
    except:
        return []


@st.cache_data(ttl=60)
def get_action_items():
    try:
        docs = db.collection("action_items").order_by(
            "created_at", direction=firestore.Query.DESCENDING
        ).stream()
        return [{"id": d.id, **d.to_dict()} for d in docs]
    except:
        return []


@st.cache_data(ttl=300)
def get_staff_names():
    try:
        doc = db.collection("config").document("staff_names").get()
        if doc.exists:
            return doc.to_dict().get("names", [])
        return []
    except:
        return []


def normalize_status(row):
    s = row.get("status", "대기중")
    if s == "처리완료":
        return "처리완료"
    if s == "답변완료" or row.get("has_reply"):
        return "답변완료"
    return "대기중"


def score_to_float(s):
    try:
        return float(s)
    except:
        return None


def score_to_pct(score_val, platform):
    if score_val is None or pd.isna(score_val):
        return None
    if platform in ("구글(Google)", "트립어드바이저(TripAdvisor)"):
        return score_val * 20
    else:
        return score_val * 10


def parse_date(date_str):
    if not date_str or pd.isna(date_str):
        return None
    s = str(date_str).strip()
    formats = [
        "%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d",
        "%Y년 %m월 %d일", "%Y년%m월%d일",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y", "%d %B %Y", "%d %b %Y",
    ]
    for f in formats:
        try:
            return datetime.strptime(s[:len(f)+5], f)
        except:
            continue
    m = re.search(r"(\d{4})[년\.\-/](\d{1,2})[월\.\-/](\d{1,2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except:
            pass
    return None


def call_gemini(prompt, model_name=None):
    models_to_try = [model_name] if model_name else [PRIMARY_MODEL] + FALLBACK_MODELS
    last_error = None
    for m in models_to_try:
        try:
            model = genai.GenerativeModel(m)
            res = model.generate_content(prompt)
            return res.text
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"모든 Gemini 모델 호출 실패. 마지막 오류: {last_error}")


def build_review_text(row):
    parts = []
    if row.get("title"):
        parts.append(f"제목: {row['title']}")
    if row.get("positive"):
        parts.append(f"좋았던 점: {row['positive']}")
    if row.get("negative"):
        parts.append(f"아쉬운 점: {row['negative']}")
    if not parts:
        parts.append(row.get("content", ""))
    return "\n".join(parts)


def build_reply_prompt(row, similar_templates=None):
    review_text = build_review_text(row)
    score = row.get("score", "")
    user = row.get("user", "")
    platform = row.get("platform", "")

    score_val = None
    try:
        score_val = float(score)
    except:
        pass

    max_score = 5 if platform in ("구글(Google)", "트립어드바이저(TripAdvisor)") else 10
    score_pct = (score_val / max_score) * 100 if score_val is not None else None

    score_guide = ""
    score_display = ""
    if score_val is not None:
        score_display = f"{score}/{max_score}점"
        if score_pct <= 50:
            score_guide = "점수가 매우 낮은 리뷰입니다. 변명하지 말고 진심으로 미안한 마음을 담아 응답하세요. 단, 과장된 사죄 표현은 절대 쓰지 마세요. 구체적으로 어떤 부분을 어떻게 살펴보겠다는 한 마디를 포함해주세요."
        elif score_pct < 80:
            score_guide = "점수가 중간 정도입니다. 좋았던 점은 진심으로 감사하고, 아쉬운 점은 담백하게 받아들이세요. 변명 없이."
        else:
            score_guide = "점수가 높은 리뷰입니다. 과하지 않게 감사의 마음을 표현하세요. 영업멘트로 흐르지 않도록 주의."

    template_section = ""
    if similar_templates:
        template_section = "\n[참고할 만한 과거 좋은 답변 예시]\n"
        for t in similar_templates[:2]:
            template_section += f"- {t.get('reply', '')[:300]}\n"

    return f"""당신은 제주 '엠버퓨어힐 호텔앤리조트'의 지배인입니다.
{platform} 에 올라온 아래 리뷰에 대한 답변을 작성해 주세요.

[톤]
- '정중하지만 거리감 없는' 느낌. 단골에게 답하듯 따뜻하게.
- AI가 쓴 듯한 격식 차린 클리셰는 절대 금지.

[금지 표현]
- "죄송한 마음 금할 길이 없습니다", "사죄의 말씀을 올립니다"
- "고객님께 막중한 책임감을 느낍니다"
- "최고의 서비스로 보답하겠습니다"
- "더욱 노력하는 호텔이 되도록 하겠습니다"
- "심심한 사과의 말씀", "깊이 반성하고 있습니다"
- "다시 한번 진심으로 사과드립니다"
- "고객님의 소중한 의견"
- "성원에 보답하기 위해", "기대에 부응할 수 있도록"
- "불편을 끼쳐드려 대단히 죄송합니다"

[작성 지침]
- 첫 문장: "안녕하세요, 엠버퓨어힐입니다."
- 작성자 이름이 있으면 자연스럽게 한 번 호명.
- 리뷰에서 구체적으로 언급된 부분을 답변에도 한두 가지 짚어주세요.
- 분량: 200~350자. 이모티콘: 0~1개.

[점수 가이드]
{score_guide}
{template_section}
[답변할 리뷰]
플랫폼: {platform}
작성자: {user or "(이름 없음)"}
점수: {score_display or "(없음)"}
{review_text}

답변 본문만 출력합니다."""


def build_category_tagging_prompt(row):
    review_text = build_review_text(row)
    categories = list(HOTEL_CATEGORIES.keys())
    return f"""다음 호텔 리뷰가 어떤 주제를 다루고 있는지 분류해주세요.

[가능한 카테고리]
{", ".join(categories)}

[리뷰]
{review_text}

규칙:
- 해당하는 카테고리만 골라서 JSON 배열로 반환
- 여러 개 가능. 해당 없으면 []
- 다른 설명 없이 JSON 배열만 출력

예: ["조식", "직원/서비스"]
"""


def build_recommend_intent_prompt(row):
    review_text = build_review_text(row)
    return f"""다음 호텔 리뷰에서 작성자의 '재방문/추천 의도'를 분류해주세요.

[리뷰]
{review_text}

[분류 기준]
- "positive": 재방문 의사가 명확하거나 다른 사람에게 추천하겠다는 표현 ("또 가고 싶다", "추천한다", "다음에도", "재방문")
- "negative": 재방문 안 하겠다는 표현 ("다시는 안 간다", "비추", "후회한다", "추천 안 함")
- "neutral": 재방문/추천에 대한 명확한 언급 없음

JSON 형식으로만 답하세요. 다른 설명 없이.
예: {{"intent": "positive", "reason": "또 묵고 싶다고 명시"}}
"""


def update_review(doc_id, fields):
    db.collection("reviews").document(doc_id).update(fields)


def detect_categories_keyword(text):
    if not text:
        return []
    found = []
    text_lower = text.lower()
    for cat, keywords in HOTEL_CATEGORIES.items():
        if any(kw in text_lower for kw in keywords):
            found.append(cat)
    return found


def diff_ratio(original, edited):
    if not original or not edited:
        return 0
    o, e = original.strip(), edited.strip()
    if o == e:
        return 0
    sm = difflib.SequenceMatcher(None, o, e)
    return 1 - sm.ratio()


def get_response_hours(row):
    if not row.get("completed_at"):
        return None
    review_dt = row.get("date_dt")
    if review_dt is None or pd.isna(review_dt):
        return None
    try:
        complete_dt = datetime.fromisoformat(row["completed_at"])
        delta = complete_dt - review_dt
        hours = delta.total_seconds() / 3600
        return hours if hours >= 0 else None
    except:
        return None


# ─────────────────────────────────────────────────────────
# 5. 데이터 로딩 & 전처리
# ─────────────────────────────────────────────────────────
st.title("🏨 앰버 통합 리뷰 AI 지배인")

df = get_reviews()

if df.empty:
    st.info("아직 수집된 리뷰가 없습니다. 크롤러를 먼저 실행해 주세요.")
    st.stop()

df["status_norm"] = df.apply(normalize_status, axis=1)
df["score_num"] = df["score"].apply(score_to_float)
df["score_pct"] = df.apply(lambda r: score_to_pct(r["score_num"], r["platform"]), axis=1)
df["date_dt"] = df["date"].apply(parse_date)
df["full_text"] = df.apply(
    lambda r: " ".join(
        str(r.get(f, "")) for f in ["title", "content", "positive", "negative"]
    ),
    axis=1,
)
df["categories_auto"] = df["full_text"].apply(detect_categories_keyword)
df["categories_final"] = df.apply(
    lambda r: r["ai_categories"] if r["ai_categories"] else r["categories_auto"], axis=1
)
df["response_hours"] = df.apply(get_response_hours, axis=1)

# ─────────────────────────────────────────────────────────
# 6. 사이드바
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 전체 현황")
    total = len(df)
    waiting = len(df[df["status_norm"] == "대기중"])
    done = len(df[df["status_norm"] == "처리완료"])
    st.metric("전체 리뷰", f"{total:,}개")
    col_a, col_b = st.columns(2)
    col_a.metric("⏳ 대기", f"{waiting}")
    col_b.metric("🎯 완료", f"{done}")

    avg_all = df["score_pct"].dropna().mean()
    st.metric(
        "평균 만족도",
        f"{avg_all:.1f}%" if not pd.isna(avg_all) else "—",
    )

    now = datetime.now()
    stale_negative = df[
        (df["status_norm"] == "대기중")
        & (df["score_pct"].notna())
        & (df["score_pct"] <= 50)
        & (df["date_dt"].notna())
        & ((now - df["date_dt"]).dt.days >= 5)
    ]
    if len(stale_negative) > 0:
        st.error(f"🚨 5일+ 방치된 부정 리뷰: **{len(stale_negative)}개**")

    responded = df[df["response_hours"].notna()]
    if len(responded) > 0:
        within_24h = len(responded[responded["response_hours"] <= 24])
        sla_rate = within_24h / len(responded) * 100
        st.metric(
            "24h 내 응대율",
            f"{sla_rate:.0f}%",
            help="처리완료된 리뷰 중 24시간 내 답변한 비율",
        )

    st.markdown("---")
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"🕐 로딩 시각: {datetime.now().strftime('%H:%M:%S')}")


# ─────────────────────────────────────────────────────────
# 7. 메인 탭 구조
# ─────────────────────────────────────────────────────────
main_tabs = st.tabs([
    "📊 대시보드",
    "🎯 액션 아이템",
    "🔍 키워드 분석",
    "🔗 연관 분석",
    "📝 리뷰 관리",
    "🤖 AI 분석",
    "📈 운영 인사이트",
    "⚙️ 운영 도구",
])


# ═════════════════════════════════════════════════════════
# TAB 1: 대시보드 (🐛 버그 수정)
# ═════════════════════════════════════════════════════════
with main_tabs[0]:
    st.header("📊 대시보드")

    period_col1, period_col2 = st.columns([3, 1])
    with period_col2:
        period = st.selectbox(
            "기간",
            ["전체", "최근 30일", "최근 90일", "최근 1년", "올해"],
            key="dash_period",
        )

    df_dash = df.copy()
    if period == "최근 30일":
        df_dash = df_dash[df_dash["date_dt"].notna() & (df_dash["date_dt"] >= now - timedelta(days=30))]
    elif period == "최근 90일":
        df_dash = df_dash[df_dash["date_dt"].notna() & (df_dash["date_dt"] >= now - timedelta(days=90))]
    elif period == "최근 1년":
        df_dash = df_dash[df_dash["date_dt"].notna() & (df_dash["date_dt"] >= now - timedelta(days=365))]
    elif period == "올해":
        df_dash = df_dash[df_dash["date_dt"].notna() & (df_dash["date_dt"].dt.year == now.year)]

    with period_col1:
        st.caption(f"분석 대상: **{len(df_dash):,}개** 리뷰 ({period})")

    # KPI 카드
    st.subheader("📈 이번 달 vs 지난 달")
    this_month_start = datetime(now.year, now.month, 1)
    last_month_end = this_month_start - timedelta(days=1)
    last_month_start = datetime(last_month_end.year, last_month_end.month, 1)

    df_this = df[df["date_dt"].notna() & (df["date_dt"] >= this_month_start)]
    df_last = df[
        df["date_dt"].notna()
        & (df["date_dt"] >= last_month_start)
        & (df["date_dt"] < this_month_start)
    ]

    avg_this = df_this["score_pct"].dropna().mean()
    avg_last = df_last["score_pct"].dropna().mean()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("이번 달 리뷰 수", f"{len(df_this)}개", delta=f"{len(df_this) - len(df_last):+d} vs 지난달")
    if not pd.isna(avg_this):
        delta = f"{avg_this - avg_last:+.1f}%p" if not pd.isna(avg_last) else None
        k2.metric("이번 달 만족도", f"{avg_this:.1f}%", delta=delta)
    else:
        k2.metric("이번 달 만족도", "—")

    low_this = df_this[df_this["score_pct"].notna() & (df_this["score_pct"] <= 50)]
    low_last = df_last[df_last["score_pct"].notna() & (df_last["score_pct"] <= 50)]
    k3.metric("🚨 부정 리뷰 (이번달)", f"{len(low_this)}개",
              delta=f"{len(low_this) - len(low_last):+d}", delta_color="inverse")

    response_rate = (
        len(df_dash[df_dash["status_norm"].isin(["답변완료", "처리완료"])])
        / len(df_dash) * 100 if len(df_dash) > 0 else 0
    )
    k4.metric("응답률", f"{response_rate:.1f}%")

    st.markdown("---")

    # 🐛 버그 수정: NaN 플랫폼 제외
    st.subheader("🏢 플랫폼별 평균 만족도")
    df_scored = df_dash[df_dash["score_pct"].notna()]

    if len(df_scored) > 0:
        plat_stats = (
            df_scored.groupby("platform")
            .agg(평균만족도=("score_pct", "mean"), 리뷰수=("score_pct", "count"))
            .reset_index()
        )
        all_plat = df_dash.groupby("platform").size().reset_index(name="총리뷰수")
        resp = df_dash[df_dash["status_norm"].isin(["답변완료", "처리완료"])].groupby("platform").size().reset_index(name="응답완료")
        plat_stats = plat_stats.merge(all_plat, on="platform", how="left")
        plat_stats = plat_stats.merge(resp, on="platform", how="left").fillna({"응답완료": 0})
        plat_stats["응답률"] = (plat_stats["응답완료"] / plat_stats["총리뷰수"] * 100).round(1)
        plat_stats = plat_stats.sort_values("리뷰수", ascending=False)

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig = px.bar(
                plat_stats, x="평균만족도", y="platform", orientation="h",
                text=plat_stats["평균만족도"].round(1).astype(str) + "%",
                color="평균만족도",
                color_continuous_scale=["#ef4444", "#facc15", "#10b981"],
                range_color=[40, 100],
                title="플랫폼별 평균 만족도 (%)",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                height=max(300, len(plat_stats) * 50),
            )
            st.plotly_chart(fig, use_container_width=True)

        plat_count_all = df_dash.groupby("platform").size().reset_index(name="리뷰수").sort_values("리뷰수", ascending=False)
        with col_chart2:
            fig2 = px.bar(
                plat_count_all, x="리뷰수", y="platform", orientation="h",
                text="리뷰수",
                color_discrete_sequence=["#6366f1"],
                title="플랫폼별 리뷰 수 (전체)",
            )
            fig2.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                height=max(300, len(plat_count_all) * 50),
            )
            st.plotly_chart(fig2, use_container_width=True)

        no_score_plats = set(plat_count_all["platform"]) - set(plat_stats["platform"])
        if no_score_plats:
            st.caption(f"💡 다음 플랫폼은 별점 데이터가 없거나 부족해 만족도 차트에서 제외됨: {', '.join(sorted(no_score_plats))}")

        with st.expander("플랫폼별 상세 표 보기"):
            display_df = plat_stats.copy()
            display_df["평균만족도"] = display_df["평균만족도"].round(1).astype(str) + "%"
            display_df["응답률"] = display_df["응답률"].astype(str) + "%"
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("점수 데이터가 있는 리뷰가 부족합니다.")

    st.markdown("---")

    # 시계열
    st.subheader("📅 시계열 추이")
    df_ts = df_dash[df_dash["date_dt"].notna() & df_dash["score_pct"].notna()].copy()
    if len(df_ts) > 0:
        granularity = st.radio("단위", ["주별", "월별"], horizontal=True, key="ts_gran")
        if granularity == "주별":
            df_ts["period"] = df_ts["date_dt"].dt.to_period("W").dt.start_time
        else:
            df_ts["period"] = df_ts["date_dt"].dt.to_period("M").dt.start_time

        ts_agg = (
            df_ts.groupby("period")
            .agg(리뷰수=("id", "count"), 평균만족도=("score_pct", "mean"))
            .reset_index().sort_values("period")
        )

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Bar(
            x=ts_agg["period"], y=ts_agg["리뷰수"],
            name="리뷰 수", yaxis="y", marker_color="#cbd5e1",
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts_agg["period"], y=ts_agg["평균만족도"],
            name="평균 만족도 (%)", yaxis="y2",
            line=dict(color="#ef4444", width=3), mode="lines+markers",
        ))
        fig_ts.update_layout(
            title=f"{granularity} 리뷰 수 & 평균 만족도",
            yaxis=dict(title="리뷰 수", side="left"),
            yaxis2=dict(title="평균 만족도 (%)", side="right", overlaying="y", range=[0, 100]),
            height=400, hovermode="x unified",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        if len(ts_agg) >= 2:
            last_val = ts_agg["평균만족도"].iloc[-1]
            prev_val = ts_agg["평균만족도"].iloc[-2]
            if not pd.isna(last_val) and not pd.isna(prev_val):
                diff = last_val - prev_val
                if abs(diff) >= 5:
                    if diff < 0:
                        st.warning(f"⚠️ 직전 {granularity[:-1]} 대비 만족도가 **{abs(diff):.1f}%p 하락**했어요. 점검 필요!")
                    else:
                        st.success(f"📈 직전 {granularity[:-1]} 대비 만족도가 **{diff:.1f}%p 상승**했어요!")
    else:
        st.info("날짜 + 점수 정보가 있는 리뷰가 부족합니다.")

    st.markdown("---")

    # 카테고리별
    st.subheader("🏷️ 카테고리별 언급 & 만족도")
    cat_data = []
    for cat in HOTEL_CATEGORIES.keys():
        cat_reviews = df_dash[df_dash["categories_final"].apply(lambda x: cat in (x or []))]
        if len(cat_reviews) == 0:
            continue
        cat_data.append({
            "카테고리": cat,
            "언급수": len(cat_reviews),
            "평균만족도": cat_reviews["score_pct"].dropna().mean(),
        })
    cat_df = pd.DataFrame(cat_data).sort_values("언급수", ascending=False) if cat_data else pd.DataFrame()

    if not cat_df.empty:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_c1 = px.bar(
                cat_df, x="언급수", y="카테고리", orientation="h",
                text="언급수", color_discrete_sequence=["#6366f1"],
                title="카테고리별 언급 빈도",
            )
            fig_c1.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
            st.plotly_chart(fig_c1, use_container_width=True)
        with col_c2:
            cat_with_score = cat_df.dropna(subset=["평균만족도"])
            if not cat_with_score.empty:
                fig_c2 = px.bar(
                    cat_with_score, x="평균만족도", y="카테고리", orientation="h",
                    text=cat_with_score["평균만족도"].round(1).astype(str) + "%",
                    color="평균만족도",
                    color_continuous_scale=["#ef4444", "#facc15", "#10b981"],
                    range_color=[40, 100],
                    title="카테고리별 평균 만족도",
                )
                fig_c2.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, height=450)
                st.plotly_chart(fig_c2, use_container_width=True)
    else:
        st.info("카테고리 데이터 부족. '운영 도구 → 카테고리 자동 태깅' 을 실행하면 더 정확해집니다.")

    st.markdown("---")

    # 세그먼트
    st.subheader("👥 고객 세그먼트 분석")
    seg_tabs = st.tabs(["🌍 국적별", "👤 투숙 유형별", "🛏 객실 타입별"])

    def seg_chart(df_seg, col_name, label):
        seg_df = df_seg[df_seg[col_name].notna() & (df_seg[col_name] != "")]
        if len(seg_df) == 0:
            st.info(f"{label} 데이터가 부족합니다.")
            return
        agg = (
            seg_df.groupby(col_name)
            .agg(리뷰수=("id", "count"), 평균만족도=("score_pct", "mean"))
            .reset_index().sort_values("리뷰수", ascending=False).head(15)
        )
        fig = px.bar(
            agg, x="리뷰수", y=col_name, orientation="h",
            text=agg["평균만족도"].round(1).astype(str) + "%",
            color="평균만족도",
            color_continuous_scale=["#ef4444", "#facc15", "#10b981"],
            range_color=[40, 100],
            title=f"{label} 리뷰 수 & 평균 만족도",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with seg_tabs[0]:
        seg_chart(df_dash, "country", "국적별")
    with seg_tabs[1]:
        seg_chart(df_dash, "traveler_type", "투숙 유형별")
    with seg_tabs[2]:
        seg_chart(df_dash, "room_type", "객실 타입별")

    st.markdown("---")

    # 벤치마크
    st.subheader("📊 벤치마크 비교")
    benchmark = get_benchmark()
    our_avg = df_dash["score_pct"].dropna().mean()

    bk1, bk2, bk3 = st.columns(3)
    bk1.metric("우리 호텔", f"{our_avg:.1f}%" if not pd.isna(our_avg) else "—")
    if benchmark.get("jeju_avg") is not None:
        diff = our_avg - benchmark["jeju_avg"] if not pd.isna(our_avg) else None
        bk2.metric("제주 4-5성 평균", f"{benchmark['jeju_avg']:.1f}%",
                   delta=f"{diff:+.1f}%p" if diff is not None else None)
    else:
        bk2.metric("제주 4-5성 평균", "미설정", help="운영 도구에서 입력하세요")

    if benchmark.get("competitor_avg") is not None:
        diff = our_avg - benchmark["competitor_avg"] if not pd.isna(our_avg) else None
        bk3.metric("주요 경쟁사 평균", f"{benchmark['competitor_avg']:.1f}%",
                   delta=f"{diff:+.1f}%p" if diff is not None else None)
    else:
        bk3.metric("주요 경쟁사 평균", "미설정")

    if benchmark.get("note"):
        st.caption(f"📝 {benchmark['note']}")


# ═════════════════════════════════════════════════════════
# TAB 2: 액션 아이템 ⭐
# ═════════════════════════════════════════════════════════
with main_tabs[1]:
    st.header("🎯 VOC → 액션 아이템")
    st.caption("부정 리뷰들을 분석해서 부서별 처리 액션으로 변환합니다.")

    action_subtabs = st.tabs(["📋 진행 중 칸반", "🤖 새 액션 생성", "📨 부서별 리포트"])

    with action_subtabs[0]:
        action_items = get_action_items()
        if not action_items:
            st.info("아직 생성된 액션 아이템이 없습니다. '새 액션 생성' 탭에서 만들어보세요.")
        else:
            all_depts = sorted(set(a.get("department", "기타") for a in action_items))
            dept_filter = st.multiselect("부서 필터", all_depts, default=all_depts, key="action_dept_filter")

            filtered_actions = [a for a in action_items if a.get("department", "기타") in dept_filter]

            todo = [a for a in filtered_actions if a.get("status", "todo") == "todo"]
            doing = [a for a in filtered_actions if a.get("status") == "doing"]
            done = [a for a in filtered_actions if a.get("status") == "done"]

            kan1, kan2, kan3 = st.columns(3)
            with kan1:
                st.markdown(f"### 📥 할 일 ({len(todo)})")
                for a in todo:
                    with st.container(border=True):
                        st.markdown(f"**{a.get('title', '(제목 없음)')}**")
                        st.caption(f"🏢 {a.get('department', '')}")
                        priority = a.get("priority", "보통")
                        prio_color = {"높음": "🔴", "보통": "🟡", "낮음": "🟢"}.get(priority, "⚪")
                        st.caption(f"{prio_color} 우선순위: {priority}")
                        if a.get("description"):
                            st.markdown(f"_{a['description'][:150]}_")
                        if a.get("evidence_count"):
                            st.caption(f"📌 근거: {a['evidence_count']}개 리뷰")
                        col_b1, col_b2 = st.columns(2)
                        if col_b1.button("▶️ 시작", key=f"start_{a['id']}", use_container_width=True):
                            db.collection("action_items").document(a["id"]).update({"status": "doing"})
                            st.cache_data.clear()
                            st.rerun()
                        if col_b2.button("🗑", key=f"del_{a['id']}", use_container_width=True):
                            db.collection("action_items").document(a["id"]).delete()
                            st.cache_data.clear()
                            st.rerun()

            with kan2:
                st.markdown(f"### 🔧 처리 중 ({len(doing)})")
                for a in doing:
                    with st.container(border=True):
                        st.markdown(f"**{a.get('title', '')}**")
                        st.caption(f"🏢 {a.get('department', '')}")
                        if a.get("description"):
                            st.markdown(f"_{a['description'][:150]}_")
                        col_b1, col_b2 = st.columns(2)
                        if col_b1.button("✅ 완료", key=f"done_{a['id']}", use_container_width=True):
                            db.collection("action_items").document(a["id"]).update({
                                "status": "done",
                                "done_at": datetime.now().isoformat(),
                            })
                            st.cache_data.clear()
                            st.rerun()
                        if col_b2.button("◀️", key=f"back_{a['id']}", use_container_width=True, help="할 일로 되돌리기"):
                            db.collection("action_items").document(a["id"]).update({"status": "todo"})
                            st.cache_data.clear()
                            st.rerun()

            with kan3:
                st.markdown(f"### ✅ 완료 ({len(done)})")
                for a in done[:20]:
                    with st.container(border=True):
                        st.markdown(f"~~**{a.get('title', '')}**~~")
                        st.caption(f"🏢 {a.get('department', '')} | ✅ {a.get('done_at', '')[:10]}")

    with action_subtabs[1]:
        st.subheader("🤖 부정 리뷰 분석 → 액션 아이템 생성")

        col_ag1, col_ag2 = st.columns([3, 1])
        with col_ag1:
            ag_period = st.selectbox("분석 기간", ["최근 30일", "최근 60일", "최근 90일", "전체"], key="action_period")
        with col_ag2:
            ag_threshold = st.slider("부정 기준 (%)", 30, 70, 60, key="ag_threshold")

        df_neg = df.copy()
        if ag_period == "최근 30일":
            df_neg = df_neg[df_neg["date_dt"].notna() & (df_neg["date_dt"] >= now - timedelta(days=30))]
        elif ag_period == "최근 60일":
            df_neg = df_neg[df_neg["date_dt"].notna() & (df_neg["date_dt"] >= now - timedelta(days=60))]
        elif ag_period == "최근 90일":
            df_neg = df_neg[df_neg["date_dt"].notna() & (df_neg["date_dt"] >= now - timedelta(days=90))]

        df_neg = df_neg[df_neg["score_pct"].notna() & (df_neg["score_pct"] <= ag_threshold)]

        st.info(f"분석 대상: {len(df_neg)}개 부정 리뷰 ({ag_period}, ≤{ag_threshold}%)")

        if st.button("🤖 AI로 액션 아이템 추출", use_container_width=True, disabled=len(df_neg) == 0):
            with st.spinner("AI가 부정 리뷰를 분석해 액션을 추출 중..."):
                review_blocks = []
                for _, r in df_neg.head(60).iterrows():
                    cats = ",".join(r.get("categories_final") or []) or "기타"
                    block = f"[{r['platform']}|{r.get('score','')}점|{cats}] {build_review_text(r)[:300]}"
                    review_blocks.append(block)
                context = "\n\n".join(review_blocks)[:10000]

                categories_list = list(HOTEL_CATEGORIES.keys())
                dept_list = sorted(set(CATEGORY_TO_DEPT.values()))

                prompt = f"""당신은 호텔 운영 컨설턴트입니다. 아래 부정 리뷰들을 분석해서 호텔이 즉시 실행해야 할 액션 아이템을 추출하세요.

[가능한 카테고리]
{", ".join(categories_list)}

[가능한 부서]
{", ".join(dept_list)}

[규칙]
- 비슷한 불만은 하나의 액션으로 묶기
- 액션은 구체적이고 실행 가능해야 함
- 우선순위: "높음" / "보통" / "낮음"
  - 높음: 안전/위생/즉시 조치 필요
  - 보통: 만족도에 큰 영향, 1-2주 내 조치
  - 낮음: 장기 개선 과제
- 최대 8개

[출력 형식 — JSON 배열만]
[
  {{
    "title": "조식 한식 메뉴 다양성 확대",
    "department": "F&B팀",
    "category": "조식",
    "priority": "보통",
    "description": "한식 사이드가 김치/계란말이 위주로 반복된다는 지적 다수. 주 단위 한식 로테이션 메뉴 도입 검토.",
    "estimated_days": 14,
    "evidence_count": 5
  }}
]

[부정 리뷰 데이터 (총 {len(df_neg.head(60))}건)]
{context}

JSON 배열만 출력. 다른 설명 일절 금지."""

                try:
                    result = call_gemini(prompt)
                    result = result.strip()
                    if result.startswith("```"):
                        result = re.sub(r"```(?:json)?\n?", "", result)
                        result = result.rstrip("`").strip()
                    actions = json.loads(result)

                    if not isinstance(actions, list):
                        st.error("AI가 잘못된 형식으로 답변했습니다.")
                    else:
                        st.success(f"✅ {len(actions)}개 액션 추출 완료. 검토 후 등록하세요.")

                        for idx, a in enumerate(actions):
                            with st.container(border=True):
                                col_p1, col_p2 = st.columns([3, 1])
                                with col_p1:
                                    st.markdown(f"### {a.get('title', '')}")
                                    st.caption(
                                        f"🏢 **{a.get('department', '')}** | "
                                        f"🏷 {a.get('category', '')} | "
                                        f"우선순위: **{a.get('priority', '')}** | "
                                        f"예상: {a.get('estimated_days', '')}일 | "
                                        f"근거: {a.get('evidence_count', 0)}건"
                                    )
                                    st.markdown(a.get("description", ""))
                                with col_p2:
                                    if st.button("➕ 칸반에 추가", key=f"add_action_{idx}", use_container_width=True):
                                        db.collection("action_items").add({
                                            "title": a.get("title", ""),
                                            "department": a.get("department", "기타"),
                                            "category": a.get("category", ""),
                                            "priority": a.get("priority", "보통"),
                                            "description": a.get("description", ""),
                                            "estimated_days": a.get("estimated_days", 7),
                                            "evidence_count": a.get("evidence_count", 0),
                                            "status": "todo",
                                            "created_at": datetime.now().isoformat(),
                                        })
                                        st.success("등록됨!")
                                        st.cache_data.clear()

                        if st.button("📥 전체 한꺼번에 등록", use_container_width=True, type="primary"):
                            for a in actions:
                                db.collection("action_items").add({
                                    "title": a.get("title", ""),
                                    "department": a.get("department", "기타"),
                                    "category": a.get("category", ""),
                                    "priority": a.get("priority", "보통"),
                                    "description": a.get("description", ""),
                                    "estimated_days": a.get("estimated_days", 7),
                                    "evidence_count": a.get("evidence_count", 0),
                                    "status": "todo",
                                    "created_at": datetime.now().isoformat(),
                                })
                            st.success(f"✅ {len(actions)}개 모두 등록됨!")
                            st.cache_data.clear()
                            st.rerun()
                except Exception as e:
                    st.error(f"분석 실패: {e}")
                    with st.expander("상세 오류"):
                        st.code(traceback.format_exc())

    with action_subtabs[2]:
        st.subheader("📨 부서별 주간 리포트")
        st.caption("부서장에게 그대로 보낼 수 있는 리포트를 생성합니다.")

        all_depts = sorted(set(CATEGORY_TO_DEPT.values()))
        sel_dept = st.selectbox("부서 선택", all_depts)

        if st.button("📨 리포트 생성", use_container_width=True):
            with st.spinner("리포트 작성 중..."):
                dept_categories = [c for c, d in CATEGORY_TO_DEPT.items() if d == sel_dept]

                week_ago = now - timedelta(days=7)
                df_week = df[df["date_dt"].notna() & (df["date_dt"] >= week_ago)]
                df_week_dept = df_week[
                    df_week["categories_final"].apply(
                        lambda x: any(c in (x or []) for c in dept_categories)
                    )
                ]

                actions = get_action_items()
                dept_actions = [a for a in actions if a.get("department") == sel_dept]
                todo_actions = [a for a in dept_actions if a.get("status") == "todo"]
                doing_actions = [a for a in dept_actions if a.get("status") == "doing"]

                positive_count = len(df_week_dept[df_week_dept["score_pct"] >= 80])
                negative_count = len(df_week_dept[df_week_dept["score_pct"] <= 50])

                review_samples = []
                for _, r in df_week_dept.head(10).iterrows():
                    review_samples.append(
                        f"- ({r.get('score','')}점, {r['platform']}) {build_review_text(r)[:200]}"
                    )

                prompt = f"""당신은 호텔 총지배인입니다. **{sel_dept}** 부서장에게 보낼 주간 리포트를 작성하세요.

[데이터]
- 지난 7일 관련 리뷰: {len(df_week_dept)}건
- 긍정(80%+): {positive_count}건, 부정(50%-): {negative_count}건
- 미처리 액션: {len(todo_actions)}건, 처리 중: {len(doing_actions)}건

[관련 리뷰 샘플]
{chr(10).join(review_samples)}

[미처리 액션]
{chr(10).join([f"- {a.get('title')}: {a.get('description', '')[:100]}" for a in todo_actions])}

[리포트 형식]
1. 인사말 (한 줄)
2. 이번 주 핵심 요약 (3줄)
3. 칭찬받은 부분 (구체적 인용 1-2개)
4. 주의해야 할 부분 (구체적 인용 1-2개)
5. 이번 주 부탁드릴 액션 (3가지)
6. 마무리 (한 줄)

존댓말로, 사무적이지 않게, 동료에게 말하듯 자연스럽게."""

                try:
                    report = call_gemini(prompt)
                    st.success("리포트 완성")
                    st.text_area("리포트 (복사해서 전달)", value=report, height=500)
                    st.download_button("📥 다운로드", report,
                                       file_name=f"weekly_report_{sel_dept}_{now.strftime('%Y%m%d')}.txt")
                except Exception as e:
                    st.error(f"실패: {e}")


# ═════════════════════════════════════════════════════════
# TAB 3: 키워드 분석
# ═════════════════════════════════════════════════════════
with main_tabs[2]:
    st.header("🔍 키워드 분석")

    STOPWORDS = {
        "있다", "없다", "하다", "되다", "이다", "그", "저", "이", "그리고", "근데",
        "정말", "진짜", "너무", "조금", "약간", "그런", "이런", "저런", "또한", "또",
        "수", "것", "거", "게", "수가", "안", "잘", "더", "좀", "다", "도", "은",
        "는", "이", "가", "을", "를", "에", "의", "와", "과", "에서", "로", "으로",
        "호텔", "방", "객실",
    }

    def extract_keywords(texts, top_n=50):
        counter = Counter()
        for text in texts:
            if not text:
                continue
            words = re.findall(r"[가-힣]{2,}", str(text))
            for w in words:
                if w not in STOPWORDS:
                    counter[w] += 1
        return counter.most_common(top_n)

    pos_texts = df[df["score_pct"].notna() & (df["score_pct"] >= 80)]["full_text"].tolist()
    neg_texts = df[df["score_pct"].notna() & (df["score_pct"] <= 50)]["full_text"].tolist()
    pos_texts += df[df["positive"] != ""]["positive"].tolist()
    neg_texts += df[df["negative"] != ""]["negative"].tolist()

    col_kw1, col_kw2 = st.columns(2)
    with col_kw1:
        st.markdown("### 😊 긍정 키워드 TOP 30")
        pos_kw = extract_keywords(pos_texts, top_n=30)
        if pos_kw:
            kw_df = pd.DataFrame(pos_kw, columns=["키워드", "빈도"])
            fig = px.bar(kw_df.head(20), x="빈도", y="키워드", orientation="h",
                         color="빈도", color_continuous_scale="Greens")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("전체"):
                st.dataframe(kw_df, use_container_width=True, hide_index=True)
        else:
            st.info("데이터 부족")

    with col_kw2:
        st.markdown("### 😞 부정 키워드 TOP 30")
        neg_kw = extract_keywords(neg_texts, top_n=30)
        if neg_kw:
            kw_df = pd.DataFrame(neg_kw, columns=["키워드", "빈도"])
            fig = px.bar(kw_df.head(20), x="빈도", y="키워드", orientation="h",
                         color="빈도", color_continuous_scale="Reds")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("전체"):
                st.dataframe(kw_df, use_container_width=True, hide_index=True)
        else:
            st.info("데이터 부족")

    st.markdown("---")

    st.subheader("☁️ 워드클라우드")
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        font_path = None
        for f in fm.findSystemFonts():
            name = f.lower()
            if "nanum" in name or "malgun" in name or "applegothic" in name or ("noto" in name and "kr" in name):
                font_path = f
                break

        def make_wc(text, colormap):
            if not text:
                return None
            try:
                wc = WordCloud(
                    width=600, height=400, background_color="white",
                    font_path=font_path, colormap=colormap, max_words=80,
                ).generate(text)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                return fig
            except Exception as e:
                st.caption(f"오류: {e}")
                return None

        wc_col1, wc_col2 = st.columns(2)
        with wc_col1:
            st.markdown("**긍정**")
            pos_text = " ".join([w for w, _ in pos_kw]) if pos_kw else ""
            fig = make_wc(pos_text, "Greens")
            if fig:
                st.pyplot(fig)
        with wc_col2:
            st.markdown("**부정**")
            neg_text = " ".join([w for w, _ in neg_kw]) if neg_kw else ""
            fig = make_wc(neg_text, "Reds")
            if fig:
                st.pyplot(fig)

        if not font_path:
            st.caption("⚠️ 한글 폰트 없음. `packages.txt` 에 `fonts-nanum` 추가 필요.")
    except ImportError:
        st.warning("`wordcloud` 패키지 필요")

    st.markdown("---")

    st.subheader("📈 카테고리별 언급 추이")
    df_ts2 = df[df["date_dt"].notna()].copy()
    if len(df_ts2) > 0:
        sel_categories = st.multiselect(
            "추적할 카테고리",
            list(HOTEL_CATEGORIES.keys()),
            default=["조식", "직원/서비스", "청결", "객실"],
        )
        if sel_categories:
            df_ts2["month"] = df_ts2["date_dt"].dt.to_period("M").dt.start_time
            trend_data = []
            for cat in sel_categories:
                cat_df_t = df_ts2[df_ts2["categories_final"].apply(lambda x: cat in (x or []))]
                monthly = cat_df_t.groupby("month").size().reset_index(name="언급수")
                monthly["카테고리"] = cat
                trend_data.append(monthly)
            if trend_data:
                trend_df = pd.concat(trend_data)
                fig = px.line(trend_df, x="month", y="언급수", color="카테고리",
                              markers=True, title="월별 카테고리 언급 추이")
                fig.update_layout(height=450, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════
# TAB 4: 연관 분석 ⭐
# ═════════════════════════════════════════════════════════
with main_tabs[3]:
    st.header("🔗 카테고리 연관 분석")
    st.caption("어떤 불만이 함께 나타나는지 분석합니다. 근본 원인 파악에 도움됩니다.")

    co_period = st.selectbox("분석 기간", ["전체", "최근 90일", "최근 180일"], key="co_period")

    df_co = df.copy()
    if co_period == "최근 90일":
        df_co = df_co[df_co["date_dt"].notna() & (df_co["date_dt"] >= now - timedelta(days=90))]
    elif co_period == "최근 180일":
        df_co = df_co[df_co["date_dt"].notna() & (df_co["date_dt"] >= now - timedelta(days=180))]

    co_mode = st.radio(
        "분석 대상",
        ["전체 리뷰", "부정 리뷰만 (≤50%)", "긍정 리뷰만 (≥80%)"],
        horizontal=True,
    )
    if co_mode == "부정 리뷰만 (≤50%)":
        df_co = df_co[df_co["score_pct"].notna() & (df_co["score_pct"] <= 50)]
    elif co_mode == "긍정 리뷰만 (≥80%)":
        df_co = df_co[df_co["score_pct"].notna() & (df_co["score_pct"] >= 80)]

    if len(df_co) < 5:
        st.info("분석할 데이터가 부족합니다.")
    else:
        categories = list(HOTEL_CATEGORIES.keys())
        co_matrix = pd.DataFrame(0, index=categories, columns=categories)

        for _, r in df_co.iterrows():
            cats = r.get("categories_final") or []
            for c1 in cats:
                for c2 in cats:
                    if c1 in categories and c2 in categories:
                        co_matrix.loc[c1, c2] += 1

        non_zero_cats = [c for c in categories if co_matrix.loc[c, c] > 0]
        co_matrix_clean = co_matrix.loc[non_zero_cats, non_zero_cats]

        if co_matrix_clean.empty:
            st.info("연관 분석할 카테고리 데이터가 부족합니다.")
        else:
            st.subheader("🔥 카테고리 동시 출현 히트맵")
            st.caption("**대각선**(자기자신) = 카테고리 총 언급수, **비대각선** = 두 카테고리가 함께 언급된 횟수")

            fig = px.imshow(
                co_matrix_clean, text_auto=True, aspect="auto",
                color_continuous_scale="OrRd",
                title=f"카테고리 동시 출현 ({co_mode}, {co_period})",
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🔍 가장 자주 함께 나타나는 카테고리 쌍 TOP 10")
            pairs = []
            for i, c1 in enumerate(non_zero_cats):
                for c2 in non_zero_cats[i+1:]:
                    count = co_matrix_clean.loc[c1, c2]
                    if count > 0:
                        c1_total = co_matrix_clean.loc[c1, c1]
                        c2_total = co_matrix_clean.loc[c2, c2]
                        confidence = count / min(c1_total, c2_total) if min(c1_total, c2_total) > 0 else 0
                        pairs.append({
                            "카테고리 A": c1,
                            "카테고리 B": c2,
                            "함께 언급": int(count),
                            "A 총 언급": int(c1_total),
                            "B 총 언급": int(c2_total),
                            "연관도": f"{confidence*100:.1f}%",
                        })

            if pairs:
                pairs_df = pd.DataFrame(pairs).sort_values("함께 언급", ascending=False).head(10)
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)

                if st.button("🤖 AI에게 이 패턴 해석 부탁하기"):
                    with st.spinner("분석 중..."):
                        pattern_text = "\n".join([
                            f"- '{p['카테고리 A']}' 와 '{p['카테고리 B']}' 가 함께 언급된 횟수: {p['함께 언급']}회"
                            for p in pairs[:8]
                        ])
                        prompt = f"""호텔 리뷰 분석에서 다음과 같은 카테고리 동시 출현 패턴이 나왔습니다.
대상: {co_mode}

[패턴]
{pattern_text}

이 패턴이 의미하는 바를 호텔 운영 관점에서 해석하고, 어떤 인사이트가 있는지 3-4문장으로 설명하세요.
근본 원인이 보인다면 그것도 짚어주세요."""
                        try:
                            interpretation = call_gemini(prompt)
                            st.info(interpretation)
                        except Exception as e:
                            st.error(f"해석 실패: {e}")


# ═════════════════════════════════════════════════════════
# TAB 5: 리뷰 관리
# ═════════════════════════════════════════════════════════
with main_tabs[4]:
    st.header("📝 리뷰 관리")

    # ─── 🔎 강력한 검색 바 (최상단, 항상 보이게) ───
    search_col1, search_col2, search_col3 = st.columns([5, 2, 2])
    with search_col1:
        search_keyword = st.text_input(
            "🔎 리뷰 검색 (본문, 제목, 작성자, 좋았던점/아쉬운점 모두 검색)",
            placeholder="예: 지난주에 다녀왔어요 / 조식 / 김지배 / 청결 / 부드러웠어요",
            key="rm_search",
            help="플랫폼 관리자 페이지에서 본 문구 일부를 그대로 검색해 보세요. 검색 시 다른 필터는 자동으로 풀려 모든 리뷰에서 찾습니다.",
        )
    with search_col2:
        search_mode = st.selectbox(
            "검색 모드",
            ["부분 일치", "정확히 일치", "단어 모두 포함 (AND)"],
            key="rm_search_mode",
            help="부분 일치: 입력값이 포함된 리뷰 / 정확히 일치: 띄어쓰기까지 동일 / AND: 여러 단어 모두 들어간 리뷰",
        )
    with search_col3:
        search_in_reply = st.checkbox(
            "AI 답변도 검색",
            value=False,
            key="rm_search_reply",
            help="작성한 AI 답변 본문에서도 검색합니다",
        )

    is_searching = bool(search_keyword and search_keyword.strip())

    # ─── 필터 옵션 (검색 중이면 자동으로 접힘) ───
    with st.expander(
        "🔍 필터 옵션 " + ("(검색 중이라 자동 완화됨)" if is_searching else ""),
        expanded=not is_searching,
    ):
        if is_searching:
            st.caption("💡 검색어가 입력된 동안에는 플랫폼/상태 필터가 무시되고 **모든 리뷰**에서 찾아요. 다른 필터(점수/카테고리)는 적용됩니다.")

        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            platforms_all = sorted(df["platform"].unique())
            platforms_sel = st.multiselect("플랫폼", platforms_all, default=platforms_all, key="rm_plat")
            status_sel = st.multiselect("처리 상태", ["대기중", "답변완료", "처리완료"],
                                        default=["대기중"], key="rm_status")
        with f_col2:
            score_filter = st.select_slider(
                "점수 범위",
                options=["전체", "낮음 (≤50%)", "중간 (50~80%)", "높음 (≥80%)"],
                value="전체", key="rm_score",
            )
            category_sel = st.multiselect("카테고리 필터", list(HOTEL_CATEGORIES.keys()),
                                          default=[], key="rm_cat")
        with f_col3:
            sort_order = st.selectbox(
                "정렬",
                ["최신 수집순", "오래된 수집순", "낮은 점수 먼저", "높은 점수 먼저",
                 "리뷰 작성일 (최신순)", "리뷰 작성일 (오래된순)"],
                key="rm_sort",
            )

    # ─── 필터 적용 ───
    # 검색 중이면 플랫폼/상태 필터는 풀어줌 (전체에서 찾기)
    if is_searching:
        filtered = df.copy()
    else:
        filtered = df[df["platform"].isin(platforms_sel)]
        filtered = filtered[filtered["status_norm"].isin(status_sel)]

    if score_filter == "낮음 (≤50%)":
        filtered = filtered[filtered["score_pct"].notna() & (filtered["score_pct"] <= 50)]
    elif score_filter == "중간 (50~80%)":
        filtered = filtered[
            filtered["score_pct"].notna()
            & (filtered["score_pct"] > 50)
            & (filtered["score_pct"] < 80)
        ]
    elif score_filter == "높음 (≥80%)":
        filtered = filtered[filtered["score_pct"].notna() & (filtered["score_pct"] >= 80)]

    if category_sel:
        filtered = filtered[
            filtered["categories_final"].apply(
                lambda x: any(c in (x or []) for c in category_sel)
            )
        ]

    # ─── 검색 적용 ───
    if is_searching:
        kw_raw = search_keyword.strip()
        kw = kw_raw.lower()

        # 검색 대상 컬럼: 본문 + 작성자 이름 + (옵션)AI답변
        def build_search_target(row):
            parts = [
                str(row.get("content", "")),
                str(row.get("title", "")),
                str(row.get("positive", "")),
                str(row.get("negative", "")),
                str(row.get("satisfaction_tags", "")),
                str(row.get("user", "")),
                str(row.get("owner_reply", "")),
            ]
            if search_in_reply:
                parts.append(str(row.get("ai_reply", "")))
                parts.append(str(row.get("final_reply", "")))
            return " ".join(parts).lower()

        filtered["_search_target"] = filtered.apply(build_search_target, axis=1)

        if search_mode == "정확히 일치":
            mask = filtered["_search_target"].str.contains(re.escape(kw), na=False)
        elif search_mode == "단어 모두 포함 (AND)":
            words = [w.strip() for w in kw.split() if w.strip()]
            mask = pd.Series([True] * len(filtered), index=filtered.index)
            for w in words:
                mask = mask & filtered["_search_target"].str.contains(re.escape(w), na=False)
        else:  # 부분 일치
            mask = filtered["_search_target"].str.contains(re.escape(kw), na=False)

        filtered = filtered[mask].drop(columns=["_search_target"])

    if sort_order == "오래된 수집순":
        filtered = filtered.iloc[::-1]
    elif sort_order == "낮은 점수 먼저":
        filtered = filtered.sort_values("score_pct", ascending=True, na_position="last")
    elif sort_order == "높은 점수 먼저":
        filtered = filtered.sort_values("score_pct", ascending=False, na_position="last")
    elif sort_order == "리뷰 작성일 (최신순)":
        filtered = filtered.sort_values("date_dt", ascending=False, na_position="last")
    elif sort_order == "리뷰 작성일 (오래된순)":
        filtered = filtered.sort_values("date_dt", ascending=True, na_position="last")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("필터 결과", f"{len(filtered)}개")
    c2.metric("⏳ 대기중", f"{len(filtered[filtered['status_norm'] == '대기중'])}개")
    low_score_waiting = filtered[
        (filtered["status_norm"] == "대기중")
        & (filtered["score_pct"].notna())
        & (filtered["score_pct"] <= 50)
    ]
    c3.metric("🚨 부정 리뷰 (대기)", f"{len(low_score_waiting)}개")
    avg_pct = filtered["score_pct"].dropna().mean()
    c4.metric("평균 만족도", f"{avg_pct:.0f}%" if not pd.isna(avg_pct) else "—")

    # ─── 검색 결과 알림 ───
    if is_searching:
        if len(filtered) == 0:
            st.error(f"😶 '{search_keyword}' 검색 결과가 없어요. 띄어쓰기/오타를 확인하거나 검색 모드를 '부분 일치'로 바꿔보세요.")
        else:
            plat_breakdown = filtered.groupby("platform").size().sort_values(ascending=False)
            breakdown_text = ", ".join([f"{p}: {c}개" for p, c in plat_breakdown.items()])
            st.success(f"🔎 **'{search_keyword}'** 검색 결과 **{len(filtered)}개** 매칭 — {breakdown_text}")

    if len(low_score_waiting) > 0 and not is_searching:
        st.warning(f"🚨 점수가 낮은 리뷰 {len(low_score_waiting)}개가 답변을 기다리고 있습니다.")

    # ─── 매칭 부분 하이라이트 함수 ───
    def highlight_match(text, keyword, mode):
        """검색어를 노란색 마크다운으로 하이라이트"""
        if not text or not keyword:
            return text
        text = str(text)
        if mode == "단어 모두 포함 (AND)":
            words = [w.strip() for w in keyword.split() if w.strip()]
        else:
            words = [keyword.strip()]
        # 정규식: 대소문자 무시
        for w in words:
            if not w:
                continue
            pattern = re.compile(re.escape(w), re.IGNORECASE)
            text = pattern.sub(lambda m: f"<mark style='background-color:#fef08a;padding:0 2px;border-radius:3px'>{m.group(0)}</mark>", text)
        return text

    st.markdown("---")

    waiting_in_filter = filtered[filtered["status_norm"] == "대기중"]
    if st.button(
        f"🤖 대기 중 리뷰 일괄 AI 답변 ({len(waiting_in_filter)}개)",
        use_container_width=True,
        disabled=(len(waiting_in_filter) == 0),
    ):
        progress = st.progress(0)
        success_count = 0
        fail_count = 0
        templates = get_templates()
        for i, (_, row) in enumerate(waiting_in_filter.iterrows()):
            try:
                if row.get("ai_reply"):
                    continue
                prompt = build_reply_prompt(row, similar_templates=templates)
                reply = call_gemini(prompt)
                update_review(row["id"], {"ai_reply": reply})
                success_count += 1
            except Exception:
                fail_count += 1
            progress.progress((i + 1) / len(waiting_in_filter))
        st.success(f"✅ {success_count}개 초안 작성 완료 (실패 {fail_count}개)")
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    if len(filtered) == 0:
        st.info("필터 조건에 맞는 리뷰가 없습니다.")
    else:
        platforms_in_filter = sorted(filtered["platform"].unique())
        tabs = st.tabs([f"{p} ({len(filtered[filtered['platform']==p])})" for p in platforms_in_filter])

        for tab_idx, tab in enumerate(tabs):
            with tab:
                p_name = platforms_in_filter[tab_idx]
                p_df = filtered[filtered["platform"] == p_name]
                admin_url = ADMIN_URLS.get(p_name, "#")
                st.link_button(f"🌐 {p_name} 관리자 페이지 열기", admin_url)

                for _, row in p_df.iterrows():
                    unique_key = f"{row['platform']}_{row['id']}"
                    status = row["status_norm"]

                    header_parts = []
                    if status == "대기중":
                        header_parts.append("⏳")
                    elif status == "답변완료":
                        header_parts.append("✅")
                    else:
                        header_parts.append("🎯")

                    if row.get("score"):
                        header_parts.append(f"**{row['score']}점**")
                    if row.get("user"):
                        header_parts.append(row["user"][:20])
                    header_parts.append(row.get("date", ""))

                    preview = (
                        row.get("title") or row.get("negative")
                        or row.get("positive") or row.get("content", "")
                    )
                    preview = preview.replace("\n", " ")[:50]
                    header_parts.append(f"| {preview}...")

                    cats = row.get("categories_final") or []
                    if cats:
                        header_parts.append(f"🏷 {','.join(cats[:3])}")

                    intent = row.get("recommend_intent", "")
                    if intent == "positive":
                        header_parts.append("👍")
                    elif intent == "negative":
                        header_parts.append("👎")

                    is_waiting = status == "대기중"
                    # 검색 중이면 매칭된 리뷰 모두 자동 펼침
                    expand_this = is_waiting or is_searching
                    header = " ".join(header_parts)

                    with st.expander(header, expanded=expand_this):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            # 하이라이트 적용 헬퍼
                            def hl(text):
                                if is_searching and text:
                                    return highlight_match(text, search_keyword, search_mode)
                                return text

                            if row.get("title"):
                                if is_searching:
                                    st.markdown(f"### {hl(row['title'])}", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"### {row['title']}")

                            if row.get("positive") or row.get("negative"):
                                if row.get("positive"):
                                    st.markdown("**😊 좋았던 점**")
                                    if is_searching:
                                        st.markdown(f"> {hl(row['positive'])}", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"> {row['positive']}")
                                if row.get("negative"):
                                    st.markdown("**😞 아쉬운 점**")
                                    if is_searching:
                                        st.markdown(f"> {hl(row['negative'])}", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"> {row['negative']}")
                            else:
                                st.markdown("**[원문]**")
                                content = row.get('content', '(내용 없음)')
                                if is_searching:
                                    st.markdown(f"> {hl(content)}", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"> {content}")

                            if row.get("satisfaction_tags"):
                                st.caption(f"⭐ {row['satisfaction_tags']}")

                            meta = []
                            if row.get("room_type"):
                                meta.append(f"🛏 {row['room_type']}")
                            if row.get("stay_period"):
                                meta.append(f"📅 {row['stay_period']}")
                            if row.get("travel_date"):
                                meta.append(f"✈ {row['travel_date']}")
                            if row.get("post_time"):
                                meta.append(f"🕐 {row['post_time']}")
                            if row.get("country"):
                                meta.append(f"🌍 {row['country']}")
                            if row.get("traveler_type"):
                                meta.append(f"👥 {row['traveler_type']}")
                            if meta:
                                st.caption(" | ".join(meta))

                            if row.get("owner_reply"):
                                with st.expander("📩 호텔 측 기존 답변 보기", expanded=is_searching):
                                    if is_searching:
                                        st.markdown(f"> {hl(row['owner_reply'])}", unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"> {row['owner_reply']}")

                            st.markdown("---")

                            if not row.get("ai_reply"):
                                if st.button("🤖 AI 답변 초안 만들기", key=f"btn_{unique_key}"):
                                    with st.spinner("작성 중..."):
                                        try:
                                            templates = get_templates()
                                            prompt = build_reply_prompt(row, similar_templates=templates)
                                            reply = call_gemini(prompt)
                                            update_review(row["id"], {"ai_reply": reply})
                                            st.cache_data.clear()
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"실패: {e}")
                            else:
                                st.markdown("**✍️ AI 답변 초안 (수정 가능)**")
                                edited_reply = st.text_area(
                                    "답변 수정", value=row["ai_reply"],
                                    height=180, key=f"edit_{unique_key}",
                                    label_visibility="collapsed",
                                )
                                st.code(edited_reply, language=None)

                                if edited_reply != row["ai_reply"]:
                                    ratio = diff_ratio(row["ai_reply"], edited_reply)
                                    st.caption(f"📝 AI 원본 대비 변경률: {ratio*100:.1f}%")

                                c_a, c_b, c_c, c_d = st.columns(4)
                                with c_a:
                                    st.link_button("🌐 관리자", admin_url)
                                with c_b:
                                    if st.button("🔄 재생성", key=f"regen_{unique_key}"):
                                        with st.spinner("재생성 중..."):
                                            try:
                                                templates = get_templates()
                                                prompt = build_reply_prompt(row, similar_templates=templates)
                                                reply = call_gemini(prompt)
                                                update_review(row["id"], {"ai_reply": reply})
                                                st.cache_data.clear()
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"실패: {e}")
                                with c_c:
                                    if st.button("✅ 확정 완료", key=f"confirm_{unique_key}"):
                                        update_review(row["id"], {
                                            "ai_reply": edited_reply,
                                            "final_reply": edited_reply,
                                            "status": "처리완료",
                                            "completed_at": datetime.now().isoformat(),
                                        })
                                        st.cache_data.clear()
                                        st.rerun()
                                with c_d:
                                    if st.button("⭐ 템플릿 저장", key=f"tmpl_{unique_key}"):
                                        db.collection("reply_templates").add({
                                            "reply": edited_reply,
                                            "platform": row["platform"],
                                            "score": row.get("score", ""),
                                            "categories": row.get("categories_final", []),
                                            "saved_at": datetime.now().isoformat(),
                                        })
                                        st.success("저장됨!")
                                        st.cache_data.clear()

                        with col2:
                            st.write(f"**상태**: {status}")
                            if row.get("score"):
                                score_val = score_to_float(row["score"])
                                if score_val is not None:
                                    score_pct = score_to_pct(score_val, row["platform"])
                                    max_score = (
                                        5 if row["platform"]
                                        in ("구글(Google)", "트립어드바이저(TripAdvisor)")
                                        else 10
                                    )
                                    score_label = f"{row['score']}/{max_score}"
                                    if score_pct is not None:
                                        if score_pct <= 50:
                                            st.error(f"⚠️ {score_label}")
                                        elif score_pct < 80:
                                            st.warning(f"{score_label}")
                                        else:
                                            st.success(f"{score_label}")

                            if intent:
                                intent_label = {
                                    "positive": "👍 추천 의도",
                                    "negative": "👎 비추천 의도",
                                    "neutral": "😐 중립"
                                }.get(intent, "")
                                if intent == "positive":
                                    st.success(intent_label)
                                elif intent == "negative":
                                    st.error(intent_label)
                                else:
                                    st.info(intent_label)

                            if cats:
                                st.write("**카테고리**")
                                st.write(", ".join(cats))

                            if row.get("booking_id"):
                                st.caption(f"예약: `{row['booking_id']}`")
                            if row.get("review_id"):
                                st.caption(f"리뷰: `{row['review_id']}`")

                            if status == "답변완료":
                                if st.button("↩️ 대기로", key=f"undo_{unique_key}"):
                                    update_review(row["id"], {"status": "대기중", "has_reply": False})
                                    st.cache_data.clear()
                                    st.rerun()
                            elif status == "대기중":
                                if st.button("✅ 답변완료 표시", key=f"mark_replied_{unique_key}"):
                                    update_review(row["id"], {"status": "답변완료", "has_reply": True})
                                    st.cache_data.clear()
                                    st.rerun()


# ═════════════════════════════════════════════════════════
# TAB 6: AI 분석
# ═════════════════════════════════════════════════════════
with main_tabs[5]:
    st.header("🤖 AI 분석")
    ai_tabs = st.tabs(["💬 자유 질의", "📊 종합 보고서", "📚 답변 템플릿"])

    with ai_tabs[0]:
        st.subheader("💬 리뷰 데이터에 자유롭게 질문")
        st.caption("예: 지난 3개월간 조식 관련 가장 큰 불만은? / 일본 고객들이 좋아한 점은?")

        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            user_query = st.text_input("질문", key="rag_query")
        with col_q2:
            sample_size = st.number_input("분석 리뷰 수", 20, 200, 80, 10)

        if st.button("🔍 질문하기", use_container_width=True, disabled=not user_query):
            with st.spinner("AI가 데이터를 분석 중..."):
                relevant_df = df.copy()
                for cat in HOTEL_CATEGORIES.keys():
                    if cat in user_query:
                        relevant_df = relevant_df[
                            relevant_df["categories_final"].apply(lambda x: cat in (x or []))
                        ]
                        break

                if len(relevant_df) > sample_size:
                    relevant_df = relevant_df.head(sample_size)

                review_blocks = []
                for _, r in relevant_df.iterrows():
                    block = f"[{r['platform']}|{r.get('score','')}점|{r.get('date','')}] {build_review_text(r)[:300]}"
                    review_blocks.append(block)
                context = "\n\n".join(review_blocks)[:10000]

                prompt = f"""당신은 호텔 데이터 분석가입니다. 아래 리뷰 데이터를 바탕으로 사용자의 질문에 답하세요.

[규칙]
- 데이터에 없는 내용은 절대 만들지 말 것
- 구체적인 사례를 인용
- 가능하면 숫자/빈도로 답변
- 명확하지 않으면 "데이터가 부족합니다" 라고 답할 것

[리뷰 데이터 (총 {len(relevant_df)}건)]
{context}

[사용자 질문]
{user_query}

[답변]"""

                try:
                    answer = call_gemini(prompt)
                    st.info(answer)
                except Exception as e:
                    st.error(f"분석 실패: {e}")

    with ai_tabs[1]:
        st.subheader("📊 종합 분석 보고서")
        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            report_period = st.selectbox("분석 기간", ["전체", "최근 30일", "최근 90일"], key="report_period")
        with col_r2:
            report_sample = st.number_input("분석 리뷰 수", 20, 200, 80, 10, key="rep_sample")

        if st.button("📑 보고서 생성", use_container_width=True):
            with st.spinner("작성 중..."):
                df_rep = df.copy()
                if report_period == "최근 30일":
                    df_rep = df_rep[df_rep["date_dt"].notna() & (df_rep["date_dt"] >= now - timedelta(days=30))]
                elif report_period == "최근 90일":
                    df_rep = df_rep[df_rep["date_dt"].notna() & (df_rep["date_dt"] >= now - timedelta(days=90))]

                sample_df = df_rep.head(report_sample)
                review_blocks = [
                    f"[{r['platform']}|{r.get('score','')}점] {build_review_text(r)[:300]}"
                    for _, r in sample_df.iterrows()
                ]
                all_text = "\n\n".join(review_blocks)[:10000]

                prompt = f"""당신은 호텔 컨설팅 전문가입니다. 아래 리뷰들을 분석해 보고서를 작성하세요.

## 1. 핵심 강점 Best 3 (구체적 표현 인용)
## 2. 개선 필요 사항 Worst 3 (구체적 사례 + 우선순위)
## 3. 플랫폼별 고객 성향 차이 (있는 경우만)
## 4. 이번 주 운영 전략 제안 (실행 가능한 3가지)

[리뷰 데이터 (총 {len(sample_df)}건)]
{all_text}

[보고서]"""

                try:
                    report = call_gemini(prompt)
                    st.success("✅ 작성 완료")
                    st.markdown(report)
                    st.download_button("📥 다운로드", report,
                                       file_name=f"amber_report_{now.strftime('%Y%m%d_%H%M')}.txt")
                except Exception as e:
                    st.error(f"실패: {e}")

    with ai_tabs[2]:
        st.subheader("📚 답변 템플릿 라이브러리")
        templates = get_templates()
        if templates:
            for t in templates:
                with st.expander(
                    f"⭐ {t.get('platform', '')} | {t.get('score', '')}점 | {', '.join(t.get('categories', [])[:3])}"
                ):
                    st.markdown(f"> {t.get('reply', '')}")
                    st.caption(f"저장: {t.get('saved_at', '')}")
                    if st.button("🗑 삭제", key=f"del_tmpl_{t['id']}"):
                        db.collection("reply_templates").document(t["id"]).delete()
                        st.cache_data.clear()
                        st.rerun()
        else:
            st.info("저장된 템플릿 없음. 리뷰 관리에서 ⭐ 누르면 추가됨.")


# ═════════════════════════════════════════════════════════
# TAB 7: 운영 인사이트 ⭐
# ═════════════════════════════════════════════════════════
with main_tabs[6]:
    st.header("📈 운영 인사이트")

    insight_tabs = st.tabs([
        "⏱ 응대 SLA",
        "🔄 Before/After 비교",
        "📅 요일/시즌 패턴",
        "👤 직원 언급 트래커",
        "👍 추천 의도 갭",
    ])

    # 응대 SLA
    with insight_tabs[0]:
        st.subheader("⏱ 응대 시간 분석 (SLA)")
        st.caption("리뷰 작성 → 답변 완료 시간을 추적합니다.")

        responded_df = df[df["response_hours"].notna()].copy()
        if len(responded_df) == 0:
            st.info("처리완료된 리뷰가 아직 없어요.")
        else:
            sla1, sla2, sla3, sla4 = st.columns(4)
            sla1.metric("처리완료 리뷰", f"{len(responded_df)}개")

            avg_hours = responded_df["response_hours"].mean()
            sla2.metric("평균 응대 시간", f"{avg_hours:.1f}시간" if avg_hours < 48 else f"{avg_hours/24:.1f}일")

            within_24h = len(responded_df[responded_df["response_hours"] <= 24])
            sla3.metric("24시간 내 응대", f"{within_24h}개",
                        delta=f"{within_24h/len(responded_df)*100:.0f}%")

            within_48h = len(responded_df[responded_df["response_hours"] <= 48])
            sla4.metric("48시간 내 응대", f"{within_48h}개",
                        delta=f"{within_48h/len(responded_df)*100:.0f}%")

            st.markdown("---")

            st.subheader("플랫폼별 응대 시간")
            plat_sla = (
                responded_df.groupby("platform")["response_hours"]
                .agg(["mean", "median", "count"]).reset_index()
                .rename(columns={"mean": "평균(시간)", "median": "중간값(시간)", "count": "건수"})
                .sort_values("평균(시간)")
            )
            plat_sla["평균(시간)"] = plat_sla["평균(시간)"].round(1)
            plat_sla["중간값(시간)"] = plat_sla["중간값(시간)"].round(1)

            fig = px.bar(
                plat_sla, x="평균(시간)", y="platform", orientation="h",
                text="평균(시간)",
                color="평균(시간)",
                color_continuous_scale=["#10b981", "#facc15", "#ef4444"],
                title="플랫폼별 평균 응대 시간 (작을수록 좋음)",
            )
            fig.update_layout(yaxis={"categoryorder": "total descending"}, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(plat_sla, use_container_width=True, hide_index=True)

            st.subheader("응대 시간 분포")
            fig2 = px.histogram(
                responded_df[responded_df["response_hours"] < 168],
                x="response_hours", nbins=30,
                title="응대 시간 분포 (1주 이내)",
                labels={"response_hours": "응대 시간(시간)"},
            )
            fig2.add_vline(x=24, line_dash="dash", line_color="orange", annotation_text="24시간")
            fig2.add_vline(x=48, line_dash="dash", line_color="red", annotation_text="48시간")
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("⚠️ 가장 늦게 답한 리뷰 TOP 10")
            slow = responded_df.sort_values("response_hours", ascending=False).head(10)
            for _, r in slow.iterrows():
                hours = r["response_hours"]
                display = f"{hours:.1f}시간" if hours < 48 else f"{hours/24:.1f}일"
                with st.expander(f"⏰ {display} | {r['platform']} | {r.get('user', '')} | {r.get('score', '')}점"):
                    st.markdown(f"**리뷰일**: {r.get('date', '')}")
                    st.markdown(f"**처리일**: {r.get('completed_at', '')[:19]}")
                    st.markdown(f"**리뷰**: {build_review_text(r)[:200]}")

    # Before/After
    with insight_tabs[1]:
        st.subheader("🔄 Before/After 비교")
        st.caption("두 기간을 비교해서 변화의 효과를 측정하세요.")

        ba_col1, ba_col2 = st.columns(2)
        with ba_col1:
            st.markdown("**Before 기간**")
            before_start = st.date_input("시작일", value=(now - timedelta(days=180)).date(), key="ba_bs")
            before_end = st.date_input("종료일", value=(now - timedelta(days=91)).date(), key="ba_be")
        with ba_col2:
            st.markdown("**After 기간**")
            after_start = st.date_input("시작일", value=(now - timedelta(days=90)).date(), key="ba_as")
            after_end = st.date_input("종료일", value=now.date(), key="ba_ae")

        df_before = df[
            df["date_dt"].notna()
            & (df["date_dt"].dt.date >= before_start)
            & (df["date_dt"].dt.date <= before_end)
        ]
        df_after = df[
            df["date_dt"].notna()
            & (df["date_dt"].dt.date >= after_start)
            & (df["date_dt"].dt.date <= after_end)
        ]

        if len(df_before) == 0 or len(df_after) == 0:
            st.warning("두 기간 모두 리뷰가 있어야 비교 가능합니다.")
        else:
            st.markdown("---")
            st.subheader("📊 핵심 지표 변화")
            bc1, bc2, bc3, bc4 = st.columns(4)

            bc1.metric("리뷰 수", f"{len(df_after)}개",
                       delta=f"{len(df_after) - len(df_before):+d}")

            avg_b = df_before["score_pct"].dropna().mean()
            avg_a = df_after["score_pct"].dropna().mean()
            if not pd.isna(avg_a) and not pd.isna(avg_b):
                bc2.metric("평균 만족도", f"{avg_a:.1f}%", delta=f"{avg_a - avg_b:+.1f}%p")

            neg_b = len(df_before[df_before["score_pct"] <= 50])
            neg_a = len(df_after[df_after["score_pct"] <= 50])
            bc3.metric(
                "부정 리뷰 비율",
                f"{neg_a/len(df_after)*100:.1f}%" if len(df_after) > 0 else "—",
                delta=f"{(neg_a/len(df_after) - neg_b/len(df_before))*100:+.1f}%p" if len(df_before) > 0 else None,
                delta_color="inverse",
            )

            pos_b = len(df_before[df_before["score_pct"] >= 80])
            pos_a = len(df_after[df_after["score_pct"] >= 80])
            bc4.metric(
                "긍정 리뷰 비율",
                f"{pos_a/len(df_after)*100:.1f}%" if len(df_after) > 0 else "—",
                delta=f"{(pos_a/len(df_after) - pos_b/len(df_before))*100:+.1f}%p" if len(df_before) > 0 else None,
            )

            st.markdown("---")
            st.subheader("🏷 카테고리별 변화")

            cat_change = []
            for cat in HOTEL_CATEGORIES.keys():
                b_cat = df_before[df_before["categories_final"].apply(lambda x: cat in (x or []))]
                a_cat = df_after[df_after["categories_final"].apply(lambda x: cat in (x or []))]
                b_score = b_cat["score_pct"].dropna().mean()
                a_score = a_cat["score_pct"].dropna().mean()
                if len(b_cat) > 0 or len(a_cat) > 0:
                    cat_change.append({
                        "카테고리": cat,
                        "Before 언급": len(b_cat),
                        "After 언급": len(a_cat),
                        "Before 점수": round(b_score, 1) if not pd.isna(b_score) else None,
                        "After 점수": round(a_score, 1) if not pd.isna(a_score) else None,
                        "변화(점수)": round(a_score - b_score, 1) if (not pd.isna(a_score) and not pd.isna(b_score)) else None,
                    })

            cat_change_df = pd.DataFrame(cat_change)
            if not cat_change_df.empty:
                st.dataframe(cat_change_df, use_container_width=True, hide_index=True)

                change_chart = cat_change_df.dropna(subset=["변화(점수)"]).copy()
                if not change_chart.empty:
                    change_chart = change_chart.sort_values("변화(점수)")
                    fig = px.bar(
                        change_chart, x="변화(점수)", y="카테고리", orientation="h",
                        color="변화(점수)",
                        color_continuous_scale=["#ef4444", "#facc15", "#10b981"],
                        color_continuous_midpoint=0,
                        title="카테고리별 만족도 변화 (After - Before)",
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

            if st.button("🤖 AI에게 변화 분석 요청"):
                with st.spinner("분석 중..."):
                    b_text = "\n".join([
                        f"[{r['platform']}|{r.get('score','')}점] {build_review_text(r)[:200]}"
                        for _, r in df_before.head(30).iterrows()
                    ])
                    a_text = "\n".join([
                        f"[{r['platform']}|{r.get('score','')}점] {build_review_text(r)[:200]}"
                        for _, r in df_after.head(30).iterrows()
                    ])

                    prompt = f"""호텔 리뷰의 두 기간 비교 분석을 해주세요.

[BEFORE ({before_start} ~ {before_end}, 총 {len(df_before)}건)]
{b_text[:4000]}

[AFTER ({after_start} ~ {after_end}, 총 {len(df_after)}건)]
{a_text[:4000]}

분석해주세요:
1. 새로 등장한 불만
2. 사라진 불만 (개선된 부분)
3. 새로 등장한 칭찬
4. 만족도가 크게 변한 영역
5. 종합 평가"""

                    try:
                        result = call_gemini(prompt)
                        st.info(result)
                    except Exception as e:
                        st.error(f"분석 실패: {e}")

    # 요일/시즌
    with insight_tabs[2]:
        st.subheader("📅 요일별 / 월별 패턴")

        df_pattern = df[df["date_dt"].notna() & df["score_pct"].notna()].copy()
        if len(df_pattern) < 10:
            st.info("패턴 분석할 데이터가 부족합니다.")
        else:
            day_kor = {"Monday": "월", "Tuesday": "화", "Wednesday": "수", "Thursday": "목",
                       "Friday": "금", "Saturday": "토", "Sunday": "일"}
            df_pattern["요일_kor"] = df_pattern["date_dt"].dt.day_name().map(day_kor)
            df_pattern["월"] = df_pattern["date_dt"].dt.month

            st.markdown("### 요일별 평균 만족도")
            day_stats = (
                df_pattern.groupby("요일_kor")
                .agg(평균만족도=("score_pct", "mean"), 리뷰수=("id", "count"))
                .reindex(["월", "화", "수", "목", "금", "토", "일"])
                .reset_index()
            )

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                fig = px.bar(
                    day_stats, x="요일_kor", y="평균만족도",
                    text=day_stats["평균만족도"].round(1).astype(str) + "%",
                    color="평균만족도",
                    color_continuous_scale=["#ef4444", "#facc15", "#10b981"],
                    range_color=[60, 100],
                    title="요일별 평균 만족도",
                )
                fig.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col_d2:
                fig2 = px.bar(
                    day_stats, x="요일_kor", y="리뷰수",
                    text="리뷰수",
                    color_discrete_sequence=["#6366f1"],
                    title="요일별 리뷰 수",
                )
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 월별 패턴")
            month_stats = (
                df_pattern.groupby("월")
                .agg(평균만족도=("score_pct", "mean"), 리뷰수=("id", "count"))
                .reset_index()
            )

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(
                x=month_stats["월"], y=month_stats["리뷰수"],
                name="리뷰 수", marker_color="#cbd5e1", yaxis="y",
            ))
            fig3.add_trace(go.Scatter(
                x=month_stats["월"], y=month_stats["평균만족도"],
                name="평균 만족도", yaxis="y2",
                line=dict(color="#ef4444", width=3), mode="lines+markers",
            ))
            fig3.update_layout(
                title="월별 리뷰 수 & 평균 만족도",
                xaxis=dict(title="월", tickmode="linear"),
                yaxis=dict(title="리뷰 수"),
                yaxis2=dict(title="평균 만족도 (%)", side="right", overlaying="y", range=[0, 100]),
                height=400, hovermode="x unified",
            )
            st.plotly_chart(fig3, use_container_width=True)

            # 인사이트
            day_stats_nona = day_stats.dropna(subset=["평균만족도"])
            month_stats_nona = month_stats.dropna(subset=["평균만족도"])
            if not day_stats_nona.empty and not month_stats_nona.empty:
                worst_day = day_stats_nona.loc[day_stats_nona["평균만족도"].idxmin(), "요일_kor"]
                best_day = day_stats_nona.loc[day_stats_nona["평균만족도"].idxmax(), "요일_kor"]
                worst_month = int(month_stats_nona.loc[month_stats_nona["평균만족도"].idxmin(), "월"])
                best_month = int(month_stats_nona.loc[month_stats_nona["평균만족도"].idxmax(), "월"])

                st.info(
                    f"📌 **인사이트**\n"
                    f"- 가장 만족도 높은 요일: **{best_day}요일**\n"
                    f"- 가장 만족도 낮은 요일: **{worst_day}요일** ← 주의\n"
                    f"- 가장 만족도 높은 달: **{best_month}월**\n"
                    f"- 가장 만족도 낮은 달: **{worst_month}월** ← 주의"
                )

    # 직원
    with insight_tabs[3]:
        st.subheader("👤 직원 이름 언급 트래커")

        staff_names = get_staff_names()

        with st.expander("⚙️ 직원 이름 관리"):
            st.caption("추적할 직원 이름을 등록하세요. 이름이 정확히 들어간 리뷰만 카운트됩니다.")
            current_names = st.text_area(
                "직원 이름 목록 (한 줄에 하나)",
                value="\n".join(staff_names),
                height=150,
                key="staff_textarea",
            )
            if st.button("💾 저장", key="save_staff"):
                names = [n.strip() for n in current_names.split("\n") if n.strip()]
                db.collection("config").document("staff_names").set({"names": names})
                st.success("저장됨!")
                st.cache_data.clear()
                st.rerun()

        if not staff_names:
            st.info("⚠️ 위에서 직원 이름을 먼저 등록하세요. 예: 김지배, 박매니저, 이프론트")
        else:
            staff_data = []
            for name in staff_names:
                mentioned = df[df["full_text"].str.contains(name, na=False)]
                if len(mentioned) == 0:
                    continue
                avg_score = mentioned["score_pct"].dropna().mean()
                pos_count = len(mentioned[mentioned["score_pct"] >= 80])
                neg_count = len(mentioned[mentioned["score_pct"] <= 50])
                staff_data.append({
                    "이름": name,
                    "언급 횟수": len(mentioned),
                    "평균 만족도": round(avg_score, 1) if not pd.isna(avg_score) else None,
                    "긍정 언급": pos_count,
                    "부정 언급": neg_count,
                })

            if not staff_data:
                st.info("등록된 직원 이름이 리뷰에 언급된 적이 없습니다.")
            else:
                staff_df = pd.DataFrame(staff_data).sort_values("언급 횟수", ascending=False)
                st.dataframe(staff_df, use_container_width=True, hide_index=True)

                fig = px.bar(
                    staff_df, x="이름", y=["긍정 언급", "부정 언급"],
                    barmode="group",
                    color_discrete_map={"긍정 언급": "#10b981", "부정 언급": "#ef4444"},
                    title="직원별 긍정/부정 언급",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.markdown("### 🔍 직원별 리뷰 보기")
                sel_staff = st.selectbox("직원 선택", staff_df["이름"].tolist())
                if sel_staff:
                    sel_reviews = df[df["full_text"].str.contains(sel_staff, na=False)]
                    for _, r in sel_reviews.head(15).iterrows():
                        with st.expander(
                            f"{r.get('score', '')}점 | {r.get('user', '')} | {r['platform']} | {r.get('date', '')}"
                        ):
                            text = build_review_text(r)
                            highlighted = text.replace(sel_staff, f"**{sel_staff}**")
                            st.markdown(highlighted)

    # 추천 의도
    with insight_tabs[4]:
        st.subheader("👍 별점 ≠ 추천 의도 갭 분석")
        st.caption("별점은 높은데 재방문 안 한다는 사람, 또는 그 반대 케이스를 찾아냅니다.")

        intent_df = df[df["recommend_intent"].isin(["positive", "neutral", "negative"])]
        if len(intent_df) == 0:
            st.info("**운영 도구 → 추천 의도 분석** 에서 먼저 분류를 실행해주세요.")
        else:
            ic1, ic2, ic3 = st.columns(3)
            pos_intent = len(intent_df[intent_df["recommend_intent"] == "positive"])
            neg_intent = len(intent_df[intent_df["recommend_intent"] == "negative"])
            neu_intent = len(intent_df[intent_df["recommend_intent"] == "neutral"])

            ic1.metric("👍 재방문/추천 의도", f"{pos_intent}개",
                       delta=f"{pos_intent/len(intent_df)*100:.1f}%")
            ic2.metric("😐 중립", f"{neu_intent}개")
            ic3.metric("👎 비추천", f"{neg_intent}개",
                       delta=f"{neg_intent/len(intent_df)*100:.1f}%", delta_color="inverse")

            nps = (pos_intent - neg_intent) / len(intent_df) * 100
            st.metric("🎯 NPS 유사 점수", f"{nps:+.1f}", help="(추천 - 비추천) / 전체. 높을수록 좋음.")

            st.markdown("---")
            st.markdown("### 🚨 별점 ↔ 추천 의도 미스매치")

            gap1 = intent_df[
                (intent_df["score_pct"] >= 80)
                & (intent_df["recommend_intent"] == "negative")
            ]
            gap2 = intent_df[
                (intent_df["score_pct"] <= 50)
                & (intent_df["recommend_intent"] == "positive")
            ]

            st.warning(f"⚠️ 별점 높은데 비추천 의도: **{len(gap1)}개** — 놓치고 있는 고객")
            for _, r in gap1.head(5).iterrows():
                with st.expander(f"{r.get('score', '')}점 | {r.get('user', '')} | {r['platform']}"):
                    st.markdown(build_review_text(r))

            st.success(f"💡 별점 낮은데 추천 의도: **{len(gap2)}개** — 충성 가능성 있는 고객")
            for _, r in gap2.head(5).iterrows():
                with st.expander(f"{r.get('score', '')}점 | {r.get('user', '')} | {r['platform']}"):
                    st.markdown(build_review_text(r))

            st.markdown("---")
            st.markdown("### 📈 월별 추천 의도 추이")
            intent_df_dt = intent_df[intent_df["date_dt"].notna()].copy()
            if len(intent_df_dt) > 0:
                intent_df_dt["month"] = intent_df_dt["date_dt"].dt.to_period("M").dt.start_time
                intent_trend = (
                    intent_df_dt.groupby(["month", "recommend_intent"])
                    .size().reset_index(name="count")
                )
                fig = px.bar(
                    intent_trend, x="month", y="count", color="recommend_intent",
                    color_discrete_map={
                        "positive": "#10b981", "neutral": "#94a3b8", "negative": "#ef4444"
                    },
                    title="월별 추천 의도 분포",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════
# TAB 8: 운영 도구
# ═════════════════════════════════════════════════════════
with main_tabs[7]:
    st.header("⚙️ 운영 도구")

    op_tabs = st.tabs([
        "🏷 카테고리 태깅",
        "👍 추천 의도 분석",
        "📈 답변 품질 추적",
        "🎯 벤치마크 설정",
        "📥 CSV 내보내기",
    ])

    with op_tabs[0]:
        st.subheader("🏷 카테고리 자동 태깅 (Gemini)")
        untagged = df[df["ai_categories"].apply(lambda x: not x or len(x) == 0)]
        st.info(f"AI 태깅 안 된 리뷰: {len(untagged)}개")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            batch_size = st.number_input("한 번에 처리할 개수", 10, 200, 50, 10, key="cat_batch")
        with col_t2:
            st.write("")
            if st.button("🤖 카테고리 태깅 실행", use_container_width=True, disabled=len(untagged) == 0):
                target = untagged.head(batch_size)
                progress = st.progress(0)
                success_count = 0
                fail_count = 0
                for i, (_, row) in enumerate(target.iterrows()):
                    try:
                        prompt = build_category_tagging_prompt(row)
                        result = call_gemini(prompt)
                        result = result.strip()
                        if result.startswith("```"):
                            result = re.sub(r"```(?:json)?\n?", "", result)
                            result = result.rstrip("`").strip()
                        cats = json.loads(result)
                        if isinstance(cats, list):
                            valid_cats = [c for c in cats if c in HOTEL_CATEGORIES.keys()]
                            update_review(row["id"], {"ai_categories": valid_cats})
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception:
                        fail_count += 1
                    progress.progress((i + 1) / len(target))
                st.success(f"✅ {success_count}개 완료 (실패 {fail_count}개)")
                st.cache_data.clear()
                st.rerun()

    with op_tabs[1]:
        st.subheader("👍 추천 의도 자동 분석")
        st.caption("재방문/추천 의도를 Gemini로 분류합니다.")

        no_intent = df[~df["recommend_intent"].isin(["positive", "neutral", "negative"])]
        st.info(f"분석 안 된 리뷰: {len(no_intent)}개")

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            intent_batch = st.number_input("한 번에 처리할 개수", 10, 200, 50, 10, key="intent_batch")
        with col_i2:
            st.write("")
            if st.button("🤖 추천 의도 분석 실행", use_container_width=True, disabled=len(no_intent) == 0):
                target = no_intent.head(intent_batch)
                progress = st.progress(0)
                success_count = 0
                fail_count = 0
                for i, (_, row) in enumerate(target.iterrows()):
                    try:
                        prompt = build_recommend_intent_prompt(row)
                        result = call_gemini(prompt)
                        result = result.strip()
                        if result.startswith("```"):
                            result = re.sub(r"```(?:json)?\n?", "", result)
                            result = result.rstrip("`").strip()
                        data = json.loads(result)
                        intent = data.get("intent", "neutral")
                        if intent in ("positive", "neutral", "negative"):
                            update_review(row["id"], {"recommend_intent": intent})
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception:
                        fail_count += 1
                    progress.progress((i + 1) / len(target))
                st.success(f"✅ {success_count}개 완료 (실패 {fail_count}개)")
                st.cache_data.clear()
                st.rerun()

    with op_tabs[2]:
        st.subheader("📈 답변 품질 추적")
        completed = df[
            (df["status_norm"] == "처리완료")
            & (df["ai_reply"] != "")
            & (df["final_reply"] != "")
        ].copy()

        if len(completed) == 0:
            st.info("처리완료된 리뷰가 아직 없습니다.")
        else:
            completed["edit_ratio"] = completed.apply(
                lambda r: diff_ratio(r["ai_reply"], r["final_reply"]), axis=1
            )

            qk1, qk2, qk3 = st.columns(3)
            qk1.metric("처리완료", f"{len(completed)}개")
            qk2.metric("평균 수정률", f"{completed['edit_ratio'].mean()*100:.1f}%")
            no_edit = len(completed[completed["edit_ratio"] < 0.05])
            qk3.metric("거의 수정 안 함", f"{no_edit}개")

            st.markdown("### 수정률 높았던 답변")
            top_edited = completed.sort_values("edit_ratio", ascending=False).head(10)
            for _, r in top_edited.iterrows():
                with st.expander(
                    f"수정률 {r['edit_ratio']*100:.0f}% | {r['platform']} | {r.get('user', '')}"
                ):
                    st.markdown("**🤖 AI 원본**")
                    st.markdown(f"> {r['ai_reply']}")
                    st.markdown("**✍️ 최종**")
                    st.markdown(f"> {r['final_reply']}")

    with op_tabs[3]:
        st.subheader("🎯 벤치마크 값 설정")
        benchmark = get_benchmark()
        bm_col1, bm_col2 = st.columns(2)
        with bm_col1:
            jeju_avg = st.number_input("제주 4-5성 평균 (%)", 0.0, 100.0,
                                       float(benchmark.get("jeju_avg") or 85.0), 0.1)
        with bm_col2:
            competitor_avg = st.number_input("경쟁사 평균 (%)", 0.0, 100.0,
                                             float(benchmark.get("competitor_avg") or 87.0), 0.1)
        note = st.text_area("메모", value=benchmark.get("note", ""))

        if st.button("💾 저장", use_container_width=True, key="save_bm"):
            db.collection("config").document("benchmark").set({
                "jeju_avg": jeju_avg,
                "competitor_avg": competitor_avg,
                "note": note,
                "updated_at": datetime.now().isoformat(),
            })
            st.success("저장 완료!")
            st.cache_data.clear()

    with op_tabs[4]:
        st.subheader("📥 CSV / Excel 내보내기")

        ex_platforms = st.multiselect(
            "플랫폼", sorted(df["platform"].unique()),
            default=sorted(df["platform"].unique()),
        )
        ex_status = st.multiselect(
            "상태", ["대기중", "답변완료", "처리완료"],
            default=["대기중", "답변완료", "처리완료"],
        )

        ex_df = df[df["platform"].isin(ex_platforms) & df["status_norm"].isin(ex_status)]

        out_cols = [
            "platform", "date", "user", "country", "score", "score_pct",
            "title", "positive", "negative", "content",
            "categories_final", "recommend_intent", "status_norm",
            "ai_reply", "final_reply", "owner_reply",
            "room_type", "traveler_type", "booking_id",
        ]
        out_df = ex_df[[c for c in out_cols if c in ex_df.columns]].copy()
        col_names = [
            "플랫폼", "리뷰일자", "작성자", "국적", "점수", "만족도(%)",
            "제목", "좋았던점", "아쉬운점", "본문",
            "카테고리", "추천의도", "상태",
            "AI답변", "최종답변", "기존호텔답변",
            "객실타입", "투숙유형", "예약번호",
        ]
        out_df.columns = col_names[:len(out_df.columns)]

        st.dataframe(out_df.head(20), use_container_width=True)
        st.caption(f"미리보기 (상위 20개) / 전체 {len(out_df)}개")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            csv = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "📥 CSV", csv,
                file_name=f"amber_reviews_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv", use_container_width=True,
            )
        with col_ex2:
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    out_df.to_excel(writer, index=False, sheet_name="리뷰")
                st.download_button(
                    "📥 Excel", output.getvalue(),
                    file_name=f"amber_reviews_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except ImportError:
                st.caption("`openpyxl` 필요")
