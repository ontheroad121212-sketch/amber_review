"""
앰버 7대 플랫폼 통합 AI 지배인 (Streamlit) v3 - 대시보드 + 분석 풀버전
주요 추가:
- 대시보드 (KPI, 시계열, 플랫폼별 차트)
- 키워드 분석 (워드클라우드, 호텔 카테고리, 키워드 추이)
- 카테고리 자동 태깅 (Gemini)
- 답변 품질 추적
- 벤치마크 비교
- CSV 내보내기
- 자유 질의 (RAG)
- 답변 템플릿 라이브러리
- 국적/투숙유형/객실 타입 분석
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
# 3. 호텔 도메인 카테고리 사전
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
    "와이파이": ["와이파이", "wifi", "인터넷", "와이파이"],
}


# ─────────────────────────────────────────────────────────
# 4. 데이터 함수
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def get_reviews():
    """Firestore에서 전체 리뷰 조회"""
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
            d.setdefault("ai_categories", [])  # 카테고리 자동 태깅 결과
            d.setdefault("final_reply", "")    # 최종 확정 답변 (품질 추적용)
            data.append(d)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"데이터 로딩 실패: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_benchmark():
    """벤치마크 값 가져오기"""
    try:
        doc = db.collection("config").document("benchmark").get()
        if doc.exists:
            return doc.to_dict()
        return {"jeju_avg": None, "competitor_avg": None, "note": ""}
    except:
        return {"jeju_avg": None, "competitor_avg": None, "note": ""}


@st.cache_data(ttl=300)
def get_templates():
    """답변 템플릿 가져오기"""
    try:
        docs = db.collection("reply_templates").stream()
        return [{"id": d.id, **d.to_dict()} for d in docs]
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
    """점수를 0-100 백분율로 정규화"""
    if score_val is None or pd.isna(score_val):
        return None
    if platform in ("구글(Google)", "트립어드바이저(TripAdvisor)"):
        return score_val * 20
    else:
        return score_val * 10


def parse_date(date_str):
    """다양한 날짜 포맷을 datetime 으로 시도 변환. 실패하면 None."""
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
    # 한국식 "2024년 3월 15일"
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
- 마치 사람이 한 명씩 직접 손으로 쓴 듯 자연스럽게.

[금지 표현 — 절대 사용 금지]
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
- 첫 문장: "안녕하세요, 엠버퓨어힐입니다." 정도로 짧고 자연스럽게.
- 작성자 이름이 있으면 자연스럽게 한 번 호명.
- 리뷰에서 **구체적으로 언급된 부분**을 답변에도 한두 가지 짚어주세요. 두루뭉술 금지.
- 좋았던 점 → 무엇이 어떻게 좋았다는지 받아 적고 감사.
- 아쉬운 점 → 변명 없이 인정. 가능하면 어떤 부분을 어떻게 살펴보겠다는 식의 구체적인 한 마디.
- 마지막은 재방문 환영.
- 분량: **200~350자** 사이.
- 이모티콘: 0~1개.
- 영업/마케팅 표현 금지.

[점수 가이드]
{score_guide}
{template_section}
[답변할 리뷰]
플랫폼: {platform}
작성자: {user or "(이름 없음)"}
점수: {score_display or "(없음)"}
{review_text}

위 리뷰에 대한 호텔 답변만 작성해 주세요. 따옴표나 서두 설명 없이 답변 본문만 출력합니다.
"""


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
- 여러 개 가능
- 해당 없으면 빈 배열 []
- 다른 설명 없이 JSON 배열만 출력

예: ["조식", "직원/서비스"]
"""


def update_review(doc_id, fields):
    db.collection("reviews").document(doc_id).update(fields)


def detect_categories_keyword(text):
    """키워드 기반 카테고리 감지 (Gemini 없이 빠르게)"""
    if not text:
        return []
    found = []
    text_lower = text.lower()
    for cat, keywords in HOTEL_CATEGORIES.items():
        if any(kw in text_lower for kw in keywords):
            found.append(cat)
    return found


def diff_ratio(original, edited):
    """원본 vs 수정본 변경률 (0~1)"""
    if not original or not edited:
        return 0
    o, e = original.strip(), edited.strip()
    if o == e:
        return 0
    longer = max(len(o), len(e))
    if longer == 0:
        return 0
    # 간단한 글자 단위 차이
    import difflib
    sm = difflib.SequenceMatcher(None, o, e)
    return 1 - sm.ratio()


# ─────────────────────────────────────────────────────────
# 5. 데이터 로딩
# ─────────────────────────────────────────────────────────
st.title("🏨 앰버 통합 리뷰 AI 지배인")

df = get_reviews()

if df.empty:
    st.info("아직 수집된 리뷰가 없습니다. 크롤러를 먼저 실행해 주세요.")
    st.stop()

# 공통 전처리
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
# 키워드 기반 즉시 카테고리 (ai_categories 없을 때 fallback)
df["categories_auto"] = df["full_text"].apply(detect_categories_keyword)
df["categories_final"] = df.apply(
    lambda r: r["ai_categories"] if r["ai_categories"] else r["categories_auto"], axis=1
)

# ─────────────────────────────────────────────────────────
# 6. 사이드바
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 전체 현황")
    total = len(df)
    waiting = len(df[df["status_norm"] == "대기중"])
    replied = len(df[df["status_norm"] == "답변완료"])
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

    st.markdown("---")
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption(f"🕐 로딩 시각: {datetime.now().strftime('%H:%M:%S')}")


# ─────────────────────────────────────────────────────────
# 7. 메인 탭 구조
# ─────────────────────────────────────────────────────────
main_tabs = st.tabs(
    ["📊 대시보드", "🔍 키워드 분석", "📝 리뷰 관리", "🤖 AI 분석", "⚙️ 운영 도구"]
)


# ═════════════════════════════════════════════════════════
# TAB 1: 대시보드
# ═════════════════════════════════════════════════════════
with main_tabs[0]:
    st.header("📊 대시보드")

    # ── 기간 필터 ──
    period_col1, period_col2 = st.columns([3, 1])
    with period_col2:
        period = st.selectbox(
            "기간",
            ["전체", "최근 30일", "최근 90일", "최근 1년", "올해"],
            key="dash_period",
        )

    df_dash = df.copy()
    now = datetime.now()
    if period == "최근 30일":
        cutoff = now - timedelta(days=30)
        df_dash = df_dash[df_dash["date_dt"].notna() & (df_dash["date_dt"] >= cutoff)]
    elif period == "최근 90일":
        cutoff = now - timedelta(days=90)
        df_dash = df_dash[df_dash["date_dt"].notna() & (df_dash["date_dt"] >= cutoff)]
    elif period == "최근 1년":
        cutoff = now - timedelta(days=365)
        df_dash = df_dash[df_dash["date_dt"].notna() & (df_dash["date_dt"] >= cutoff)]
    elif period == "올해":
        df_dash = df_dash[df_dash["date_dt"].notna() & (df_dash["date_dt"].dt.year == now.year)]

    with period_col1:
        st.caption(f"분석 대상: **{len(df_dash):,}개** 리뷰 ({period})")

    # ── KPI 카드 (이번달 vs 지난달) ──
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
    k1.metric(
        "이번 달 리뷰 수",
        f"{len(df_this)}개",
        delta=f"{len(df_this) - len(df_last):+d} vs 지난달",
    )
    if not pd.isna(avg_this):
        delta = f"{avg_this - avg_last:+.1f}%p" if not pd.isna(avg_last) else None
        k2.metric("이번 달 만족도", f"{avg_this:.1f}%", delta=delta)
    else:
        k2.metric("이번 달 만족도", "—")

    low_this = df_this[df_this["score_pct"].notna() & (df_this["score_pct"] <= 50)]
    low_last = df_last[df_last["score_pct"].notna() & (df_last["score_pct"] <= 50)]
    k3.metric(
        "🚨 부정 리뷰 (이번달)",
        f"{len(low_this)}개",
        delta=f"{len(low_this) - len(low_last):+d}",
        delta_color="inverse",
    )

    response_rate = (
        len(df_dash[df_dash["status_norm"].isin(["답변완료", "처리완료"])])
        / len(df_dash)
        * 100
        if len(df_dash) > 0
        else 0
    )
    k4.metric("응답률", f"{response_rate:.1f}%")

    st.markdown("---")

    # ── 플랫폼별 평균 만족도 ──
    st.subheader("🏢 플랫폼별 평균 만족도")
    plat_stats = (
        df_dash.groupby("platform")
        .agg(
            평균만족도=("score_pct", "mean"),
            리뷰수=("score_pct", "count"),
            응답완료=("status_norm", lambda x: (x.isin(["답변완료", "처리완료"])).sum()),
        )
        .reset_index()
    )
    plat_stats = plat_stats.sort_values("리뷰수", ascending=False)
    plat_stats["응답률"] = (plat_stats["응답완료"] / plat_stats["리뷰수"] * 100).round(1)

    if not plat_stats.empty:
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig = px.bar(
                plat_stats,
                x="평균만족도",
                y="platform",
                orientation="h",
                text=plat_stats["평균만족도"].round(1).astype(str) + "%",
                color="평균만족도",
                color_continuous_scale=["#ef4444", "#facc15", "#10b981"],
                range_color=[40, 100],
                title="플랫폼별 평균 만족도 (%)",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col_chart2:
            fig2 = px.bar(
                plat_stats,
                x="리뷰수",
                y="platform",
                orientation="h",
                text="리뷰수",
                color_discrete_sequence=["#6366f1"],
                title="플랫폼별 리뷰 수",
            )
            fig2.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                height=400,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # 테이블
        with st.expander("플랫폼별 상세 표 보기"):
            display_df = plat_stats.copy()
            display_df["평균만족도"] = display_df["평균만족도"].round(1).astype(str) + "%"
            display_df["응답률"] = display_df["응답률"].astype(str) + "%"
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 시계열 트렌드 ──
    st.subheader("📅 시계열 추이")
    df_ts = df_dash[df_dash["date_dt"].notna()].copy()
    if len(df_ts) > 0:
        granularity = st.radio(
            "단위", ["주별", "월별"], horizontal=True, key="ts_gran"
        )
        if granularity == "주별":
            df_ts["period"] = df_ts["date_dt"].dt.to_period("W").dt.start_time
        else:
            df_ts["period"] = df_ts["date_dt"].dt.to_period("M").dt.start_time

        ts_agg = (
            df_ts.groupby("period")
            .agg(리뷰수=("id", "count"), 평균만족도=("score_pct", "mean"))
            .reset_index()
            .sort_values("period")
        )

        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Bar(
                x=ts_agg["period"],
                y=ts_agg["리뷰수"],
                name="리뷰 수",
                yaxis="y",
                marker_color="#cbd5e1",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=ts_agg["period"],
                y=ts_agg["평균만족도"],
                name="평균 만족도 (%)",
                yaxis="y2",
                line=dict(color="#ef4444", width=3),
                mode="lines+markers",
            )
        )
        fig_ts.update_layout(
            title=f"{granularity} 리뷰 수 & 평균 만족도",
            xaxis_title="기간",
            yaxis=dict(title="리뷰 수", side="left"),
            yaxis2=dict(
                title="평균 만족도 (%)",
                side="right",
                overlaying="y",
                range=[0, 100],
            ),
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # 변화 감지
        if len(ts_agg) >= 2:
            last_val = ts_agg["평균만족도"].iloc[-1]
            prev_val = ts_agg["평균만족도"].iloc[-2]
            if not pd.isna(last_val) and not pd.isna(prev_val):
                diff = last_val - prev_val
                if abs(diff) >= 5:
                    if diff < 0:
                        st.warning(
                            f"⚠️ 직전 {granularity[:-1]} 대비 만족도가 **{abs(diff):.1f}%p 하락**했어요. 점검 필요!"
                        )
                    else:
                        st.success(
                            f"📈 직전 {granularity[:-1]} 대비 만족도가 **{diff:.1f}%p 상승**했어요!"
                        )
    else:
        st.info("날짜 정보가 있는 리뷰가 부족합니다.")

    st.markdown("---")

    # ── 카테고리별 만족도 ──
    st.subheader("🏷️ 카테고리별 언급 & 만족도")

    cat_data = []
    for cat in HOTEL_CATEGORIES.keys():
        cat_reviews = df_dash[df_dash["categories_final"].apply(lambda x: cat in (x or []))]
        if len(cat_reviews) == 0:
            continue
        cat_data.append(
            {
                "카테고리": cat,
                "언급수": len(cat_reviews),
                "평균만족도": cat_reviews["score_pct"].dropna().mean(),
            }
        )
    cat_df = pd.DataFrame(cat_data).sort_values("언급수", ascending=False) if cat_data else pd.DataFrame()

    if not cat_df.empty:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_c1 = px.bar(
                cat_df,
                x="언급수",
                y="카테고리",
                orientation="h",
                text="언급수",
                color_discrete_sequence=["#6366f1"],
                title="카테고리별 언급 빈도",
            )
            fig_c1.update_layout(
                yaxis={"categoryorder": "total ascending"}, height=450
            )
            st.plotly_chart(fig_c1, use_container_width=True)
        with col_c2:
            fig_c2 = px.bar(
                cat_df,
                x="평균만족도",
                y="카테고리",
                orientation="h",
                text=cat_df["평균만족도"].round(1).astype(str) + "%",
                color="평균만족도",
                color_continuous_scale=["#ef4444", "#facc15", "#10b981"],
                range_color=[40, 100],
                title="카테고리별 평균 만족도",
            )
            fig_c2.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                height=450,
            )
            st.plotly_chart(fig_c2, use_container_width=True)
    else:
        st.info("카테고리 데이터 부족. '운영 도구 → 카테고리 자동 태깅' 을 실행하면 더 정확해집니다.")

    st.markdown("---")

    # ── 국적/투숙유형/객실타입 분석 ──
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
            .reset_index()
            .sort_values("리뷰수", ascending=False)
            .head(15)
        )
        fig = px.bar(
            agg,
            x="리뷰수",
            y=col_name,
            orientation="h",
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

    # ── 벤치마크 비교 ──
    st.subheader("📊 벤치마크 비교")
    benchmark = get_benchmark()
    our_avg = df_dash["score_pct"].dropna().mean()

    bk1, bk2, bk3 = st.columns(3)
    bk1.metric("우리 호텔", f"{our_avg:.1f}%" if not pd.isna(our_avg) else "—")
    if benchmark.get("jeju_avg") is not None:
        diff = our_avg - benchmark["jeju_avg"] if not pd.isna(our_avg) else None
        bk2.metric(
            "제주 4-5성 평균",
            f"{benchmark['jeju_avg']:.1f}%",
            delta=f"{diff:+.1f}%p" if diff is not None else None,
        )
    else:
        bk2.metric("제주 4-5성 평균", "미설정", help="운영 도구에서 입력하세요")

    if benchmark.get("competitor_avg") is not None:
        diff = our_avg - benchmark["competitor_avg"] if not pd.isna(our_avg) else None
        bk3.metric(
            "주요 경쟁사 평균",
            f"{benchmark['competitor_avg']:.1f}%",
            delta=f"{diff:+.1f}%p" if diff is not None else None,
        )
    else:
        bk3.metric("주요 경쟁사 평균", "미설정")

    if benchmark.get("note"):
        st.caption(f"📝 {benchmark['note']}")


# ═════════════════════════════════════════════════════════
# TAB 2: 키워드 분석
# ═════════════════════════════════════════════════════════
with main_tabs[1]:
    st.header("🔍 키워드 분석")

    # 형태소 분석기는 옵션 (없어도 동작)
    use_morph = False
    try:
        from kiwipiepy import Kiwi
        @st.cache_resource
        def get_kiwi():
            return Kiwi()
        kiwi = get_kiwi()
        use_morph = True
    except ImportError:
        st.caption("💡 `kiwipiepy` 설치 시 더 정확한 한국어 키워드 분석이 가능합니다.")

    # 불용어
    STOPWORDS = {
        "있다", "없다", "하다", "되다", "이다", "그", "저", "이", "그리고", "근데",
        "정말", "진짜", "너무", "조금", "약간", "그런", "이런", "저런", "또한", "또",
        "수", "것", "거", "게", "수가", "안", "잘", "더", "좀", "다", "도", "은",
        "는", "이", "가", "을", "를", "에", "의", "와", "과", "에서", "로", "으로",
        "호텔", "방", "객실",  # 너무 자주 나와서 의미 없음
    }

    def extract_keywords(texts, top_n=50):
        """형태소 분석 + 빈도"""
        if use_morph:
            counter = Counter()
            for text in texts:
                if not text:
                    continue
                try:
                    tokens = kiwi.tokenize(str(text))
                    for t in tokens:
                        # 명사(NNG, NNP) 또는 형용사 어간(VA)
                        if t.tag in ("NNG", "NNP", "VA"):
                            word = t.form
                            if len(word) >= 2 and word not in STOPWORDS:
                                counter[word] += 1
                except:
                    continue
            return counter.most_common(top_n)
        else:
            # fallback: 단순 공백 분리
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
    # positive/negative 컬럼이 따로 있는 경우 (부킹닷컴식)
    pos_texts += df[df["positive"] != ""]["positive"].tolist()
    neg_texts += df[df["negative"] != ""]["negative"].tolist()

    col_kw1, col_kw2 = st.columns(2)

    with col_kw1:
        st.markdown("### 😊 긍정 리뷰 키워드 TOP 30")
        pos_kw = extract_keywords(pos_texts, top_n=30)
        if pos_kw:
            kw_df = pd.DataFrame(pos_kw, columns=["키워드", "빈도"])
            fig = px.bar(
                kw_df.head(20),
                x="빈도",
                y="키워드",
                orientation="h",
                color="빈도",
                color_continuous_scale="Greens",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("전체 키워드 보기"):
                st.dataframe(kw_df, use_container_width=True, hide_index=True)
        else:
            st.info("긍정 리뷰 키워드 데이터 부족")

    with col_kw2:
        st.markdown("### 😞 부정 리뷰 키워드 TOP 30")
        neg_kw = extract_keywords(neg_texts, top_n=30)
        if neg_kw:
            kw_df = pd.DataFrame(neg_kw, columns=["키워드", "빈도"])
            fig = px.bar(
                kw_df.head(20),
                x="빈도",
                y="키워드",
                orientation="h",
                color="빈도",
                color_continuous_scale="Reds",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                showlegend=False,
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("전체 키워드 보기"):
                st.dataframe(kw_df, use_container_width=True, hide_index=True)
        else:
            st.info("부정 리뷰 키워드 데이터 부족")

    st.markdown("---")

    # ── 워드클라우드 ──
    st.subheader("☁️ 워드클라우드")
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        import os

        # 한글 폰트 찾기
        font_path = None
        for f in fm.findSystemFonts():
            name = f.lower()
            if "nanum" in name or "malgun" in name or "applegothic" in name or "noto" in name and "kr" in name:
                font_path = f
                break

        wc_col1, wc_col2 = st.columns(2)

        def make_wc(text, colormap):
            if not text:
                return None
            try:
                wc = WordCloud(
                    width=600,
                    height=400,
                    background_color="white",
                    font_path=font_path,
                    colormap=colormap,
                    max_words=80,
                ).generate(text)
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                return fig
            except Exception as e:
                st.caption(f"워드클라우드 생성 오류: {e}")
                return None

        with wc_col1:
            st.markdown("**긍정 워드클라우드**")
            pos_text = " ".join([w for w, _ in pos_kw]) if pos_kw else ""
            fig = make_wc(pos_text, "Greens")
            if fig:
                st.pyplot(fig)
            else:
                st.info("데이터 부족")
        with wc_col2:
            st.markdown("**부정 워드클라우드**")
            neg_text = " ".join([w for w, _ in neg_kw]) if neg_kw else ""
            fig = make_wc(neg_text, "Reds")
            if fig:
                st.pyplot(fig)
            else:
                st.info("데이터 부족")

        if not font_path:
            st.caption(
                "⚠️ 시스템에 한글 폰트가 없어 한글이 □ 로 보일 수 있어요. "
                "Streamlit Cloud 사용 시 `packages.txt` 에 `fonts-nanum` 을 추가하세요."
            )
    except ImportError:
        st.warning("`wordcloud` 패키지가 필요합니다. `requirements.txt` 에 `wordcloud` 를 추가하세요.")

    st.markdown("---")

    # ── 카테고리별 키워드 추이 ──
    st.subheader("📈 카테고리별 언급 추이")
    df_ts2 = df[df["date_dt"].notna()].copy()
    if len(df_ts2) > 0:
        sel_categories = st.multiselect(
            "추적할 카테고리 선택",
            list(HOTEL_CATEGORIES.keys()),
            default=["조식", "직원/서비스", "청결", "객실"],
        )

        if sel_categories:
            df_ts2["month"] = df_ts2["date_dt"].dt.to_period("M").dt.start_time
            trend_data = []
            for cat in sel_categories:
                cat_df = df_ts2[df_ts2["categories_final"].apply(lambda x: cat in (x or []))]
                monthly = cat_df.groupby("month").size().reset_index(name="언급수")
                monthly["카테고리"] = cat
                trend_data.append(monthly)

            if trend_data:
                trend_df = pd.concat(trend_data)
                fig = px.line(
                    trend_df,
                    x="month",
                    y="언급수",
                    color="카테고리",
                    markers=True,
                    title="월별 카테고리 언급 추이",
                )
                fig.update_layout(height=450, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════
# TAB 3: 리뷰 관리 (기존 + 카테고리 필터 추가)
# ═════════════════════════════════════════════════════════
with main_tabs[2]:
    st.header("📝 리뷰 관리")

    # ── 필터 ──
    with st.expander("🔍 필터 옵션", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            platforms_all = sorted(df["platform"].unique())
            platforms_sel = st.multiselect(
                "플랫폼", platforms_all, default=platforms_all, key="rm_plat"
            )
            status_sel = st.multiselect(
                "처리 상태",
                ["대기중", "답변완료", "처리완료"],
                default=["대기중"],
                key="rm_status",
            )
        with f_col2:
            score_filter = st.select_slider(
                "점수 범위",
                options=["전체", "낮음 (≤50%)", "중간 (50~80%)", "높음 (≥80%)"],
                value="전체",
                key="rm_score",
            )
            category_sel = st.multiselect(
                "카테고리 필터",
                list(HOTEL_CATEGORIES.keys()),
                default=[],
                key="rm_cat",
            )
        with f_col3:
            search_keyword = st.text_input(
                "🔎 본문 검색",
                placeholder="예: 조식, 청결...",
                key="rm_search",
            )
            sort_order = st.selectbox(
                "정렬",
                [
                    "최신 수집순",
                    "오래된 수집순",
                    "낮은 점수 먼저",
                    "높은 점수 먼저",
                    "리뷰 작성일 (최신순)",
                    "리뷰 작성일 (오래된순)",
                ],
                key="rm_sort",
            )

    # 필터 적용
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

    if search_keyword:
        kw = search_keyword.lower()
        mask = (
            filtered["content"].str.lower().str.contains(kw, na=False)
            | filtered["title"].str.lower().str.contains(kw, na=False)
            | filtered["positive"].str.lower().str.contains(kw, na=False)
            | filtered["negative"].str.lower().str.contains(kw, na=False)
            | filtered["satisfaction_tags"].str.lower().str.contains(kw, na=False)
        )
        filtered = filtered[mask]

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

    # 상단 요약
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

    if len(low_score_waiting) > 0:
        st.warning(
            f"🚨 점수가 낮은 리뷰 {len(low_score_waiting)}개가 답변을 기다리고 있습니다."
        )

    st.markdown("---")

    # 일괄 답변
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
        tabs = st.tabs(
            [f"{p} ({len(filtered[filtered['platform']==p])})" for p in platforms_in_filter]
        )

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
                        row.get("title")
                        or row.get("negative")
                        or row.get("positive")
                        or row.get("content", "")
                    )
                    preview = preview.replace("\n", " ")[:50]
                    header_parts.append(f"| {preview}...")

                    # 카테고리 태그 표시
                    cats = row.get("categories_final") or []
                    if cats:
                        header_parts.append(f"🏷 {','.join(cats[:3])}")

                    is_waiting = status == "대기중"
                    header = " ".join(header_parts)

                    with st.expander(header, expanded=is_waiting):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            if row.get("title"):
                                st.markdown(f"### {row['title']}")

                            if row.get("positive") or row.get("negative"):
                                if row.get("positive"):
                                    st.markdown("**😊 좋았던 점**")
                                    st.markdown(f"> {row['positive']}")
                                if row.get("negative"):
                                    st.markdown("**😞 아쉬운 점**")
                                    st.markdown(f"> {row['negative']}")
                            else:
                                st.markdown("**[원문]**")
                                st.markdown(f"> {row.get('content', '(내용 없음)')}")

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
                                with st.expander("📩 호텔 측 기존 답변 보기"):
                                    st.markdown(f"> {row['owner_reply']}")

                            st.markdown("---")

                            if not row.get("ai_reply"):
                                if st.button(
                                    "🤖 AI 답변 초안 만들기", key=f"btn_{unique_key}"
                                ):
                                    with st.spinner("AI가 답변을 작성하는 중..."):
                                        try:
                                            templates = get_templates()
                                            prompt = build_reply_prompt(
                                                row, similar_templates=templates
                                            )
                                            reply = call_gemini(prompt)
                                            update_review(
                                                row["id"], {"ai_reply": reply}
                                            )
                                            st.cache_data.clear()
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"AI 답변 생성 실패: {e}")
                            else:
                                st.markdown("**✍️ AI 답변 초안 (수정 가능)**")
                                edited_reply = st.text_area(
                                    "답변 수정",
                                    value=row["ai_reply"],
                                    height=180,
                                    key=f"edit_{unique_key}",
                                    label_visibility="collapsed",
                                )
                                st.code(edited_reply, language=None)

                                # 수정률 표시
                                if edited_reply != row["ai_reply"]:
                                    ratio = diff_ratio(row["ai_reply"], edited_reply)
                                    st.caption(f"📝 AI 원본 대비 변경률: {ratio*100:.1f}%")

                                c_a, c_b, c_c, c_d = st.columns(4)
                                with c_a:
                                    st.link_button("🌐 관리자", admin_url)
                                with c_b:
                                    if st.button(
                                        "🔄 재생성", key=f"regen_{unique_key}"
                                    ):
                                        with st.spinner("재생성 중..."):
                                            try:
                                                templates = get_templates()
                                                prompt = build_reply_prompt(
                                                    row, similar_templates=templates
                                                )
                                                reply = call_gemini(prompt)
                                                update_review(
                                                    row["id"], {"ai_reply": reply}
                                                )
                                                st.cache_data.clear()
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"재생성 실패: {e}")
                                with c_c:
                                    if st.button(
                                        "✅ 확정 완료", key=f"confirm_{unique_key}"
                                    ):
                                        update_review(
                                            row["id"],
                                            {
                                                "ai_reply": edited_reply,
                                                "final_reply": edited_reply,
                                                "status": "처리완료",
                                                "completed_at": datetime.now().isoformat(),
                                            },
                                        )
                                        st.cache_data.clear()
                                        st.rerun()
                                with c_d:
                                    if st.button(
                                        "⭐ 템플릿 저장",
                                        key=f"tmpl_{unique_key}",
                                        help="좋은 답변을 템플릿으로 저장",
                                    ):
                                        db.collection("reply_templates").add(
                                            {
                                                "reply": edited_reply,
                                                "platform": row["platform"],
                                                "score": row.get("score", ""),
                                                "categories": row.get(
                                                    "categories_final", []
                                                ),
                                                "saved_at": datetime.now().isoformat(),
                                            }
                                        )
                                        st.success("템플릿으로 저장됨!")
                                        st.cache_data.clear()

                        with col2:
                            st.write(f"**상태**: {status}")
                            if row.get("score"):
                                score_val = score_to_float(row["score"])
                                if score_val is not None:
                                    score_pct = score_to_pct(score_val, row["platform"])
                                    max_score = (
                                        5
                                        if row["platform"]
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

                            if cats:
                                st.write("**카테고리**")
                                st.write(", ".join(cats))

                            if row.get("booking_id"):
                                st.caption(f"예약: `{row['booking_id']}`")
                            if row.get("review_id"):
                                st.caption(f"리뷰: `{row['review_id']}`")

                            if status == "답변완료":
                                if st.button(
                                    "↩️ 대기로", key=f"undo_{unique_key}"
                                ):
                                    update_review(
                                        row["id"],
                                        {"status": "대기중", "has_reply": False},
                                    )
                                    st.cache_data.clear()
                                    st.rerun()
                            elif status == "대기중":
                                if st.button(
                                    "✅ 답변완료 표시",
                                    key=f"mark_replied_{unique_key}",
                                ):
                                    update_review(
                                        row["id"],
                                        {"status": "답변완료", "has_reply": True},
                                    )
                                    st.cache_data.clear()
                                    st.rerun()


# ═════════════════════════════════════════════════════════
# TAB 4: AI 분석
# ═════════════════════════════════════════════════════════
with main_tabs[3]:
    st.header("🤖 AI 분석")

    ai_tabs = st.tabs(["💬 자유 질의", "📊 종합 보고서", "📚 답변 템플릿"])

    # ── 자유 질의 (RAG) ──
    with ai_tabs[0]:
        st.subheader("💬 리뷰 데이터에 자유롭게 질문하기")
        st.caption("예: 지난 3개월간 조식 관련 가장 큰 불만은? / 일본 고객들이 좋아한 점은? / 청결 문제가 많은 객실 타입은?")

        col_q1, col_q2 = st.columns([3, 1])
        with col_q1:
            user_query = st.text_input(
                "질문",
                placeholder="리뷰 데이터에 대해 궁금한 점을 자유롭게 물어보세요",
                key="rag_query",
            )
        with col_q2:
            sample_size = st.number_input(
                "분석 리뷰 수", min_value=20, max_value=200, value=80, step=10
            )

        if st.button("🔍 AI에게 질문", use_container_width=True, disabled=not user_query):
            with st.spinner("AI가 데이터를 분석하는 중..."):
                # 질문에 카테고리/키워드 매칭되는 리뷰 우선
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
                    st.success("답변:")
                    st.info(answer)
                except Exception as e:
                    st.error(f"분석 실패: {e}")

    # ── 종합 보고서 ──
    with ai_tabs[1]:
        st.subheader("📊 종합 분석 보고서")
        col_r1, col_r2 = st.columns([3, 1])
        with col_r1:
            report_period = st.selectbox(
                "분석 기간",
                ["전체", "최근 30일", "최근 90일"],
                key="report_period",
            )
        with col_r2:
            report_sample = st.number_input(
                "분석 리뷰 수",
                min_value=20,
                max_value=200,
                value=80,
                step=10,
                key="rep_sample",
            )

        if st.button("📑 보고서 생성", use_container_width=True):
            with st.spinner("AI가 종합 보고서를 작성하는 중..."):
                df_rep = df.copy()
                if report_period == "최근 30일":
                    cutoff = datetime.now() - timedelta(days=30)
                    df_rep = df_rep[
                        df_rep["date_dt"].notna() & (df_rep["date_dt"] >= cutoff)
                    ]
                elif report_period == "최근 90일":
                    cutoff = datetime.now() - timedelta(days=90)
                    df_rep = df_rep[
                        df_rep["date_dt"].notna() & (df_rep["date_dt"] >= cutoff)
                    ]

                sample_df = df_rep.head(report_sample)
                review_blocks = []
                for _, r in sample_df.iterrows():
                    block = f"[{r['platform']}|{r.get('score','')}점] {build_review_text(r)[:300]}"
                    review_blocks.append(block)
                all_text = "\n\n".join(review_blocks)[:10000]

                prompt = f"""당신은 호텔 컨설팅 전문가입니다. 아래 리뷰들을 분석해 보고서를 작성하세요.

[보고서 구성]
## 1. 핵심 강점 Best 3
구체적 표현 인용해서

## 2. 개선 필요 사항 Worst 3
구체적 사례 + 우선순위

## 3. 플랫폼별 고객 성향 차이
(있는 경우만)

## 4. 이번 주 운영 전략 제안
실행 가능한 3가지

[리뷰 데이터 (총 {len(sample_df)}건)]
{all_text}

[보고서]"""

                try:
                    report = call_gemini(prompt)
                    st.success("✅ 보고서 작성 완료")
                    st.markdown(report)

                    # 다운로드
                    st.download_button(
                        "📥 보고서 다운로드 (텍스트)",
                        report,
                        file_name=f"amber_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    )
                except Exception as e:
                    st.error(f"분석 실패: {e}")

    # ── 답변 템플릿 ──
    with ai_tabs[2]:
        st.subheader("📚 답변 템플릿 라이브러리")
        st.caption("좋은 답변을 저장해두면 AI가 답변 생성 시 참고합니다.")

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
            st.info(
                "아직 저장된 템플릿이 없습니다. 리뷰 관리 탭에서 좋은 답변에 '⭐ 템플릿 저장' 을 눌러 추가하세요."
            )


# ═════════════════════════════════════════════════════════
# TAB 5: 운영 도구
# ═════════════════════════════════════════════════════════
with main_tabs[4]:
    st.header("⚙️ 운영 도구")

    op_tabs = st.tabs(
        ["🏷 카테고리 자동 태깅", "📈 답변 품질 추적", "🎯 벤치마크 설정", "📥 CSV 내보내기"]
    )

    # ── 카테고리 자동 태깅 ──
    with op_tabs[0]:
        st.subheader("🏷 카테고리 자동 태깅 (Gemini)")
        st.caption(
            "Gemini를 사용해서 리뷰의 정확한 카테고리를 분류합니다. "
            "키워드 기반은 자동으로 동작하지만, AI 기반은 더 정확합니다."
        )

        untagged = df[df["ai_categories"].apply(lambda x: not x or len(x) == 0)]
        st.info(f"AI 태깅 안 된 리뷰: {len(untagged)}개")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            batch_size = st.number_input(
                "한 번에 처리할 개수",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
            )
        with col_t2:
            st.write("")
            if st.button(
                f"🤖 카테고리 자동 태깅 실행",
                use_container_width=True,
                disabled=len(untagged) == 0,
            ):
                target = untagged.head(batch_size)
                progress = st.progress(0)
                success_count = 0
                fail_count = 0
                for i, (_, row) in enumerate(target.iterrows()):
                    try:
                        prompt = build_category_tagging_prompt(row)
                        result = call_gemini(prompt)
                        # JSON 파싱
                        result = result.strip()
                        if result.startswith("```"):
                            result = re.sub(r"```(?:json)?\n?", "", result)
                            result = result.rstrip("```").strip()
                        cats = json.loads(result)
                        if isinstance(cats, list):
                            valid_cats = [
                                c for c in cats if c in HOTEL_CATEGORIES.keys()
                            ]
                            update_review(row["id"], {"ai_categories": valid_cats})
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception:
                        fail_count += 1
                    progress.progress((i + 1) / len(target))
                st.success(
                    f"✅ {success_count}개 태깅 완료 (실패 {fail_count}개)"
                )
                st.cache_data.clear()
                st.rerun()

    # ── 답변 품질 추적 ──
    with op_tabs[1]:
        st.subheader("📈 답변 품질 추적")

        completed = df[
            (df["status_norm"] == "처리완료")
            & (df["ai_reply"] != "")
            & (df["final_reply"] != "")
        ].copy()

        if len(completed) == 0:
            st.info("처리완료된 리뷰가 아직 없습니다. (final_reply 필드가 채워진 리뷰)")
        else:
            completed["edit_ratio"] = completed.apply(
                lambda r: diff_ratio(r["ai_reply"], r["final_reply"]), axis=1
            )

            qk1, qk2, qk3 = st.columns(3)
            qk1.metric("처리완료 리뷰", f"{len(completed)}개")
            qk2.metric(
                "평균 수정률",
                f"{completed['edit_ratio'].mean()*100:.1f}%",
                help="AI 원본 대비 사람이 수정한 비율. 낮을수록 AI가 잘 쓴 것.",
            )
            no_edit = len(completed[completed["edit_ratio"] < 0.05])
            qk3.metric(
                "거의 수정 안 한 답변",
                f"{no_edit}개",
                help="수정률 5% 미만",
            )

            st.markdown("### 수정률이 높았던 답변 (개선 포인트)")
            top_edited = completed.sort_values("edit_ratio", ascending=False).head(10)
            for _, r in top_edited.iterrows():
                with st.expander(
                    f"수정률 {r['edit_ratio']*100:.0f}% | {r['platform']} | {r.get('user', '')}"
                ):
                    st.markdown("**🤖 AI 원본**")
                    st.markdown(f"> {r['ai_reply']}")
                    st.markdown("**✍️ 최종 (수정본)**")
                    st.markdown(f"> {r['final_reply']}")

    # ── 벤치마크 설정 ──
    with op_tabs[2]:
        st.subheader("🎯 벤치마크 값 설정")
        st.caption("외부 데이터 (제주 평균, 경쟁사 평균 등) 을 입력해두면 대시보드에서 비교됩니다.")

        benchmark = get_benchmark()
        bm_col1, bm_col2 = st.columns(2)
        with bm_col1:
            jeju_avg = st.number_input(
                "제주 4-5성 평균 만족도 (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(benchmark.get("jeju_avg") or 85.0),
                step=0.1,
            )
        with bm_col2:
            competitor_avg = st.number_input(
                "주요 경쟁사 평균 만족도 (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(benchmark.get("competitor_avg") or 87.0),
                step=0.1,
            )
        note = st.text_area(
            "메모 (출처, 산정 방법 등)", value=benchmark.get("note", "")
        )

        if st.button("💾 저장", use_container_width=True):
            db.collection("config").document("benchmark").set(
                {
                    "jeju_avg": jeju_avg,
                    "competitor_avg": competitor_avg,
                    "note": note,
                    "updated_at": datetime.now().isoformat(),
                }
            )
            st.success("저장 완료!")
            st.cache_data.clear()

    # ── CSV 내보내기 ──
    with op_tabs[3]:
        st.subheader("📥 CSV / Excel 내보내기")
        st.caption("필터링된 결과를 파일로 다운로드합니다.")

        ex_platforms = st.multiselect(
            "플랫폼", sorted(df["platform"].unique()), default=sorted(df["platform"].unique())
        )
        ex_status = st.multiselect(
            "상태", ["대기중", "답변완료", "처리완료"], default=["대기중", "답변완료", "처리완료"]
        )

        ex_df = df[df["platform"].isin(ex_platforms) & df["status_norm"].isin(ex_status)]

        # 출력용 정리
        out_cols = [
            "platform", "date", "user", "country", "score", "score_pct",
            "title", "positive", "negative", "content",
            "categories_final", "status_norm", "ai_reply", "final_reply",
            "owner_reply", "room_type", "traveler_type", "booking_id",
        ]
        out_df = ex_df[[c for c in out_cols if c in ex_df.columns]].copy()
        out_df.columns = [
            "플랫폼", "리뷰일자", "작성자", "국적", "점수", "만족도(%)",
            "제목", "좋았던점", "아쉬운점", "본문",
            "카테고리", "상태", "AI답변", "최종답변",
            "기존호텔답변", "객실타입", "투숙유형", "예약번호",
        ][:len(out_df.columns)]

        st.dataframe(out_df.head(20), use_container_width=True)
        st.caption(f"미리보기 (상위 20개) / 전체 {len(out_df)}개")

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            csv = out_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                "📥 CSV 다운로드",
                csv,
                file_name=f"amber_reviews_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_ex2:
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    out_df.to_excel(writer, index=False, sheet_name="리뷰")
                st.download_button(
                    "📥 Excel 다운로드",
                    output.getvalue(),
                    file_name=f"amber_reviews_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            except ImportError:
                st.caption("Excel 다운로드: `openpyxl` 패키지가 필요합니다.")
