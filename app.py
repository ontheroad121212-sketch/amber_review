"""
앰버 7대 플랫폼 통합 AI 지배인 (Streamlit) v2
주요 개선:
- Gemini 2.5 Flash 로 모델 교체 (1.5/2.0 종료 대응)
- API 키 Streamlit Secrets 로 이동 (보안)
- 답변완료/대기중/처리완료 상태 통합 처리
- 부킹닷컴 좋았던 점/아쉬운 점 분리 표시
- 검색/필터/정렬 옵션
- 에러 메시지 사용자에게 표시 (except: pass 제거)
- 일괄 AI 답변 생성 (체크박스 선택)
- 부정 리뷰 알림 강조
"""

import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
import json
import traceback
from datetime import datetime

# ─────────────────────────────────────────────────────────
# 1. 초기 설정
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="앰버 7대 플랫폼 통합 관리", layout="wide", page_icon="🏨")

# Firebase 초기화
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

# Gemini API 키 — 반드시 Streamlit Secrets 에 등록해서 사용
# ⚠️ 보안: 키를 코드에 직접 적으면 GitHub 등에서 노출되어 자동 차단됩니다.
#   Streamlit Cloud → 본인 앱 → Settings → Secrets 에서 다음과 같이 등록:
#       GOOGLE_API_KEY = "AIzaSy..."
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        st.error(
            "⚠️ Gemini API 키가 등록되지 않았습니다.\n\n"
            "Streamlit Cloud → 앱 Settings → Secrets 에서 "
            "`GOOGLE_API_KEY = \"새_키_값\"` 형식으로 등록해 주세요. "
            "키는 https://aistudio.google.com/apikey 에서 발급받을 수 있어요."
        )
        st.stop()
except Exception as e:
    st.error(f"Gemini API 키 설정 실패: {e}")
    st.stop()

# 사용할 Gemini 모델 (2026년 5월 기준 권장)
# - gemini-2.5-flash: 가성비 + 한국어 잘 함
# - gemini-2.5-flash-lite: 더 빠르고 저렴 (fallback)
PRIMARY_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = ["gemini-2.5-flash-lite", "gemini-flash-latest"]

# 관리자 페이지 주소
ADMIN_URLS = {
    "네이버(Naver)": "https://new.smartplace.naver.com/bizes/place/6736761/reviews?menu=visitor",
    "아고다(Agoda)": "https://ycs.agoda.com/ko-kr/Reviews/Index",
    "부킹닷컴(Booking.com)": "https://admin.booking.com/hotel/hoteladmin/extranet_ng/manage/reviews.html",
    "익스피디아(Expedia)": "https://www.expediapartnercentral.com/",
    "야놀자(Yanolja)": "https://partner.yanolja.com/",
    "여기어때(Yeogieotte)": "https://partner.goodchoice.kr/",
    "트립닷컴(Trip)": "https://ebooking.trip.com/pro-web/review",
    "마이리얼트립(MyRealTrip)": "https://partner.myrealtrip.com/reviews/accommodation",
    "트립어드바이저(TripAdvisor)": "https://www.tripadvisor.com/Owners",
    "구글(Google)": "https://business.google.com/reviews",
}


# ─────────────────────────────────────────────────────────
# 2. 데이터 함수
# ─────────────────────────────────────────────────────────
@st.cache_data(ttl=60)  # 1분 캐시 (너무 자주 Firestore 호출 방지)
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
            d.setdefault("date", "날짜 정보 없음")
            d.setdefault("ai_reply", "")
            d.setdefault("title", "")
            d.setdefault("positive", "")
            d.setdefault("negative", "")
            d.setdefault("score", "")
            d.setdefault("user", "")
            d.setdefault("has_reply", False)
            # 플랫폼별 추가 필드
            d.setdefault("satisfaction_tags", "")  # 마이리얼트립
            d.setdefault("post_time", "")           # 구글 (상대시각)
            d.setdefault("post_date", "")           # 트립어드바이저, 익스피디아
            d.setdefault("travel_date", "")         # 마이리얼트립
            d.setdefault("stay_period", "")         # 아고다, 익스피디아
            d.setdefault("country", "")             # 부킹, 아고다
            d.setdefault("room_type", "")           # 아고다, 마이리얼트립
            d.setdefault("traveler_type", "")       # 아고다
            d.setdefault("booking_id", "")
            d.setdefault("review_id", "")
            d.setdefault("owner_reply", "")
            data.append(d)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"데이터 로딩 실패: {e}")
        return pd.DataFrame()


def normalize_status(row):
    """status 값을 3가지로 통일: 대기중 / 답변완료(플랫폼에서) / 처리완료(우리가 보냄)"""
    s = row.get("status", "대기중")
    if s in ("처리완료",):
        return "처리완료"
    if s in ("답변완료",) or row.get("has_reply"):
        return "답변완료"
    return "대기중"


def call_gemini(prompt, model_name=None):
    """Gemini 호출. 1차 모델 실패시 fallback. 모든 에러는 호출자에게 던진다."""
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
    """리뷰 row 에서 AI 가 답변할 통합 본문 만들기"""
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


def build_reply_prompt(row):
    """AI 답변 생성용 프롬프트"""
    review_text = build_review_text(row)
    score = row.get("score", "")
    user = row.get("user", "")
    platform = row.get("platform", "")

    # 점수에 따른 답변 방향 (5점/10점 만점 자동 환산)
    score_val = None
    try:
        score_val = float(score)
    except:
        pass

    # 만점 자동 판단
    max_score = 5 if platform in ("구글(Google)", "트립어드바이저(TripAdvisor)") else 10
    score_pct = (score_val / max_score) * 100 if score_val is not None else None

    score_guide = ""
    score_display = ""
    if score_val is not None:
        score_display = f"{score}/{max_score}점"
        if score_pct <= 50:
            score_guide = "점수가 매우 낮은 리뷰입니다. 변명하지 말고 진심으로 미안한 마음을 담아 응답하세요. 단, 과장된 사죄 표현(예: '죄송한 마음 금할 길이 없습니다')은 절대 쓰지 마세요. 구체적으로 어떤 부분을 어떻게 살펴보겠다는 한 마디를 포함해주세요."
        elif score_pct < 80:
            score_guide = "점수가 중간 정도입니다. 좋았던 점은 진심으로 감사하고, 아쉬운 점은 담백하게 받아들이세요. 변명 없이."
        else:
            score_guide = "점수가 높은 리뷰입니다. 과하지 않게 감사의 마음을 표현하세요. 영업멘트로 흐르지 않도록 주의."

    return f"""당신은 제주 '엠버퓨어힐 호텔앤리조트'의 지배인입니다.
{platform} 에 올라온 아래 리뷰에 대한 답변을 작성해 주세요.

[톤]
- '정중하지만 거리감 없는' 느낌. 단골에게 답하듯 따뜻하게.
- AI가 쓴 듯한 격식 차린 클리셰는 절대 금지.
- 마치 사람이 한 명씩 직접 손으로 쓴 듯 자연스럽게.

[금지 표현 — 절대 사용 금지]
다음과 같은 진부한 호텔 답변 클리셰는 **한 글자도 쓰지 마세요**:
- "죄송한 마음 금할 길이 없습니다", "사죄의 말씀을 올립니다"
- "고객님께 막중한 책임감을 느낍니다"
- "최고의 서비스로 보답하겠습니다"
- "더욱 노력하는 호텔이 되도록 하겠습니다"
- "심심한 사과의 말씀", "깊이 반성하고 있습니다"
- "다시 한번 진심으로 사과드립니다"
- "고객님의 소중한 의견" (너무 흔함)
- "성원에 보답하기 위해", "기대에 부응할 수 있도록"
- "불편을 끼쳐드려 대단히 죄송합니다" 같은 영혼 없는 정형구

[작성 지침]
- 첫 문장: "안녕하세요, 엠버퓨어힐입니다." 정도로 짧고 자연스럽게.
- 작성자 이름이 있으면 자연스럽게 한 번 호명. (예: "Jeongho 님, 방문해 주셔서 감사합니다.")
- 리뷰에서 **구체적으로 언급된 부분** (조식, 뷔페, 풀, 객실, 직원 등) 을 답변에도 한두 가지 짚어주세요. 두루뭉술 금지.
- 좋았던 점 → 무엇이 어떻게 좋았다는지 받아 적고 감사.
- 아쉬운 점 → 변명 없이 인정. 가능하면 어떤 부분을 어떻게 살펴보겠다는 식의 구체적인 한 마디.
- 마지막은 재방문 환영. 단, "꼭 다시 모시고 싶습니다" 류는 OK, "성원에 보답..." 류는 금지.
- 분량: **200~350자** 사이.
- 이모티콘: 0~1개. 본문 끝 또는 자연스러운 위치에 1개 정도만. (예: 🌿 😊 🙏) 줄줄이 안 됨.
- 영업/마케팅 표현 금지 ("저희 호텔은 ~를 자랑합니다" 같은 거).

[점수 가이드]
{score_guide}

[답변할 리뷰]
플랫폼: {platform}
작성자: {user or "(이름 없음)"}
점수: {score_display or "(없음)"}
{review_text}

위 리뷰에 대한 호텔 답변만 작성해 주세요. 따옴표나 서두 설명 없이 답변 본문만 출력합니다.
"""


def update_review(doc_id, fields):
    """Firestore 문서 업데이트"""
    db.collection("reviews").document(doc_id).update(fields)


# ─────────────────────────────────────────────────────────
# 3. 메인 화면
# ─────────────────────────────────────────────────────────
st.title("🏨 앰버 통합 리뷰 AI 지배인")

df = get_reviews()

if df.empty:
    st.info("아직 수집된 리뷰가 없습니다. 크롤러를 먼저 실행해 주세요.")
    st.stop()

# 상태 정규화
df["status_norm"] = df.apply(normalize_status, axis=1)

# ─── 사이드바: 필터 + 통계 + 도구 ───
with st.sidebar:
    st.header("🔍 필터")

    # 플랫폼 필터
    platforms_all = sorted(df["platform"].unique())
    platforms_sel = st.multiselect("플랫폼", platforms_all, default=platforms_all)

    # 상태 필터
    status_sel = st.multiselect(
        "처리 상태",
        ["대기중", "답변완료", "처리완료"],
        default=["대기중"],
        help="대기중: 아직 답변 안 한 것 / 답변완료: 플랫폼에서 답변 완료 / 처리완료: 우리가 AI로 답변 후 확정한 것",
    )

    # 점수 필터 (구글/트립=5점만점, 부킹/아고다/익스=10점만점 — 자동 환산)
    score_filter = st.select_slider(
        "점수 범위",
        options=["전체", "낮은 점수만 (≤5)", "중간 (5~8)", "높은 점수 (≥8)"],
        value="전체",
        help="구글/트립어드바이저는 5점 만점, 나머지는 10점 만점. 자동 환산하여 비교합니다.",
    )

    # 검색
    search_keyword = st.text_input("🔎 본문 검색", placeholder="예: 조식, 청결, 직원...")

    # 정렬
    sort_order = st.radio(
        "정렬",
        ["최신 수집순", "오래된 수집순", "낮은 점수 먼저", "높은 점수 먼저"],
        index=0,
    )

    st.markdown("---")
    st.header("📊 통계")

    total = len(df)
    waiting = len(df[df["status_norm"] == "대기중"])
    replied = len(df[df["status_norm"] == "답변완료"])
    done = len(df[df["status_norm"] == "처리완료"])
    st.metric("전체 리뷰", f"{total}개")
    st.metric("⏳ 답변 대기", f"{waiting}개", delta=None)
    st.metric("✅ 플랫폼에서 답변완료", f"{replied}개")
    st.metric("🎯 우리가 처리완료", f"{done}개")

    st.markdown("---")
    if st.button("🔄 데이터 새로고침", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ─── 필터 적용 ───
filtered = df[df["platform"].isin(platforms_sel)]
filtered = filtered[filtered["status_norm"].isin(status_sel)]


def score_to_float(s):
    try:
        return float(s)
    except:
        return None


def score_to_pct(score_val, platform):
    """점수를 0-100 백분율로 정규화.
    - 구글, 트립어드바이저: 1~5점 → x*20 으로 0-100 환산
    - 부킹, 아고다, 익스피디아: 1~10점 → x*10 으로 0-100 환산
    - 마이리얼트립: 점수 없음 → None
    """
    if score_val is None or pd.isna(score_val):
        return None
    if platform in ("구글(Google)", "트립어드바이저(TripAdvisor)"):
        return score_val * 20  # 5점 만점 → 100점 환산
    else:
        return score_val * 10  # 10점 만점 → 100점 환산


filtered["score_num"] = filtered["score"].apply(score_to_float)
filtered["score_pct"] = filtered.apply(
    lambda r: score_to_pct(r["score_num"], r["platform"]), axis=1
)

# 점수 필터 - 백분율 기준 (0-100 통합)
# "낮음" = 50% 이하 (5점만점 2.5점 이하 / 10점만점 5점 이하)
# "중간" = 50-80% / "높음" = 80% 이상
if score_filter == "낮은 점수만 (≤5)":
    filtered = filtered[filtered["score_pct"].notna() & (filtered["score_pct"] <= 50)]
elif score_filter == "중간 (5~8)":
    filtered = filtered[
        filtered["score_pct"].notna()
        & (filtered["score_pct"] > 50)
        & (filtered["score_pct"] < 80)
    ]
elif score_filter == "높은 점수 (≥8)":
    filtered = filtered[filtered["score_pct"].notna() & (filtered["score_pct"] >= 80)]

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
# "최신 수집순" 은 기본 (Firestore 가 이미 DESCENDING timestamp)

# ─── 상단 요약 ───
c1, c2, c3, c4 = st.columns(4)
c1.metric("필터 결과", f"{len(filtered)}개")
c2.metric("⏳ 그 중 대기중", f"{len(filtered[filtered['status_norm'] == '대기중'])}개")

# 부정 리뷰 경고 (낮은 점수) - 백분율 50% 이하 (5점만점 2.5/10점만점 5점)
low_score_waiting = filtered[
    (filtered["status_norm"] == "대기중")
    & (filtered["score_pct"].notna())
    & (filtered["score_pct"] <= 50)
]
c3.metric("🚨 부정 리뷰 (대기)", f"{len(low_score_waiting)}개")

# 평균 점수 - 백분율로 표시 (다른 만점 체계 통합)
avg_pct = filtered["score_pct"].dropna().mean()
c4.metric(
    "평균 만족도",
    f"{avg_pct:.0f}%" if not pd.isna(avg_pct) else "—",
    help="플랫폼별 만점 다른 점수를 100% 만점으로 환산한 평균",
)

if len(low_score_waiting) > 0:
    st.warning(
        f"🚨 점수가 낮은 리뷰 {len(low_score_waiting)}개가 답변을 기다리고 있습니다. 우선 처리하시는 게 좋아요."
    )

st.markdown("---")

# ─── 종합 분석 + 일괄 답변 ───
ca, cb = st.columns(2)

with ca:
    if st.button("📊 모든 플랫폼 리뷰 종합 분석", use_container_width=True):
        with st.spinner("AI 비서가 리뷰를 종합 분석 중..."):
            # 분석 대상은 현재 필터링된 결과
            sample_df = filtered.head(80)  # 너무 많으면 토큰 폭발
            review_blocks = []
            for _, r in sample_df.iterrows():
                block = f"[{r['platform']}|{r['score']}점] {build_review_text(r)[:300]}"
                review_blocks.append(block)
            all_text = "\n\n".join(review_blocks)[:8000]

            prompt = f"""당신은 호텔 컨설팅 전문가입니다. 아래 리뷰들을 분석해 보고서를 작성하세요.

[보고서 구성]
1. **핵심 강점 Best 3** (구체적 표현 인용)
2. **개선 필요 사항 Worst 3** (구체적 사례 + 우선순위)
3. **플랫폼별 고객 성향 차이** (있는 경우만)
4. **이번 주 운영 전략 제안** (실행 가능한 3가지)

[리뷰 데이터 (총 {len(sample_df)}건)]
{all_text}

[보고서]"""

            try:
                report = call_gemini(prompt)
                st.success("✅ 종합 보고서 작성 완료!")
                st.info(report)
            except Exception as e:
                st.error(f"분석 실패: {e}")
                with st.expander("상세 오류"):
                    st.code(traceback.format_exc())

with cb:
    waiting_in_filter = filtered[filtered["status_norm"] == "대기중"]
    if st.button(
        f"🤖 대기 중 리뷰 일괄 AI 답변 ({len(waiting_in_filter)}개)",
        use_container_width=True,
        disabled=(len(waiting_in_filter) == 0),
    ):
        progress = st.progress(0)
        success_count = 0
        fail_count = 0
        for i, (_, row) in enumerate(waiting_in_filter.iterrows()):
            try:
                if row.get("ai_reply"):
                    # 이미 초안 있으면 스킵
                    continue
                prompt = build_reply_prompt(row)
                reply = call_gemini(prompt)
                update_review(row["id"], {"ai_reply": reply})
                success_count += 1
            except Exception as e:
                fail_count += 1
            progress.progress((i + 1) / len(waiting_in_filter))
        st.success(f"✅ {success_count}개 초안 작성 완료 (실패 {fail_count}개)")
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# ─── 플랫폼별 탭 ───
if len(filtered) == 0:
    st.info("필터 조건에 맞는 리뷰가 없습니다.")
    st.stop()

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

            # expander 헤더 — 점수 + 이름 + 날짜 + 미리보기
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

            # 미리보기: 제목 우선, 없으면 본문 첫 줄
            preview = (
                row.get("title")
                or row.get("negative")
                or row.get("positive")
                or row.get("content", "")
            )
            preview = preview.replace("\n", " ")[:50]
            header_parts.append(f"| {preview}...")

            is_waiting = status == "대기중"
            header = " ".join(header_parts)

            with st.expander(header, expanded=is_waiting):
                col1, col2 = st.columns([3, 1])

                with col1:
                    # 리뷰 본문 — 필드 구조화 표시
                    if row.get("title"):
                        st.markdown(f"### {row['title']}")

                    if row.get("positive") or row.get("negative"):
                        # 부킹닷컴식 좋았던 점/아쉬운 점 분리 표시
                        if row.get("positive"):
                            st.markdown(f"**😊 좋았던 점**")
                            st.markdown(f"> {row['positive']}")
                        if row.get("negative"):
                            st.markdown(f"**😞 아쉬운 점**")
                            st.markdown(f"> {row['negative']}")
                    else:
                        # 통합 본문 표시
                        st.markdown(f"**[원문]**")
                        st.markdown(f"> {row.get('content', '(내용 없음)')}")

                    # 마이리얼트립 만족도 태그 (있는 경우)
                    if row.get("satisfaction_tags"):
                        st.caption(f"⭐ {row['satisfaction_tags']}")

                    # 메타 정보
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

                    # 호텔이 이미 답변한 내용 (답변완료된 경우)
                    if row.get("owner_reply"):
                        with st.expander("📩 호텔 측 기존 답변 보기"):
                            st.markdown(f"> {row['owner_reply']}")

                    st.markdown("---")

                    # AI 답변 영역
                    if not row.get("ai_reply"):
                        if st.button("🤖 AI 답변 초안 만들기", key=f"btn_{unique_key}"):
                            with st.spinner("AI가 답변을 작성하는 중..."):
                                try:
                                    prompt = build_reply_prompt(row)
                                    reply = call_gemini(prompt)
                                    update_review(row["id"], {"ai_reply": reply})
                                    st.cache_data.clear()
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"AI 답변 생성 실패: {e}")
                                    with st.expander("상세 오류 보기"):
                                        st.code(traceback.format_exc())
                    else:
                        st.markdown("**✍️ AI 답변 초안 (수정 가능)**")
                        edited_reply = st.text_area(
                            "답변 수정",
                            value=row["ai_reply"],
                            height=180,
                            key=f"edit_{unique_key}",
                            label_visibility="collapsed",
                        )
                        st.code(edited_reply, language=None)  # 복사용

                        c_a, c_b, c_c = st.columns(3)
                        with c_a:
                            st.link_button("🌐 관리자 페이지 열기", admin_url)
                        with c_b:
                            if st.button("🔄 답변 재생성", key=f"regen_{unique_key}"):
                                with st.spinner("재생성 중..."):
                                    try:
                                        prompt = build_reply_prompt(row)
                                        reply = call_gemini(prompt)
                                        update_review(row["id"], {"ai_reply": reply})
                                        st.cache_data.clear()
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"재생성 실패: {e}")
                        with c_c:
                            if st.button("✅ 확정 및 완료", key=f"confirm_{unique_key}"):
                                update_review(
                                    row["id"],
                                    {"ai_reply": edited_reply, "status": "처리완료"},
                                )
                                st.cache_data.clear()
                                st.rerun()

                with col2:
                    st.write(f"**상태**: {status}")
                    if row.get("score"):
                        score_val = score_to_float(row["score"])
                        if score_val is not None:
                            score_pct = score_to_pct(score_val, row["platform"])
                            # 만점 표기 (구글/트립=5, 나머지=10)
                            max_score = 5 if row["platform"] in (
                                "구글(Google)", "트립어드바이저(TripAdvisor)"
                            ) else 10
                            score_label = f"{row['score']}/{max_score}"

                            if score_pct is not None:
                                if score_pct <= 50:
                                    st.error(f"⚠️ 점수 {score_label}")
                                elif score_pct < 80:
                                    st.warning(f"점수 {score_label}")
                                else:
                                    st.success(f"점수 {score_label}")
                    if row.get("booking_id"):
                        st.caption(f"예약번호\n`{row['booking_id']}`")
                    if row.get("review_id"):
                        st.caption(f"리뷰ID\n`{row['review_id']}`")

                    # 잘못 분류된 답변완료를 다시 대기로 되돌리기 (수동)
                    if status == "답변완료":
                        if st.button("↩️ 대기중으로 되돌리기", key=f"undo_{unique_key}", help="잘못 답변완료로 표시된 경우 클릭"):
                            update_review(row["id"], {"status": "대기중", "has_reply": False})
                            st.cache_data.clear()
                            st.rerun()
                    # 반대로 대기중을 답변완료로 (트립어드바이저처럼 수동 분류 필요할 때)
                    elif status == "대기중":
                        if st.button("✅ 답변완료로 표시", key=f"mark_replied_{unique_key}", help="이미 플랫폼에서 답변한 경우"):
                            update_review(row["id"], {"status": "답변완료", "has_reply": True})
                            st.cache_data.clear()
                            st.rerun()
