import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
import time

# --- 1. 초기 설정 ---
st.set_page_config(page_title="앰버 통합 리뷰 관리자", layout="wide")

if not firebase_admin._apps:
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Firebase 키 파일 에러: {e}")

db = firestore.client()

# AI 설정
GOOGLE_API_KEY = "AIzaSyAE1IUnXiR4Oxrkgr2LCxCC4RCwqrZwCBE"
genai.configure(api_key=GOOGLE_API_KEY)

# 관리자 페이지 주소
ADMIN_URLS = {
    "네이버(Naver)": "https://new.smartplace.naver.com/bizes/place/6736761/reviews?menu=visitor",
    "아고다(Agoda)": "https://ycs.agoda.com/ko-kr/Reviews/Index",
    "부킹닷컴(Booking)": "https://admin.booking.com/hotel/hoteladmin/extranet_ng/manage/reviews.html",
    "익스피디아(Expedia)": "https://www.expediapartnercentral.com/",
    "야놀자(Yanolja)": "https://partner.yanolja.com/",
    "여기어때(Yeogieotte)": "https://partner.goodchoice.kr/",
    "트립닷컴(Trip)": "https://ebooking.trip.com/pro-web/review"
}

# --- 2. 데이터 불러오기 함수 ---
def get_reviews():
    try:
        docs = db.collection("reviews").order_by("timestamp", direction=firestore.Query.DESCENDING).stream()
        data = []
        for doc in docs:
            d = doc.to_dict()
            d['id'] = doc.id
            if 'platform' not in d: d['platform'] = '기타'
            if 'content' not in d: d['content'] = '(내용 없음)'
            if 'status' not in d: d['status'] = '대기중'
            if 'date' not in d: d['date'] = '날짜 정보 없음'
            data.append(d)
        return pd.DataFrame(data)
    except: return pd.DataFrame()

# --- 3. 대시보드 메인 화면 ---
st.title("🏨 앰버 7대 플랫폼 통합 AI 지배인")

df = get_reviews()

if not df.empty:
    # 상단 요약 바
    c_a, c_b, c_c = st.columns(3)
    c_a.metric("전체 리뷰", f"{len(df)}개")
    c_b.metric("대기 중", f"{len(df[df['status'] == '대기중'])}개")
    c_c.metric("처리 완료", f"{len(df[df['status'] == '처리완료'])}개")

    st.markdown("---")

    # 🔥 [부활] 전체 리뷰 종합 분석 기능
    if st.button("📊 모든 플랫폼 리뷰 종합 분석 (인사이트 보고서 생성)"):
        with st.spinner("비서가 모든 플랫폼의 리뷰를 종합 분석 중입니다..."):
            all_text = " ".join(df['content'].tolist())[:4000] # 분석 글자수 제한
            
            try:
                # 분석은 가장 성능 좋은 모델 위주로 시도
                m = genai.GenerativeModel('gemini-1.5-flash')
                report_prompt = f"""
                당신은 호텔 컨설팅 전문가입니다. 아래 7개 플랫폼 리뷰를 분석하여 보고서를 작성하세요.
                1. 우리 호텔의 핵심 강점 (Best 3)
                2. 당장 개선이 필요한 불만 사항 (Worst 3)
                3. 플랫폼별 고객 성향 차이 요약
                4. 지배인을 위한 이번 주 운영 전략 제안
                
                리뷰 데이터: {all_text}
                """
                report_res = m.generate_content(report_prompt)
                st.success("✅ 종합 보고서 작성이 완료되었습니다!")
                st.info(report_res.text)
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")

    st.markdown("---")

    # 플랫폼별 탭 구성
    platforms = sorted(df['platform'].unique())
    tabs = st.tabs(platforms)

    for i, tab in enumerate(tabs):
        with tab:
            p_name = platforms[i]
            p_df = df[df['platform'] == p_name]
            admin_url = ADMIN_URLS.get(p_name, "#")

            for index, row in p_df.iterrows():
                unique_key = f"{row['platform']}_{row['id']}"
                is_waiting = (row['status'] == '대기중')
                
                with st.expander(f"📌 [{row['status']}] {row['date']} | {row['content'][:40]}...", expanded=is_waiting):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**[원문]**\n{row['content']}")
                        
                        if not row.get('ai_reply'):
                            if st.button("🤖 AI 답변 초안 만들기", key=f"btn_{unique_key}"):
                                with st.spinner("작성 중..."):
                                    # 모델 2중 방어
                                    for m_name in ['gemini-1.5-flash', 'gemini-2.0-flash-exp']:
                                        try:
                                            model = genai.GenerativeModel(m_name)
                                            res = model.generate_content(f"호텔 지배인으로서 {row['platform']} 리뷰에 답글 쓰기: {row['content']}")
                                            db.collection("reviews").document(row['id']).update({"ai_reply": res.text})
                                            st.rerun()
                                            break
                                        except: continue
                        else:
                            st.markdown("**✍️ 지배인님, 내용을 수정하신 후 아래 버튼을 이용하세요:**")
                            edited_reply = st.text_area("답변 수정", value=row['ai_reply'], height=150, key=f"edit_{unique_key}")
                            st.code(edited_reply, language=None) # 복사 기능 제공
                            
                            c_copy, c_link = st.columns(2)
                            with c_copy:
                                st.link_button("🌐 관리자 페이지 열기", admin_url)
                            with c_link:
                                if st.button("✅ 확정 및 완료", key=f"confirm_{unique_key}"):
                                    db.collection("reviews").document(row['id']).update({
                                        "ai_reply": edited_reply, "status": "처리완료"
                                    })
                                    st.rerun()

                    with col2:
                        st.write(f"상태: **{row['status']}**")
else:
    st.info("데이터가 없습니다.")
