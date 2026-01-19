# 표준 라이브러리
import datetime
from io import BytesIO

# 서드파티 라이브러리
import streamlit as st
import pandas as pd
import FinanceDataReader as fdr

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -----------------------------
# 데이터 유틸
# -----------------------------
@st.cache_data(ttl=60 * 60 * 12)  # 12시간 캐시 (상장사 목록은 자주 안 바뀜)
def get_krx_company_list() -> pd.DataFrame:
    """
    KRX 상장법인 목록(회사명, 종목코드) 로드
    """
    try:
        url = "http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
        df_listing = pd.read_html(url, header=0, flavor="bs4", encoding="EUC-KR")[0]

        df_listing = df_listing[["회사명", "종목코드"]].copy()
        df_listing["종목코드"] = df_listing["종목코드"].apply(lambda x: f"{x:06}")
        return df_listing

    except Exception as e:
        st.error(f"상장사 명단을 불러오는 데 실패했습니다: {e}")
        return pd.DataFrame(columns=["회사명", "종목코드"])


def get_stock_code_by_company(company_name: str) -> str:
    """
    회사명 입력 시 종목코드 반환.
    6자리 숫자면 그대로 종목코드로 처리.
    """
    company_name = (company_name or "").strip()

    if company_name.isdigit() and len(company_name) == 6:
        return company_name

    company_df = get_krx_company_list()
    codes = company_df.loc[company_df["회사명"] == company_name, "종목코드"].values
    if len(codes) > 0:
        return codes[0]

    raise ValueError(f"'{company_name}'을 찾을 수 없습니다. 종목코드 6자리를 직접 입력해보세요.")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    MA + RSI(14) 추가
    """
    df = df.copy()

    # 이동평균
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    # RSI(14): 단순 rolling mean 기반
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    return df


def build_plotly_chart(
    df: pd.DataFrame,
    company_name: str,
    show_volume: bool,
    ma_opts: list[str],
    show_rsi: bool,
) -> go.Figure:
    """
    Plotly 캔들 + MA + 거래량 + RSI
    """
    rows = 2 if show_rsi else 1
    row_heights = [0.7, 0.3] if show_rsi else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=row_heights,
        specs=[[{"secondary_y": True}]] + ([[{"secondary_y": False}]] if show_rsi else []),
    )

    # 캔들
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # MA 오버레이
    for ma in ma_opts:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[ma], mode="lines", name=ma),
                row=1,
                col=1,
                secondary_y=False,
            )

    # 거래량(보조축)
    if show_volume and "Volume" in df.columns:
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.35),
            row=1,
            col=1,
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Price", row=1, col=1, secondary_y=False)

    # RSI
    if show_rsi and "RSI14" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI(14)"),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1, title_text="RSI")

    fig.update_layout(
        height=700 if show_rsi else 520,
        title=f"{company_name} 차트",
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        legend_y=-0.15,
        margin=dict(l=10, r=10, t=60, b=10),
    )

    return fig


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="주가 조회 앱", layout="wide")
st.title("📈 주가 조회 앱")

company_name = st.sidebar.text_input("조회할 회사를 입력하세요 (회사명 또는 6자리 종목코드)")

today_dt = datetime.datetime.now()
jan_1 = datetime.date(today_dt.year, 1, 1)
today_date = today_dt.date()

selected_dates = st.sidebar.date_input(
    "조회할 날짜를 입력하세요",
    (jan_1, today_date),
    format="MM.DD.YYYY",
)

# 그래프 옵션
st.sidebar.markdown("---")
show_candle = st.sidebar.checkbox("캔들차트 보기(Plotly)", value=True)
show_volume = st.sidebar.checkbox("거래량 표시", value=True)
ma_opts = st.sidebar.multiselect(
    "이동평균선(MA) 선택",
    ["MA5", "MA20", "MA60", "MA120"],
    default=["MA20", "MA60"],
)
show_rsi = st.sidebar.checkbox("RSI(14) 표시", value=True)

confirm_btn = st.sidebar.button("조회하기")

# -----------------------------
# 메인 로직
# -----------------------------
if confirm_btn:
    if not company_name.strip():
        st.warning("조회할 회사 이름(또는 종목코드)을 입력하세요.")
    else:
        try:
            if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
                st.warning("조회할 날짜를 시작/종료 2개로 선택해주세요.")
                st.stop()

            start_date = selected_dates[0].strftime("%Y%m%d")
            end_date = selected_dates[1].strftime("%Y%m%d")

            with st.spinner("데이터를 수집하는 중..."):
                stock_code = get_stock_code_by_company(company_name)
                price_df = fdr.DataReader(stock_code, start_date, end_date)

            if price_df.empty:
                st.info("해당 기간의 주가 데이터가 없습니다.")
            else:
                # 지표 추가
                price_df = add_indicators(price_df)

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader(f"[{company_name}] 최근 데이터")
                    st.dataframe(price_df.tail(20), width="stretch")

                with col2:
                    st.subheader("📊 차트")
                    if show_candle:
                        fig = build_plotly_chart(
                            price_df,
                            company_name=company_name,
                            show_volume=show_volume,
                            ma_opts=ma_opts,
                            show_rsi=show_rsi,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Plotly 차트를 끄셨습니다. (원하면 Matplotlib 대체 차트를 추가할 수 있어요.)")

                # 엑셀 다운로드
                st.markdown("---")
                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    price_df.to_excel(writer, index=True, sheet_name="Sheet1")

                st.download_button(
                    label="📥 엑셀 파일 다운로드",
                    data=output.getvalue(),
                    file_name=f"{company_name}_주가.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
