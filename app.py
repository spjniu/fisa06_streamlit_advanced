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
# 설정 / 스타일 (금융앱 느낌)
# -----------------------------
st.set_page_config(page_title="주가 조회 앱", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1200px;}
section[data-testid="stSidebar"] { width: 340px; }

.card {
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(49, 51, 63, 0.12);
  background: rgba(255, 255, 255, 0.92);
  box-shadow: 0 8px 24px rgba(0,0,0,0.04);
  margin-bottom: 0.8rem;
}
.small-muted { color: rgba(49, 51, 63, 0.65); font-size: 0.9rem; }
.badge {
  display: inline-block; padding: 4px 10px; border-radius: 999px;
  font-size: 0.78rem; border: 1px solid rgba(49, 51, 63, 0.18);
  background: rgba(49, 51, 63, 0.04);
  vertical-align: middle; margin-left: 6px;
}
div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 주가 조회 (KRX)")
st.caption("기간을 빠르게 바꾸고(1M/3M/6M/YTD/1Y/3Y/MAX), 차트에서 확대/축소까지 가능한 금융형 대시보드")


# -----------------------------
# 데이터 유틸
# -----------------------------
@st.cache_data(ttl=60 * 60 * 12)
def get_krx_company_list() -> pd.DataFrame:
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
    company_name = (company_name or "").strip()
    if company_name.isdigit() and len(company_name) == 6:
        return company_name

    company_df = get_krx_company_list()
    codes = company_df.loc[company_df["회사명"] == company_name, "종목코드"].values
    if len(codes) > 0:
        return codes[0]

    raise ValueError(f"'{company_name}'을 찾을 수 없습니다. 종목코드 6자리를 직접 입력해보세요.")


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 이동평균
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    # RSI(14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df["RSI14"] = 100 - (100 / (1 + rs))

    return df


def calc_start_date(preset: str, end_date: datetime.date) -> datetime.date:
    """
    빠른 기간 선택(preset)에 따라 시작일 계산
    """
    if preset == "1M":
        return end_date - datetime.timedelta(days=31)
    if preset == "3M":
        return end_date - datetime.timedelta(days=92)
    if preset == "6M":
        return end_date - datetime.timedelta(days=183)
    if preset == "YTD":
        return datetime.date(end_date.year, 1, 1)
    if preset == "1Y":
        return end_date - datetime.timedelta(days=365)
    if preset == "3Y":
        return end_date - datetime.timedelta(days=365 * 3)
    # MAX는 date_input에서 받는 값 그대로 쓰도록(여기선 end_date만 반환)
    return datetime.date(end_date.year, 1, 1)


def build_plotly_chart(
    df: pd.DataFrame,
    company_name: str,
    show_volume: bool,
    ma_opts: list[str],
    show_rsi: bool,
    show_range_slider: bool,
) -> go.Figure:
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

    # 캔들 (상승: 빨강 / 하락: 파랑)
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#D84A4A",
            decreasing_line_color="#2E6BE6",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # MA
    for ma in ma_opts:
        if ma in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[ma], mode="lines", name=ma),
                row=1,
                col=1,
                secondary_y=False,
            )

    # 거래량 (상승/하락 색 분리)
    if show_volume and "Volume" in df.columns:
        up = df["Close"] >= df["Open"]
        vol_colors = np.where(up, "rgba(216,74,74,0.35)", "rgba(46,107,230,0.35)")
        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors),
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

    # range selector (상단 버튼)
    # NOTE: Plotly 상단 내장 버튼(주/월/6M/YTD/1Y/ALL)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(step="year", stepmode="todate", label="YTD"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL"),
                ]
            )
        ),
        rangeslider=dict(visible=show_range_slider),
        type="date",
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=760 if show_rsi else 560,
        title=f"{company_name} 차트",
        legend_orientation="h",
        legend_y=-0.18,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")

    return fig


# -----------------------------
# 사이드바 (필터 패널)
# -----------------------------
today_dt = datetime.datetime.now()
today_date = today_dt.date()
jan_1 = datetime.date(today_dt.year, 1, 1)

st.sidebar.markdown("## 🔎 종목/기간")
company_name = st.sidebar.text_input(
    "회사명 또는 6자리 종목코드",
    placeholder="예) 삼성전자 / 005930",
)

# 빠른 기간 선택
preset = st.sidebar.radio(
    "빠른 기간",
    ["직접 선택", "1M", "3M", "6M", "YTD", "1Y", "3Y", "MAX"],
    horizontal=False,
)

# 기간 입력 (preset에 따라 기본값 자동 셋)
default_end = today_date
default_start = jan_1

if preset != "직접 선택" and preset != "MAX":
    default_start = calc_start_date(preset, default_end)

selected_dates = st.sidebar.date_input(
    "기간 선택",
    (default_start, default_end),
    format="YYYY-MM-DD",
)

st.sidebar.markdown("## 📊 차트 옵션")
show_volume = st.sidebar.checkbox("거래량", value=True)
show_range_slider = st.sidebar.checkbox("차트 하단 슬라이더(줌)", value=False)
ma_opts = st.sidebar.multiselect(
    "이동평균선",
    ["MA5", "MA20", "MA60", "MA120"],
    default=["MA20", "MA60"],
)
show_rsi = st.sidebar.checkbox("RSI(14)", value=True)

st.sidebar.markdown("---")
confirm_btn = st.sidebar.button("📌 조회하기", use_container_width=True)


# -----------------------------
# 메인 로직
# -----------------------------
if confirm_btn:
    if not company_name.strip():
        st.warning("조회할 회사 이름(또는 종목코드)을 입력하세요.")
        st.stop()

    if not isinstance(selected_dates, (tuple, list)) or len(selected_dates) != 2:
        st.warning("조회할 날짜를 시작/종료 2개로 선택해주세요.")
        st.stop()

    # preset == MAX인 경우: 가능한 오래 가져오도록 시작일을 넉넉히 (예: 2000-01-01)
    if preset == "MAX":
        start_dt = datetime.date(2000, 1, 1)
        end_dt = selected_dates[1]
    else:
        start_dt, end_dt = selected_dates

    start_date = start_dt.strftime("%Y%m%d")
    end_date = end_dt.strftime("%Y%m%d")

    try:
        with st.spinner("데이터를 수집하는 중..."):
            stock_code = get_stock_code_by_company(company_name)
            price_df = fdr.DataReader(stock_code, start_date, end_date)

        if price_df.empty:
            st.info("해당 기간의 주가 데이터가 없습니다.")
            st.stop()

        price_df = add_indicators(price_df)

        # KPI 카드
        last = price_df.iloc[-1]
        prev = price_df.iloc[-2] if len(price_df) >= 2 else last

        chg = float(last["Close"] - prev["Close"])
        pct = (chg / float(prev["Close"]) * 100) if float(prev["Close"]) != 0 else 0.0
        direction = "▲" if chg >= 0 else "▼"

        st.markdown('<div class="card">', unsafe_allow_html=True)
        top_left, top_right = st.columns([3, 1], vertical_alignment="center")

        with top_left:
            st.markdown(f"### {company_name} <span class='badge'>KRX</span>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='small-muted'>기간</span>  {start_dt} ~ {end_dt}",
                unsafe_allow_html=True,
            )

        with top_right:
            st.metric("종가", f"{last['Close']:,.0f}", f"{direction} {abs(chg):,.0f} ({pct:.2f}%)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("시가", f"{last['Open']:,.0f}")
        c2.metric("고가", f"{last['High']:,.0f}")
        c3.metric("저가", f"{last['Low']:,.0f}")
        c4.metric("거래량", f"{int(last['Volume']):,}" if "Volume" in last else "-")
        st.markdown("</div>", unsafe_allow_html=True)

        # 탭
        tab1, tab2, tab3, tab4 = st.tabs(["📊 차트", "🧾 데이터", "📈 수익률", "⬇️ 다운로드"])

        with tab1:
            fig = build_plotly_chart(
                price_df,
                company_name=company_name,
                show_volume=show_volume,
                ma_opts=ma_opts,
                show_rsi=show_rsi,
                show_range_slider=show_range_slider,
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.dataframe(price_df, use_container_width=True)

        with tab3:
            # 금융앱 느낌: 누적수익률(기준=100) + 일간수익률
            ret = price_df["Close"].pct_change()
            cum = (1 + ret.fillna(0)).cumprod() * 100

            r1, r2 = st.columns([2, 1])
            with r1:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=price_df.index, y=cum, mode="lines", name="누적수익률(기준=100)"))
                fig2.update_layout(
                    template="plotly_white",
                    hovermode="x unified",
                    title="누적수익률 (Base=100)",
                    height=380,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                fig2.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
                fig2.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
                st.plotly_chart(fig2, use_container_width=True)

            with r2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**기간 성과 요약**")
                st.write(f"- 시작 종가: {price_df.iloc[0]['Close']:,.0f}")
                st.write(f"- 종료 종가: {price_df.iloc[-1]['Close']:,.0f}")
                total_ret = (price_df.iloc[-1]["Close"] / price_df.iloc[0]["Close"] - 1) * 100
                st.write(f"- 총 수익률: {total_ret:.2f}%")
                vol = ret.std() * np.sqrt(252) * 100 if ret.std() == ret.std() else 0.0
                st.write(f"- 변동성(연율): {vol:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)

            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=price_df.index, y=(ret * 100).fillna(0), name="일간 수익률(%)"))
            fig3.update_layout(
                template="plotly_white",
                hovermode="x unified",
                title="일간 수익률(%)",
                height=280,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            fig3.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
            fig3.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)")
            st.plotly_chart(fig3, use_container_width=True)

        with tab4:
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                price_df.to_excel(writer, index=True, sheet_name="Sheet1")
            st.download_button(
                label="📥 엑셀 파일 다운로드",
                data=output.getvalue(),
                file_name=f"{company_name}_주가.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
else:
    st.info("사이드바에서 회사명/종목코드와 기간을 선택한 뒤 '조회하기'를 눌러주세요.")
