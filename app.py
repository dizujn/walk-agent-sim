"""
규칙 기반 보행 에이전트 시뮬레이션 (Streamlit)

실행:  pip install -r requirements.txt
       streamlit run app.py

데이터 (prepare_data.py 로 만든 것)
  data/walk_links.parquet (또는 data/walk_links/*.parquet)  보행 가능 링크 + 행정동 코드
  data/population.csv                                      분기·행정동·시간대 유동인구

규칙
- 유동인구(분기 합계) ÷ 분기 일수 = 하루 평균, 이를 시간대 길이(시간)로 나눠 1시간당 인원 = 에이전트 수
- 시간 비례: 실제 1초 = 시뮬레이션 1분 (1시간 = 60초)
- 보행속도도 같은 비례로 변환 (기본 1.2 m/s → 화면상 72 m/s)
- 교차로(노드)에서는 이어진 길 중 하나를 동일 확률로 선택
"""
import calendar
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="보행 에이전트 시뮬레이션", layout="wide")

TIME_SCALE = 60  # 실제 1초 = 시뮬레이션 60초(1분)


# ---------------- 데이터 로드 ----------------
DATA = Path(__file__).parent / "data"
COORD_SCALE = 1_000_000


@st.cache_data(show_spinner="보행 링크 읽는 중…")
def load_links():
    import pyarrow.parquet as pq
    single, folder = DATA / "walk_links.parquet", DATA / "walk_links"
    if single.exists():
        df = pq.read_table(single).to_pandas()
    else:
        df = pd.concat([pq.read_table(p).to_pandas() for p in sorted(folder.glob("*.parquet"))], ignore_index=True)
    return df


@st.cache_data(show_spinner="유동인구 읽는 중…")
def load_population():
    df = pd.read_csv(DATA / "population.csv", encoding="utf-8-sig")
    if "행정동_전체명" not in df:
        df["행정동_전체명"] = df["행정동_코드_명"]
    return df


def data_ready():
    return (DATA / "population.csv").exists() and (
        (DATA / "walk_links.parquet").exists() or any((DATA / "walk_links").glob("*.parquet")))


def time_bands(pop):
    bands = []
    for c in pop.columns:
        m = re.fullmatch(r"시간대_(\d{1,2})_(\d{1,2})_유동인구_수", c)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            bands.append((c, a, b))
    return bands


def quarter_days(code):
    code = str(int(code))
    year, q = int(code[:4]), int(code[4])
    months = range(3 * q - 2, 3 * q + 1)
    return sum(calendar.monthrange(year, m)[1] for m in months)


# ---------------- 그래프 & 시뮬레이션 ----------------
@st.cache_data(show_spinner="보행 그래프 만드는 중…")
def build_graph(_links_df, adm_codes):   # _ 접두사: 큰 표는 캐시 키에서 제외
    sub = _links_df[_links_df["adm_cd"].isin(adm_codes)]
    adj = defaultdict(list)          # node -> [(edge_idx, other_node)]
    edges = []                       # (coords 시작→종료, length, 시작노드)
    node_xy = {}
    for lon, lat, s, e, L in zip(sub["lon"], sub["lat"], sub["start"], sub["end"], sub["length"]):
        coords = [(x / COORD_SCALE, y / COORD_SCALE) for x, y in zip(lon, lat)]
        s, e = int(s), int(e)
        if s == e or len(coords) < 2:
            continue
        L = float(L) if pd.notna(L) and L > 0 else 1.0
        i = len(edges)
        edges.append((coords, L, s))
        adj[s].append((i, e))
        adj[e].append((i, s))
        node_xy[s], node_xy[e] = coords[0], coords[-1]
    return dict(adj), edges, node_xy


def segment_lengths(coords, total):
    """링크 길이(m)를 링크 내부 구간들에 좌표 거리 비율로 배분"""
    a = np.asarray(coords)
    d = np.hypot(np.diff(a[:, 0]) * np.cos(np.radians(a[0, 1])), np.diff(a[:, 1]))
    s = d.sum()
    return d / s * total if s > 0 else np.full(len(d), total / len(d))


def simulate(adj, edges, node_xy, n_agents, speed_mps, duration_s, allow_uturn, seed):
    """각 에이전트의 경로를 [lon, lat, t(실제 초)] 웨이포인트로 반환"""
    rng = random.Random(seed)
    nodes = list(adj.keys())
    speed_screen = speed_mps * TIME_SCALE  # 실제 1초당 이동 거리(m)
    seg_cache = {}
    trips = []
    for _ in range(n_agents):
        node = rng.choice(nodes)
        prev_edge = None
        t = 0.0
        x, y = node_xy[node]
        path = [[x, y, 0.0]]
        while t < duration_s:
            options = adj[node]
            if not allow_uturn and len(options) > 1:
                options = [o for o in options if o[0] != prev_edge]
            ei, nxt = rng.choice(options)          # 동일 확률 선택
            coords, L, start = edges[ei]
            if ei not in seg_cache:
                seg_cache[ei] = segment_lengths(coords, L)
            segs = seg_cache[ei]
            forward = node == start          # 시작노드에서 출발하면 정방향
            pts = coords if forward else coords[::-1]
            sl = segs if forward else segs[::-1]
            for (px, py), d in zip(pts[1:], sl):
                t += d / speed_screen
                path.append([px, py, round(t, 3)])
            node, prev_edge = nxt, ei
        trips.append(path)
    return trips


# ---------------- 화면 ----------------
def render(edges, trips, duration_s, center, dot_label):
    lines = [[list(p) for p in e[0]] for e in edges]
    html = (Path(__file__).parent / "viewer.html").read_text(encoding="utf-8")
    html = (html.replace("__LINKS__", json.dumps(lines, separators=(",", ":")))
                .replace("__TRIPS__", json.dumps(trips, separators=(",", ":")))
                .replace("__DURATION__", str(duration_s))
                .replace("__SCALE__", str(TIME_SCALE))
                .replace("__CENTER__", json.dumps(center))
                .replace("__LABEL__", dot_label))
    if hasattr(st, "iframe"):          # Streamlit 최신 버전
        st.iframe(html, height=720)
    else:
        components.html(html, height=720)


st.title("규칙 기반 보행 에이전트 시뮬레이션")

if not data_ready():
    st.warning("walk_links.parquet, population.csv 파일이 없어요. prepare_data.py를 먼저 실행해 주세요.")
    st.stop()

links = load_links()
pop = load_population()
bands = time_bands(pop)
names = pop.drop_duplicates("행정동_코드").set_index("행정동_코드")["행정동_전체명"].to_dict()
has_links = set(links["adm_cd"].unique())

with st.sidebar:
    st.header("조건")
    q = st.selectbox("기준 분기", sorted(pop["기준_년분기_코드"].unique(), reverse=True))
    popq = pop[pop["기준_년분기_코드"] == q]
    codes = sorted(c for c in popq["행정동_코드"].unique() if c in has_links)
    adm = st.selectbox("행정동", codes, format_func=lambda c: names.get(c, str(c)))
    band = st.selectbox("시간대", bands, format_func=lambda b: f"{b[1]:02d}–{b[2]:02d}시")
    area = st.multiselect("보행 네트워크 범위 (행정동)", sorted(has_links, key=lambda c: names.get(c, "")),
                          default=[adm], format_func=lambda c: names.get(c, str(c)),
                          help="에이전트가 걸어 다닐 범위예요. 이웃 행정동을 더하면 경계에서 길이 끊기는 문제가 줄어요.")

    st.header("시뮬레이션")
    speed = st.number_input("평균 보행속도 (m/s)", 0.5, 2.5, 1.2, 0.1)
    sim_min = st.slider("시뮬레이션 길이 (분, 실제 초와 같음)", 10, 180, 60, 10)
    per_dot = st.number_input("점 1개 = 몇 명", 1, 1000, 1)
    max_dots = st.number_input("최대 점 개수 (브라우저 부담 제한)", 100, 20000, 3000, 100)
    uturn = st.checkbox("교차로에서 왔던 길로 되돌아가기도 선택지에 포함", False)
    seed = st.number_input("랜덤 시드", 0, 9999, 42)
    run = st.button("에이전트 풀기", type="primary")

row = popq[popq["행정동_코드"] == adm].iloc[0]
col, a, b = band
hours = (b - a) if b > a else (24 - a + b)
days = quarter_days(q)
per_day = row[col] / days
per_hour = per_day / hours
n_people = int(round(per_hour))
n_dots = min(int(np.ceil(n_people / per_dot)), int(max_dots))

c1, c2, c3, c4 = st.columns(4)
c1.metric("분기 시간대 유동인구", f"{int(row[col]):,}")
c2.metric(f"하루 평균 (÷{days}일)", f"{per_day:,.0f}")
c3.metric(f"1시간당 (÷{hours}시간)", f"{per_hour:,.0f}")
c4.metric("화면의 점", f"{n_dots:,}", help=f"점 1개 = {per_dot}명" + (" · 최대 개수로 잘림" if n_dots == max_dots else ""))
st.caption(f"시간 비례: 실제 1초 = 1분 · 보행속도 {speed} m/s → 화면상 {speed*TIME_SCALE:.0f} m/s")

if not area:
    st.info("보행 네트워크 범위를 하나 이상 골라주세요.")
    st.stop()

if run or "trips" in st.session_state:
    key = (tuple(area), q, adm, band, speed, sim_min, per_dot, max_dots, uturn, seed)
    if run or st.session_state.get("key") != key:
        adj, edges, node_xy = build_graph(links, tuple(area))
        if not adj:
            st.error("선택한 범위에 보행 가능한 링크가 없어요.")
            st.stop()
        trips = simulate(adj, edges, node_xy, n_dots, speed, sim_min * 60 / TIME_SCALE, uturn, seed)
        xs = [p[0] for e in edges for p in e[0]]; ys = [p[1] for e in edges for p in e[0]]
        st.session_state.update(key=key, trips=trips, edges=edges,
                                center=[float(np.mean(xs)), float(np.mean(ys))])
    s = st.session_state
    st.caption(f"보행 링크 {len(s['edges']):,}개 위에서 점 {len(s['trips']):,}개 이동")
    render(s["edges"], s["trips"], sim_min * 60 / TIME_SCALE, s["center"], f"점 1개 = {per_dot}명")
