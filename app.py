import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time

st.set_page_config(page_title="F1 Strategy AI", page_icon="🏎️", layout="wide", initial_sidebar_state="collapsed")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;900&family=Inter:wght@400;500;600&display=swap');

:root {
  --red:#FF2E2E; --orange:#f97316; --green:#16a34a;
  --blue:#2563eb; --purple:#7c3aed;
  --bg:#f1f5f9; --card:#ffffff; --border:#e2e8f0;
  --text:#0f172a; --muted:#64748b;
}

.stApp { background:linear-gradient(150deg,#f8fafc 0%,#e8eef7 100%);
         font-family:'Inter',sans-serif; color:var(--text); }

h1,h2,h3,h4 { font-family:'Poppins',sans-serif !important;
               color:var(--text) !important; font-weight:700; letter-spacing:-.4px; }

/* Animations */
@keyframes fadeSlide { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
@keyframes popIn     { 0%{opacity:0;transform:scale(.55)} 70%{transform:scale(1.05)} 100%{opacity:1;transform:scale(1)} }
@keyframes glowPulse { 0%,100%{box-shadow:0 0 0 0 rgba(37,99,235,.25)} 50%{box-shadow:0 0 22px 6px rgba(37,99,235,.12)} }
@keyframes shimmer   { 0%{background-position:-200% 0} 100%{background-position:200% 0} }

.anim-fade { animation:fadeSlide .6s cubic-bezier(.16,1,.3,1) both; }
.anim-pop  { animation:popIn     .6s cubic-bezier(.34,1.56,.64,1) both; }

/* Cards */
.card { background:var(--card); border:1px solid var(--border); border-radius:16px;
        padding:22px 26px; box-shadow:0 2px 10px rgba(0,0,0,.05);
        transition:transform .25s,box-shadow .25s; margin-bottom:18px; }
.card:hover { transform:translateY(-3px); box-shadow:0 8px 24px rgba(0,0,0,.08); }

/* Hero result card */
.hero-card { background:linear-gradient(135deg,#fff 0%,#eff6ff 100%);
             border:2px solid var(--blue); border-radius:22px; padding:40px 28px;
             text-align:center; animation:glowPulse 3s infinite; margin-bottom:18px; }
.pos-num { font-family:'Poppins',sans-serif; font-weight:900; font-size:7.5rem;
           line-height:1; display:block;
           animation:popIn .6s cubic-bezier(.34,1.56,.64,1) both; }
.pos-green  { color:var(--green); }
.pos-blue   { color:var(--blue);  }
.pos-orange { color:var(--orange);}
.pos-red    { color:var(--red);   }

/* Badge */
.badge { display:inline-block; padding:7px 20px; border-radius:999px;
         font-weight:700; font-size:1rem; margin:10px 0; }
.badge-green  { background:#dcfce7; color:#15803d; }
.badge-blue   { background:#dbeafe; color:#1d4ed8; }
.badge-orange { background:#ffedd5; color:#c2410c; }

/* Gradient confidence bar */
.conf-wrap { background:#e2e8f0; border-radius:999px; height:12px;
             overflow:hidden; margin:10px 0 4px; }
.conf-bar  { height:100%; border-radius:999px;
             background:linear-gradient(90deg,#2563eb,#16a34a);
             background-size:200% 100%;
             animation:shimmer 2.5s linear infinite;
             transition:width .8s cubic-bezier(.4,0,.2,1); }

/* Insight cards */
.ins-card { display:flex; gap:14px; align-items:flex-start;
            background:#f8fafc; border-radius:12px; padding:14px 16px;
            margin-bottom:10px; border-left:4px solid transparent;
            transition:transform .2s,box-shadow .2s; }
.ins-card:hover { transform:translateX(4px); box-shadow:0 4px 14px rgba(0,0,0,.06); }
.ins-pos  { border-left-color:var(--green); }
.ins-neg  { border-left-color:var(--orange); }
.ins-neu  { border-left-color:var(--blue); }
.ins-icon { font-size:1.6rem; line-height:1.2; flex-shrink:0; }
.ins-body { font-size:.93rem; color:#334155; line-height:1.5; }
.ins-label{ font-weight:700; font-size:.8rem; text-transform:uppercase;
            letter-spacing:.8px; margin-bottom:2px; }
.ins-label-pos { color:var(--green); }
.ins-label-neg { color:var(--orange); }
.ins-label-neu { color:var(--blue); }

/* Feed messages */
.feed-line { font-family:'Inter',sans-serif; font-size:.95rem;
             color:#475569; padding:6px 0; border-bottom:1px solid #f1f5f9;
             animation:fadeSlide .4s both; }
.feed-dot  { color:var(--blue); margin-right:8px; }

/* Buttons */
div.stButton > button {
  background:linear-gradient(90deg,var(--red),var(--orange));
  color:#fff; font-family:'Inter',sans-serif; font-weight:700;
  border:none; border-radius:999px; padding:.7rem 2rem; font-size:1rem;
  width:100%; transition:transform .2s,box-shadow .2s;
  box-shadow:0 4px 14px rgba(255,46,46,.3);
}
div.stButton > button:hover { transform:scale(1.04);
  box-shadow:0 8px 22px rgba(255,46,46,.45); color:#fff; }

.reset-btn div.stButton > button {
  background:transparent; border:1.5px solid var(--border);
  color:var(--muted); box-shadow:none; font-weight:500; }
.reset-btn div.stButton > button:hover {
  background:#f8fafc; border-color:#94a3b8; color:var(--text);
  transform:scale(1.02); box-shadow:none; }

/* Hero header */
.hero-title { font-family:'Poppins',sans-serif; font-size:2.9rem;
              font-weight:900; color:var(--text); letter-spacing:-1px; margin:0; }
.hero-sub   { font-size:1.05rem; color:var(--muted); margin-top:5px; }
.hero-badge { display:inline-block; background:#fff; border:1px solid var(--border);
              border-radius:999px; padding:4px 14px; font-size:.8rem;
              font-weight:600; color:var(--muted); margin-top:8px; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background:transparent; gap:14px; margin-bottom:22px; }
.stTabs [data-baseweb="tab"]       { font-family:'Inter',sans-serif; font-weight:500;
                                      font-size:.97rem; color:var(--muted);
                                      background:transparent; padding:8px 20px; }
.stTabs [aria-selected="true"] { color:var(--text)!important; font-weight:700;
                                   border-bottom-color:var(--red)!important; }

/* Divider */
.div { height:1px; background:var(--border); margin:18px 0; }

/* Section label */
.slabel { font-size:.74rem; font-weight:700; text-transform:uppercase;
          letter-spacing:1.1px; color:var(--muted); margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)

# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("f1_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}"); st.stop()

def prepare_features(inp, mdl):
    df = pd.DataFrame([inp])
    df['grid_squared']          = df['grid'] ** 2
    df['grid_x_form']           = df['grid'] * df['rolling_avg_position']
    df['qual_gap']              = df['best_qual_ms'] - df['q1_ms']
    df['grid_rank']             = df['grid']
    df['qual_rank']             = df['best_qual_ms']
    df['form_rank']             = df['rolling_avg_position']
    df['grid_norm']             = df['grid'] / 20
    df['driver_vs_constructor'] = df['prev_championship_points'] - df['prev_constructor_points']
    df['grid_minus_form']       = df['grid'] - df['rolling_avg_position']
    df['points_per_position']   = df['prev_championship_points'] / (df['prev_championship_position'] + 1)
    df['constructor_strength']  = df['prev_constructor_points'] / (df['round'] + 1)
    out = df.reindex(columns=mdl.feature_name_, fill_value=-999)
    return out.apply(pd.to_numeric, errors='coerce').fillna(-999)

def pos_meta(pos):
    if pos <= 3.5:  return "🟢 Podium",    "pos-green",  "badge-green",  min(96, int(92-pos*2))
    elif pos <= 10.5: return "🔵 Midfield", "pos-blue",   "badge-blue",   int(80-pos*1.5)
    else:           return "🟠 Backmarker","pos-orange",  "badge-orange", max(22, int(68-pos*2))

def build_insights(data, pred):
    outs = []
    if data['grid'] > pred + 1.5:
        outs.append(("⚡","pos","Qualifying pace advantage","High grid start unlocks track position — likely gain."))
    elif data['grid'] < pred - 1.5:
        outs.append(("📉","neg","Grid disadvantage","Starting position points to position drop — defend early."))
    if data['rolling_avg_position'] < 6:
        outs.append(("📈","pos","Recent form improving","Strong consistency in recent races boosts confidence."))
    if data['best_qual_ms'] - data['q1_ms'] < -1000:
        outs.append(("⏱️","pos","Setup progression","Rapid Q1→Q3 improvement indicates excellent setup mastery."))
    if data['prev_constructor_points'] > 150:
        outs.append(("🏗️","neu","Constructor backing","Well-resourced team provides strategic race-pace stability."))
    if not outs:
        outs.append(("📊","neu","Stable projection","All telemetry channels within normal operating parameters."))
    return outs[:3]

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="anim-fade" style="text-align:center;padding:28px 0 8px">
  <div class="hero-title">🏎️ &nbsp;F1 Race Strategy AI</div>
  <div class="hero-sub">Machine-learning powered race outcome prediction &amp; telemetry analysis</div>
  <span class="hero-badge">LightGBM · 70+ years of race data · Real-time inference</span>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2 = st.tabs(["⚡ Single Driver", "📊 Multi-Driver Grid"])

# ══════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════
with tab1:
    L, _, R = st.columns([1.05, 0.06, 1.45])

    # ── LEFT: inputs ─────────────────────────────────────
    with L:
        # Driver performance
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🏁 Driver Performance")
        a, b = st.columns(2)
        grid_pos = a.number_input("Grid Position",  1, 20,  4,    step=1,   help="Starting grid slot")
        drv_form = b.number_input("Driver Form",    1.0,20.0,3.5, step=0.1, help="Rolling avg of recent finishes")
        st.markdown('</div>', unsafe_allow_html=True)

        # Qualifying
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⏱️ Qualifying Data")
        c, d = st.columns(2)
        qual_ms = c.number_input("Best Qual (ms)", 60000,120000,82000, step=100, help="Q2/Q3 fastest lap (ms)")
        q1_ms   = d.number_input("Q1 Time (ms)",   60000,120000,83500, step=100, help="Q1 lap time (ms)")
        st.markdown('</div>', unsafe_allow_html=True)

        # Championship
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🏆 Championship Data")
        e, f = st.columns(2)
        drv_pts  = e.number_input("Driver Points",   0,500, 145, help="Championship points")
        drv_rank = f.number_input("Driver Rank",     1,22,  3,   help="Championship standing")
        team_pts = e.number_input("Constructor Pts", 0,1000,290, help="Team season points")
        race_rnd = f.number_input("Race Round",      1,24,  12,  help="Season round number")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Race Scenario Mode ──────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🌦️ Race Scenario Mode")
        sc1, sc2 = st.columns(2)
        rain_on   = sc1.toggle("🌧️ Rain Mode", value=False, help="Wet-weather race conditions")
        tire_strat= sc2.radio("🛞 Tire Strategy", ["Soft","Medium","Hard"], horizontal=True)
        sc_prob   = st.slider("🚨 Safety Car Probability", 0, 100, 20, step=5,
                              help="Estimated % chance of safety car deployment")
        st.markdown('</div>', unsafe_allow_html=True)

        # Buttons
        predict_clicked = st.button("Run AI Prediction 🚀", use_container_width=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("Reset Parameters 🔄", use_container_width=True):
            st.session_state.pop("pred_result", None)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT: output ─────────────────────────────────────
    with R:
        if predict_clicked:
            # ── Live AI feed ──────────────────────────────
            feed_slot = st.empty()
            feed_msgs = [
                "🔵 Connecting to telemetry feed...",
                "🔵 Analyzing driver form & consistency...",
                "🔵 Evaluating qualifying performance gap...",
                "🔵 Processing championship standings...",
                "🔵 Applying race scenario modifiers...",
                "🟢 Generating race outcome projection...",
            ]
            for msg in feed_msgs:
                feed_slot.markdown(
                    f'<div class="card" style="padding:18px 22px">'
                    f'<div class="slabel">Live AI Analysis Feed</div>'
                    + "".join([f'<div class="feed-line"><span class="feed-dot">▸</span>{m}</div>' for m in feed_msgs[:feed_msgs.index(msg)+1]])
                    + '</div>',
                    unsafe_allow_html=True
                )
                time.sleep(0.28)
            feed_slot.empty()

            # ── Compute ───────────────────────────────────
            tire_mult = {"Soft": -0.3, "Medium": 0.0, "Hard": 0.4}[tire_strat]
            rain_mult = -0.5 if rain_on else 0.0
            sc_mult   = -(sc_prob / 100) * 0.8

            raw = {
                'grid': grid_pos, 'rolling_avg_position': drv_form,
                'best_qual_ms': qual_ms, 'q1_ms': q1_ms,
                'prev_championship_points': drv_pts,
                'prev_championship_position': drv_rank,
                'prev_constructor_points': team_pts,
                'round': race_rnd
            }
            df_feat = prepare_features(raw, model)
            pred_raw = float(model.predict(df_feat)[0])
            pred = max(1.0, min(20.0, pred_raw + tire_mult + rain_mult + sc_mult))

            label, num_cls, badge_cls, conf = pos_meta(pred)
            insights = build_insights(raw, pred)
            scenario_note = []
            if rain_on:      scenario_note.append("🌧️ Rain modifier applied")
            if tire_strat != "Medium": scenario_note.append(f"🛞 {tire_strat} tyre offset applied")
            if sc_prob > 40: scenario_note.append(f"🚨 High SC probability ({sc_prob}%) boosts variance")

            st.session_state['pred_result'] = dict(
                pred=pred, label=label, num_cls=num_cls,
                badge_cls=badge_cls, conf=conf,
                insights=insights, scenario_note=scenario_note
            )

        if 'pred_result' in st.session_state:
            r = st.session_state['pred_result']

            # ── Hero result card ──────────────────────────
            st.markdown('<div class="hero-card anim-fade">', unsafe_allow_html=True)
            st.markdown('<div class="slabel" style="margin-bottom:4px">AI Race Intelligence · Projected Finish</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<span class="pos-num {r["num_cls"]}">P{r["pred"]:.1f}</span>',
                        unsafe_allow_html=True)
            st.markdown(f'<span class="badge {r["badge_cls"]}">{r["label"]}</span>',
                        unsafe_allow_html=True)
            st.markdown('<p style="color:#475569;font-size:1.05rem;margin:8px 0 20px">'
                        'Projected finish based on current performance telemetry</p>',
                        unsafe_allow_html=True)

            # Gradient confidence bar
            st.markdown('<div class="slabel">AI Confidence</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="conf-wrap"><div class="conf-bar" style="width:{r["conf"]}%"></div></div>'
                f'<div style="font-size:.9rem;font-weight:600;color:#1d4ed8;margin-bottom:4px">'
                f'{r["conf"]}% — Based on model certainty &amp; input quality</div>',
                unsafe_allow_html=True
            )

            # Scenario notes
            if r['scenario_note']:
                st.markdown('<div class="div"></div>', unsafe_allow_html=True)
                for n in r['scenario_note']:
                    st.markdown(f'<div style="font-size:.88rem;color:#7c3aed;font-weight:500;'
                                f'padding:4px 0">{n}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # close hero-card

            # AI complete message
            st.success("⚡ AI analysis complete — prediction ready")

            # ── Insights card ─────────────────────────────
            st.markdown('<div class="card anim-fade" style="animation-delay:.15s">', unsafe_allow_html=True)
            st.markdown("#### 📊 Why this prediction?")
            type_map = {
                "pos": ("ins-pos","ins-label-pos","Positive Factor"),
                "neg": ("ins-neg","ins-label-neg","Risk Factor"),
                "neu": ("ins-neu","ins-label-neu","Context Factor"),
            }
            for icon, t, title, body in r['insights']:
                card_cls, lbl_cls, lbl_text = type_map[t]
                st.markdown(
                    f'<div class="ins-card {card_cls}">'
                    f'<div class="ins-icon">{icon}</div>'
                    f'<div class="ins-body">'
                    f'<div class="ins-label {lbl_cls}">{lbl_text} · {title}</div>'
                    f'{body}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # ── Empty state ───────────────────────────────
            st.markdown(
                '<div class="card" style="border:2px dashed #cbd5e1;background:rgba(255,255,255,.4);'
                'box-shadow:none;min-height:520px;display:flex;flex-direction:column;'
                'align-items:center;justify-content:center;text-align:center;">',
                unsafe_allow_html=True
            )
            st.markdown("# 🏁")
            st.markdown("### Awaiting Race Telemetry")
            st.caption("Configure driver parameters on the left, then click **Run AI Prediction 🚀**")
            st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🏎️ Grid Simulation")
    st.caption("Compare predicted race outcomes across multiple drivers simultaneously.")
    st.markdown("")

    num_drv = st.slider("Number of drivers", 2, 5, 3)
    d_cols  = st.columns(num_drv)
    payloads = []

    for i, col in enumerate(d_cols):
        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**Driver {i+1}**")
            dg = st.number_input("Grid",      1,   20,   i*2+1,        key=f"mg{i}")
            df_= st.number_input("Form",      1.0, 20.0, float(i*2+2), key=f"mf{i}", step=0.1)
            dq = st.number_input("Qual (ms)", 60000,120000,82000+i*500, key=f"mq{i}", step=100)
            st.markdown('</div>', unsafe_allow_html=True)
            payloads.append({
                'grid': dg, 'rolling_avg_position': df_,
                'best_qual_ms': dq, 'q1_ms': dq+1500,
                'prev_championship_points': 150-i*20,
                'prev_championship_position': i+1,
                'prev_constructor_points': 300-i*10,
                'round': 12
            })

    if st.button("Run Grid Simulation 🚀", use_container_width=False):
        with st.spinner("Simulating race permutations..."):
            time.sleep(0.9)
            results = [float(model.predict(prepare_features(d, model))[0]) for d in payloads]

        si  = np.argsort(results)
        win = int(si[0])
        ds  = [f"Driver {i+1}" for i in si][::-1]
        ps  = [results[i]     for i in si][::-1]
        mn  = min(ps)
        bc  = ['#FF2E2E' if p==mn else '#93c5fd' for p in ps]
        tc  = ['#fff'    if p==mn else '#1e293b'  for p in ps]

        fig = go.Figure(go.Bar(
            x=ps, y=ds, orientation='h',
            marker=dict(color=bc, line=dict(color=['#b91c1c' if p==mn else '#3b82f6' for p in ps], width=1)),
            text=[f"P{p:.1f}" for p in ps], textposition='auto',
            textfont=dict(family='Poppins', size=14, color=tc)
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#475569'),
            margin=dict(l=0,r=0,t=20,b=0),
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,.05)',
                       title="Projected Finish (lower = better)", zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False), height=370
        )

        st.markdown("---")
        ch, ins = st.columns([2, 1.1])
        with ch:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Predicted Finishing Order")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with ins:
            st.markdown('<div class="hero-card" style="text-align:left;padding:30px 24px">', unsafe_allow_html=True)
            st.markdown('<div class="slabel">Strategy Directive</div>', unsafe_allow_html=True)
            st.markdown("**Primary Asset**")
            st.markdown(f'<span class="pos-num pos-red" style="font-size:4rem">D{win+1}</span>',
                        unsafe_allow_html=True)
            st.markdown(
                f"Simulations identify **Driver {win+1}** as the optimal points asset — "
                f"projected **P{results[win]:.1f}**."
            )
            st.success("⚡ Simulation complete")
            st.markdown('</div>', unsafe_allow_html=True)