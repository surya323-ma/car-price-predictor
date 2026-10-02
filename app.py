"""Car Price Predictor - Streamlit app.   Run:  streamlit run app.py"""
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import CURRENT_YEAR, OWNER_ORDER
from src.predict import CarSpec, Predictor, deal_verdict, emi
from src.ui import CSS, FUEL_COLORS, PALETTE, band_html, fmt_price, kpi_html, plate_html, style_fig

st.set_page_config(page_title="🚗 AutoValue AI ", page_icon="🚗", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def get_predictor() -> Predictor:
    return Predictor()


P = get_predictor()
CAT = P.catalogue()
META = P.meta
DATA = P.data
st.session_state.setdefault("saved", [])

# ------------------------------------------------------------------ header
st.markdown(
    f"""<div class="hero"><h1>What is your used car really worth?</h1>
    <p>Get an instant fair-price estimate with a realistic range, see how the value will fall over the next years,
    and check whether a seller's asking price is a good deal.</p>
    <div class="stats"><span>{META['n_rows']:,}<small>cleaned listings</small></span>
    <span>{len(CAT)}<small>brands</small></span>
    <span>±₹{META['holdout']['mae_lakh']:.1f} L<small>typical error</small></span>
    <span>{META['holdout']['median_ape_pct']:.0f}%<small>median error</small></span></div></div>""",
    unsafe_allow_html=True,
)
st.write("")
tab_est, tab_cmp, tab_mkt, tab_ins, tab_about = st.tabs(
    ["Estimate", "Compare", "Market explorer", "Model insights", "About"])

# ------------------------------------------------------------------ ESTIMATE
with tab_est:
    left, right = st.columns([5, 7], gap="large")

    with left:
        with st.container(border=True):
            st.subheader("Describe the car")
            brands = sorted(CAT)
            brand = st.selectbox("Brand", brands, index=brands.index("Maruti") if "Maruti" in brands else 0)
            models = sorted(CAT[brand], key=lambda m: -CAT[brand][m]["n"])
            model = st.selectbox("Model", models)
            info = CAT[brand][model]
            k = f"{brand}|{model}"  # widget keys change with the model so smart defaults reset

            reg_year = st.slider("Registration year", 2007, CURRENT_YEAR, 2019)
            kms = st.slider("Kilometres driven", 0, 250_000, 50_000, step=1_000, format="%d km")
            c1, c2 = st.columns(2)
            fuel = c1.selectbox("Fuel", info["fuels"], key=f"fuel|{k}")
            transmission = c2.radio("Transmission", ["Manual", "Automatic"], horizontal=True,
                                    index=0 if info["transmissions"][0] == "Manual" else 1, key=f"tr|{k}")
            ownership = st.selectbox("Ownership", OWNER_ORDER)
            with st.expander("More details (optional)"):
                insurance = st.selectbox("Insurance", ["Comprehensive", "Zero Dep", "Third Party", "Not Available"])
                seats = st.selectbox("Seats", [4, 5, 6, 7, 8], index=[4, 5, 6, 7, 8].index(info["seats"]), key=f"s|{k}")
                engine = st.number_input("Engine (cc)", 600, 6500, int(min(max(info["engine"], 600), 6500)),
                                         step=50, key=f"e|{k}")
            with st.expander("Deal check: seller's asking price (optional)"):
                asking = st.number_input("Asking price (₹ lakh)", 0.0, 500.0, 0.0, step=0.25, format="%.2f",
                                         help="Leave at 0 to skip.")

    spec = CarSpec(brand, model, fuel, transmission, ownership, insurance, CURRENT_YEAR - reg_year, kms, seats, float(engine))
    est = P.estimate(spec)

    with right:
        with st.container(border=True):
            st.markdown(plate_html(est["price"], f"Estimated fair price · {spec.title}"), unsafe_allow_html=True)
            st.markdown(band_html(est["low"], est["price"], est["high"]), unsafe_allow_html=True)
            st.markdown(f'<p class="small">80% of similar cars sell inside this range. '
                        f'Median {model} in the dataset: {fmt_price(info["median_price"])} ({info["n"]} listings).</p>',
                        unsafe_allow_html=True)

            if asking > 0:
                label, why, tone = deal_verdict(asking, est)
                st.markdown(f'<div class="verdict {tone}"><b>{label}</b><br>{why}</div>', unsafe_allow_html=True)

            fut = P.future_values(spec)
            cols = st.columns(4)
            for col, (yrs, val) in zip(cols, fut.items()):
                chg = (val / est["price"] - 1) * 100
                col.markdown(kpi_html(f"Value in {yrs} yr", fmt_price(val), f"{chg:+.0f}%", "neg" if chg < 0 else "pos"),
                             unsafe_allow_html=True)

            b1, b2 = st.columns(2)
            if b1.button("Save to compare", width="stretch"):
                st.session_state.saved.append({"spec": spec, "est": est})
                st.toast(f"Saved {spec.title}")
            wi = P.what_if(spec)
            report = "\n".join([
                "CAR PRICE REPORT", f"Generated: {date.today():%d %b %Y}", "",
                f"Car: {spec.title}", f"Fuel: {fuel} | Transmission: {transmission} | Owner: {ownership}",
                f"Driven: {kms:,} km | Insurance: {insurance} | Engine: {engine} cc | Seats: {seats}", "",
                f"Estimated fair price: Rs {est['price']:.2f} lakh",
                f"Likely range (80%): Rs {est['low']:.2f} - {est['high']:.2f} lakh", "",
                "Projected value:", *[f"  In {y} yr: Rs {v:.2f} lakh" for y, v in fut.items()], "",
                "What moves the price:", *[f"  {r['Change']}: {r['Price impact (lakh)']:+.2f} lakh" for _, r in wi.iterrows()],
                "", "Estimates come from a statistical model trained on listed prices; treat as guidance, not a valuation."])
            b2.download_button("Download report", report, file_name=f"{brand}_{model}_price_report.txt".replace(" ", "_"),
                               width="stretch")

        t1, t2, t3, t4 = st.tabs(["Value over time", "What changes the price", "Similar listings", "EMI planner"])

        with t1:
            dep = P.depreciation(spec)
            fig = go.Figure(go.Scatter(x=dep["age"], y=dep["price"], mode="lines+markers",
                                       line=dict(color=PALETTE["green"], width=3), marker=dict(size=6),
                                       hovertemplate="Age %{x} yr: ₹%{y:.2f} L<extra></extra>"))
            fig.add_trace(go.Scatter(x=[spec.car_age], y=[est["price"]], mode="markers",
                                     marker=dict(size=15, color=PALETTE["plate"], line=dict(width=3, color=PALETTE["ink"])),
                                     name="This car", hovertemplate="This car: ₹%{y:.2f} L<extra></extra>"))
            fig.update_layout(xaxis_title="Car age (years)", yaxis_title="Price (₹ lakh)", showlegend=False)
            st.plotly_chart(style_fig(fig), width="stretch")
            st.caption("Assumes the car keeps being driven at its current yearly pace.")

        with t2:
            wi_plot = wi.copy()
            wi_plot["tone"] = wi_plot["Price impact (lakh)"].map(lambda v: PALETTE["green"] if v >= 0 else PALETTE["red"])
            fig = go.Figure(go.Bar(x=wi_plot["Price impact (lakh)"], y=wi_plot["Change"], orientation="h",
                                   marker_color=wi_plot["tone"], hovertemplate="%{y}: %{x:+.2f} L<extra></extra>"))
            fig.update_layout(xaxis_title="Change in price (₹ lakh)", yaxis_title="")
            st.plotly_chart(style_fig(fig, 300), width="stretch")

        with t3:
            sim = P.similar(spec).rename(columns={"car_name": "Car", "fuel_type": "Fuel", "transmission": "Gearbox",
                                                  "kms_driven": "Km", "ownership": "Owner", "price_lakh": "Price (₹ L)"})
            st.dataframe(sim, hide_index=True, width="stretch",
                         column_config={"Km": st.column_config.NumberColumn(format="%d"),
                                        "Price (₹ L)": st.column_config.NumberColumn(format="%.2f")})
            st.caption("Closest real listings by model, age, kilometres, fuel and gearbox.")

        with t4:
            e1, e2, e3, e4 = st.columns(4)
            price_in = e1.number_input("Car price (₹ L)", 0.5, 500.0, round(asking if asking > 0 else est["price"], 2), step=0.25)
            down = e2.slider("Down payment (%)", 0, 80, 20)
            rate = e3.slider("Interest (% p.a.)", 6.0, 18.0, 10.0, step=0.25)
            yrs = e4.slider("Tenure (years)", 1, 7, 4)
            loan = price_in * (1 - down / 100)
            r = emi(loan, rate, yrs)
            m1, m2, m3 = st.columns(3)
            m1.markdown(kpi_html("Monthly EMI", f"₹{r['monthly']:,.0f}"), unsafe_allow_html=True)
            m2.markdown(kpi_html("Total interest", f"₹{r['interest']:,.0f}"), unsafe_allow_html=True)
            m3.markdown(kpi_html("Loan amount", fmt_price(loan)), unsafe_allow_html=True)

# ------------------------------------------------------------------ COMPARE
with tab_cmp:
    saved = st.session_state.saved
    if not saved:
        st.markdown('<div class="empty"><b>No cars saved yet</b><br>Use “Save to compare” on the Estimate tab to line '
                    'up two or more cars here.</div>', unsafe_allow_html=True)
    else:
        rows = []
        for i, s in enumerate(saved, 1):
            c, e = s["spec"], s["est"]
            rows.append({"#": i, "Car": c.title, "Fuel": c.fuel_type, "Gearbox": c.transmission, "Km": c.kms_driven,
                         "Owner": c.ownership, "Estimate (₹ L)": e["price"], "Low": e["low"], "High": e["high"]})
        tbl = pd.DataFrame(rows)
        fig = go.Figure(go.Bar(x=[f"{r['#']}. {r['Car']}" for r in rows], y=tbl["Estimate (₹ L)"],
                               marker_color=PALETTE["green"],
                               error_y=dict(type="data", symmetric=False, array=tbl["High"] - tbl["Estimate (₹ L)"],
                                            arrayminus=tbl["Estimate (₹ L)"] - tbl["Low"], color=PALETTE["ink"]),
                               hovertemplate="%{x}: ₹%{y:.2f} L<extra></extra>"))
        fig.update_layout(yaxis_title="Estimated price (₹ lakh)")
        st.plotly_chart(style_fig(fig, 360), width="stretch")
        st.dataframe(tbl, hide_index=True, width="stretch",
                     column_config={"Km": st.column_config.NumberColumn(format="%d"),
                                    "Estimate (₹ L)": st.column_config.NumberColumn(format="%.2f"),
                                    "Low": st.column_config.NumberColumn(format="%.2f"),
                                    "High": st.column_config.NumberColumn(format="%.2f")})
        a, b = st.columns([1, 5])
        if a.button("Clear all"):
            st.session_state.saved = []
            st.rerun()
        b.download_button("Download comparison (CSV)", tbl.to_csv(index=False), "car_comparison.csv")

# ------------------------------------------------------------------ MARKET EXPLORER
with tab_mkt:
    f1, f2, f3, f4 = st.columns([3, 2, 3, 3])
    sel_brands = f1.multiselect("Brand", sorted(DATA["brand"].unique()), placeholder="All brands")
    sel_fuel = f2.multiselect("Fuel", sorted(DATA["fuel_type"].unique()), placeholder="All")
    pmax = float(DATA["price_lakh"].max())
    price_rng = f3.slider("Price (₹ lakh)", 0.0, float(round(pmax)), (0.0, float(round(pmax))))
    age_rng = f4.slider("Car age (years)", int(DATA["car_age"].min()), int(DATA["car_age"].max()),
                        (int(DATA["car_age"].min()), int(DATA["car_age"].max())))
    d = DATA
    if sel_brands:
        d = d[d["brand"].isin(sel_brands)]
    if sel_fuel:
        d = d[d["fuel_type"].isin(sel_fuel)]
    d = d[d["price_lakh"].between(*price_rng) & d["car_age"].between(*age_rng)]

    if d.empty:
        st.markdown('<div class="empty"><b>No listings match these filters</b><br>Widen the price or age range.</div>',
                    unsafe_allow_html=True)
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi_html("Listings", f"{len(d):,}"), unsafe_allow_html=True)
        k2.markdown(kpi_html("Median price", fmt_price(d["price_lakh"].median())), unsafe_allow_html=True)
        k3.markdown(kpi_html("Median age", f"{d['car_age'].median():.0f} yr"), unsafe_allow_html=True)
        k4.markdown(kpi_html("Median km", f"{d['kms_driven'].median():,.0f}"), unsafe_allow_html=True)
        st.write("")
        c1, c2 = st.columns(2)
        fig = px.scatter(d, x="car_age", y="price_lakh", color="fuel_type", color_discrete_map=FUEL_COLORS,
                         hover_name="car_name", opacity=0.75, labels={"car_age": "Car age (years)",
                                                                        "price_lakh": "Price (₹ lakh)", "fuel_type": "Fuel"})
        fig.update_layout(title="Price vs age")
        c1.plotly_chart(style_fig(fig, 380), width="stretch")
        top = d.groupby("brand")["price_lakh"].median().sort_values().tail(12).reset_index()
        fig = px.bar(top, x="price_lakh", y="brand", orientation="h", color_discrete_sequence=[PALETTE["green"]],
                     labels={"price_lakh": "Median price (₹ lakh)", "brand": ""})
        fig.update_layout(title="Median price by brand")
        c2.plotly_chart(style_fig(fig, 380), width="stretch")
        c3, c4 = st.columns(2)
        fig = px.histogram(d, x="price_lakh", nbins=30, color_discrete_sequence=[PALETTE["ink"]],
                           labels={"price_lakh": "Price (₹ lakh)"})
        fig.update_layout(title="Price distribution", yaxis_title="Listings")
        c3.plotly_chart(style_fig(fig, 320), width="stretch")
        fig = px.scatter(d, x="kms_driven", y="price_lakh", color="transmission", hover_name="car_name", opacity=0.75,
                         color_discrete_sequence=[PALETTE["green"], PALETTE["plate"]],
                         labels={"kms_driven": "Kilometres driven", "price_lakh": "Price (₹ lakh)", "transmission": "Gearbox"})
        fig.update_layout(title="Price vs kilometres")
        c4.plotly_chart(style_fig(fig, 320), width="stretch")
        show = d[["car_name", "fuel_type", "transmission", "kms_driven", "ownership", "insurance", "price_lakh"]]
        st.dataframe(show.rename(columns={"car_name": "Car", "fuel_type": "Fuel", "transmission": "Gearbox",
                                          "kms_driven": "Km", "ownership": "Owner", "insurance": "Insurance",
                                          "price_lakh": "Price (₹ L)"}),
                     hide_index=True, width="stretch", height=300,
                     column_config={"Km": st.column_config.NumberColumn(format="%d"),
                                    "Price (₹ L)": st.column_config.NumberColumn(format="%.2f")})
        st.download_button("Download filtered listings (CSV)", show.to_csv(index=False), "filtered_listings.csv")

# ------------------------------------------------------------------ MODEL INSIGHTS
with tab_ins:
    h = META["holdout"]
    cvb = META["cv_results"][META["best_model"]]
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(kpi_html("Model", META["best_model"].replace("Monotone ", "")), unsafe_allow_html=True)
    m2.markdown(kpi_html("R² (hold-out)", f"{h['r2']:.2f}", f"CV {cvb['cv_r2']:.2f}"), unsafe_allow_html=True)
    m3.markdown(kpi_html("Mean error", f"₹{h['mae_lakh']:.1f} L"), unsafe_allow_html=True)
    m4.markdown(kpi_html("Median error", f"{h['median_ape_pct']:.0f}%"), unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns(2)
    res = pd.DataFrame(META["cv_results"]).T.reset_index().rename(columns={"index": "Model"})
    fig = px.bar(res.sort_values("cv_r2"), x="cv_r2", y="Model", orientation="h",
                 color_discrete_sequence=[PALETTE["green"]], labels={"cv_r2": "Cross-validated R²"})
    fig.update_layout(title="Models compared")
    c1.plotly_chart(style_fig(fig, 320), width="stretch")
    imp = pd.Series(META["importance"]).sort_values().reset_index()
    imp.columns = ["Feature", "Importance"]
    imp["Feature"] = imp["Feature"].replace({"model": "🚗Car model", "engine_cc": "Engine size", "car_age": "Age",
                                             "kms_driven": "Kilometres", "brand": "Brand", "transmission": "Gearbox",
                                             "insurance": "Insurance", "seats": "Seats", "fuel_type": "⛽Fuel",
                                             "ownership": "🤵🏻‍♂️Ownership"})
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[PALETTE["ink"]])
    fig.update_layout(title="What drives price")
    c2.plotly_chart(style_fig(fig, 320), width="stretch")

    pa = pd.DataFrame(META["oof_actual_pred"], columns=["Actual", "Predicted"])
    mx = float(pa.max().max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pa["Actual"], y=pa["Predicted"], mode="markers", opacity=0.55,
                             marker=dict(color=PALETTE["green"], size=6), name="Listings"))
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=PALETTE["ink"], dash="dash"),
                             name="Perfect prediction"))
    fig.update_layout(title="Predicted vs actual price (out-of-sample)", xaxis_title="Actual (₹ lakh)",
                      yaxis_title="Predicted (₹ lakh)", xaxis_type="log", yaxis_type="log")
    st.plotly_chart(style_fig(fig, 420), width="stretch")
    st.caption("Points near the dashed line are accurate. Each prediction here was made by a model that never saw that car.")

# ------------------------------------------------------------------ ABOUT
with tab_about:
    st.markdown(f"""
### How it works
The estimate comes from a **Random Forest trained on {META['n_rows']:,} cleaned used-car listings**. It predicts the logarithm of
the price, so a ₹1 L error on a cheap hatchback counts as much as a proportional error on a luxury SUV. The shaded range is the
10th–90th percentile of the model's honest, out-of-sample errors, not a guess.

**Behaves sensibly.** The model is constrained so a car can never be predicted *more* valuable because it is older or has run
more kilometres, which keeps every curve and what-if consistent.

### Data cleaning
- Removed duplicate listings and 3 listings with impossible prices.
- Mileage, power and torque columns were corrupted in many rows, so they are **not used**; engine size is used only where valid.
- Brand and model are parsed from the listing title; car age is computed from the registration year.

### Limits
Small dataset: very rare models and exotic cars carry wider uncertainty. Condition, accident history, colour and city are not
captured. Treat the result as an informed starting point for negotiation, not a formal valuation.
""")
st.markdown(f'<p class="small" style="text-align:center;margin-top:2rem">🚗 AutoValue AI – Intelligent Car Price Prediction· prices in Indian rupees · '
            f'ages computed for {CURRENT_YEAR}</p>', unsafe_allow_html=True)
