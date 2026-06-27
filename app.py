import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1rem 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #818cf8;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        white-space: nowrap;
    }
    .segment-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    .rec-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .section-header {
        color: #e2e8f0;
        font-size: 1.4rem;
        font-weight: 700;
        border-bottom: 2px solid #334155;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Load models & data ───────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl','rb') as f: km = pickle.load(f)
    with open('scaler.pkl','rb') as f: scaler = pickle.load(f)
    with open('item_sim.pkl','rb') as f: item_sim_df = pickle.load(f)
    with open('labels_map.json') as f: labels_map = json.load(f)
    return km, scaler, item_sim_df, labels_map

@st.cache_data
def load_data():
    df = pd.read_csv('clean_data.csv')
    rfm = pd.read_csv('rfm_data.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    return df, rfm

km, scaler, item_sim_df, labels_map = load_models()
df, rfm = load_data()

SEGMENT_COLORS = {
    'High-Value':  {'bg':'#4f46e5','text':'#e0e7ff','emoji':'👑'},
    'Regular':     {'bg':'#059669','text':'#d1fae5','emoji':'✅'},
    'Occasional':  {'bg':'#d97706','text':'#fef3c7','emoji':'⏰'},
    'At-Risk':     {'bg':'#dc2626','text':'#fee2e2','emoji':'⚠️'},
}

SEGMENT_DESCRIPTIONS = {
    'High-Value':  'Top spenders who buy frequently and recently. Prioritize for loyalty rewards and premium offers.',
    'Regular':     'Steady, consistent buyers. Great candidates for upselling and cross-selling campaigns.',
    'Occasional':  'Infrequent buyers. Nurture with targeted re-engagement emails and discount incentives.',
    'At-Risk':     "Haven't purchased in a long time. Act fast with win-back campaigns and exclusive deals.",
}

# ─── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Dashboard",
        "🔍 Customer Segmentation",
        "🎯 Product Recommendations",
        "📈 EDA Insights"
    ])
    st.markdown("---")
    st.markdown(f"**Dataset**")
    st.markdown(f"- Transactions: `{len(df):,}`")
    st.markdown(f"- Customers: `{rfm.shape[0]:,}`")
    st.markdown(f"- Products: `{df['Description'].nunique():,}`")
    st.markdown(f"- Countries: `{df['Country'].nunique():,}`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown('<div class="hero-title">🛒 Shopper Spectrum</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Customer Segmentation & Product Recommendation in E-Commerce</div>', unsafe_allow_html=True)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    total_revenue = df['TotalAmount'].sum()
    avg_order = df.groupby('InvoiceNo')['TotalAmount'].sum().mean()
    top_country = df.groupby('Country')['TotalAmount'].sum().idxmax()
    
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">£{total_revenue:,.0f}</div>
            <div class="metric-label">Total Revenue</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{rfm.shape[0]:,}</div>
            <div class="metric-label">Total Customers</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">£{avg_order:,.1f}</div>
            <div class="metric-label">Avg Order Value</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{top_country}</div>
            <div class="metric-label">Top Market</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue trend + Segment distribution
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-header">📅 Monthly Revenue Trend</div>', unsafe_allow_html=True)
        monthly = df.resample('ME', on='InvoiceDate')['TotalAmount'].sum().reset_index()
        monthly.columns = ['Month','Revenue']
        fig = px.area(monthly, x='Month', y='Revenue',
                      color_discrete_sequence=['#6366f1'],
                      template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)',
                         margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        fig.update_traces(fill='tozeroy', fillcolor='rgba(99,102,241,0.15)')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">👥 Customer Segments</div>', unsafe_allow_html=True)
        seg_counts = rfm['Segment'].value_counts().reset_index()
        seg_counts.columns = ['Segment','Count']
        colors = [SEGMENT_COLORS[s]['bg'] for s in seg_counts['Segment']]
        fig2 = px.pie(seg_counts, names='Segment', values='Count',
                      color_discrete_sequence=colors, hole=0.5,
                      template='plotly_dark')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # Top products + Country sales
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">🏆 Top 10 Products by Revenue</div>', unsafe_allow_html=True)
        top_prod = df.groupby('Description')['TotalAmount'].sum().nlargest(10).reset_index()
        fig3 = px.bar(top_prod, x='TotalAmount', y='Description', orientation='h',
                      color='TotalAmount', color_continuous_scale='Purples',
                      template='plotly_dark')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)',
                          margin=dict(l=0,r=0,t=10,b=0), showlegend=False,
                          yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">🌍 Revenue by Country</div>', unsafe_allow_html=True)
        country_rev = df.groupby('Country')['TotalAmount'].sum().nlargest(8).reset_index()
        fig4 = px.bar(country_rev, x='Country', y='TotalAmount',
                      color='TotalAmount', color_continuous_scale='Viridis',
                      template='plotly_dark')
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)',
                          margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Customer Segmentation":
    st.markdown('<div class="hero-title">🔍 Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Predict which segment a customer belongs to using RFM values</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        st.markdown("### Enter Customer RFM Values")
        st.markdown("---")
        recency = st.number_input("📅 Recency (days since last purchase)", min_value=1, max_value=730, value=30, step=1)
        frequency = st.number_input("🔁 Frequency (number of transactions)", min_value=1, max_value=50, value=8, step=1)
        monetary = st.number_input("💰 Monetary (total spend in £)", min_value=1.0, max_value=5000.0, value=350.0, step=10.0)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Predict Segment", use_container_width=True, type="primary")

        # Quick reference
        st.markdown("---")
        st.markdown("**💡 Segment Benchmarks**")
        seg_bench = rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(1)
        st.dataframe(seg_bench, use_container_width=True)

    with col_right:
        if predict_btn:
            X_input = scaler.transform([[recency, frequency, monetary]])
            cluster_id = km.predict(X_input)[0]
            segment = labels_map[str(cluster_id)]
            info = SEGMENT_COLORS[segment]
            desc = SEGMENT_DESCRIPTIONS[segment]

            st.markdown(f"""
            <div style="background:{info['bg']}22; border:2px solid {info['bg']}; border-radius:16px; padding:2rem; text-align:center; margin-bottom:1rem;">
                <div style="font-size:3rem">{info['emoji']}</div>
                <div style="color:{info['text']}; font-size:2rem; font-weight:800; margin:0.5rem 0">{segment}</div>
                <div style="color:#cbd5e1; font-size:0.95rem">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # RFM summary cards
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Recency", f"{recency}d", delta=f"{'✅ Recent' if recency < 60 else '⚠️ Old'}", delta_color="off")
            with c2:
                st.metric("Frequency", f"{frequency}", delta=f"{'✅ Active' if frequency >= 8 else '⚠️ Low'}", delta_color="off")
            with c3:
                st.metric("Monetary", f"£{monetary:,.0f}", delta=f"{'✅ High' if monetary >= 300 else '⚠️ Low'}", delta_color="off")

            # RFM radar vs avg
            st.markdown("#### 📊 Your RFM vs Segment Average")
            avg = rfm[rfm['Segment']==segment][['Recency','Frequency','Monetary']].mean()
            cats = ['Recency (inv)','Frequency','Monetary']
            # Invert recency so lower = better on radar
            max_r = rfm['Recency'].max()
            user_vals = [1 - recency/max_r, frequency/rfm['Frequency'].max(), monetary/rfm['Monetary'].max()]
            avg_vals  = [1 - avg['Recency']/max_r, avg['Frequency']/rfm['Frequency'].max(), avg['Monetary']/rfm['Monetary'].max()]

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=user_vals+[user_vals[0]], theta=cats+[cats[0]], fill='toself',
                                          name='You', line_color='#818cf8'))
            fig.add_trace(go.Scatterpolar(r=avg_vals+[avg_vals[0]], theta=cats+[cats[0]], fill='toself',
                                          name=f'{segment} Avg', line_color='#34d399', opacity=0.5))
            fig.update_layout(polar=dict(bgcolor='#1e293b',
                              radialaxis=dict(visible=True, range=[0,1], gridcolor='#334155'),
                              angularaxis=dict(gridcolor='#334155')),
                              paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color='#e2e8f0')),
                              font=dict(color='#e2e8f0'), margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("""
            <div style="background:#1e293b; border:1px dashed #334155; border-radius:16px; padding:3rem; text-align:center; color:#64748b;">
                <div style="font-size:3rem; margin-bottom:1rem">👈</div>
                <div style="font-size:1.1rem">Enter RFM values and click <strong style="color:#818cf8">Predict Segment</strong></div>
            </div>
            """, unsafe_allow_html=True)

        # 3D Cluster visualization
        st.markdown("#### 🔮 All Customer Clusters (3D)")
        fig3d = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary',
                              color='Segment',
                              color_discrete_map={k: v['bg'] for k,v in SEGMENT_COLORS.items()},
                              opacity=0.7, size_max=4,
                              template='plotly_dark',
                              hover_data={'CustomerID':True})
        fig3d.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                           scene=dict(bgcolor='#1e293b',
                                      xaxis=dict(gridcolor='#334155'),
                                      yaxis=dict(gridcolor='#334155'),
                                      zaxis=dict(gridcolor='#334155')),
                           margin=dict(l=0,r=0,t=0,b=0), height=420)
        st.plotly_chart(fig3d, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRODUCT RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Product Recommendations":
    st.markdown('<div class="hero-title">🎯 Product Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Item-based Collaborative Filtering using Cosine Similarity</div>', unsafe_allow_html=True)

    all_products = sorted(item_sim_df.index.tolist())

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("### 🔎 Search Product")
        selected = st.selectbox("Choose a product:", all_products, index=all_products.index('PARTY BUNTING') if 'PARTY BUNTING' in all_products else 0)
        n_recs = st.slider("Number of recommendations", 3, 10, 5)
        rec_btn = st.button("✨ Get Recommendations", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown("**📦 Product Stats**")
        prod_df = df[df['Description'] == selected]
        total_sold  = int(prod_df['Quantity'].sum())
        customers   = int(prod_df['CustomerID'].nunique())
        avg_price   = float(prod_df['UnitPrice'].mean())
        revenue     = float(prod_df['TotalAmount'].sum())
        st.metric("Total Sold",     f"{total_sold:,}")
        st.metric("Unique Buyers",  f"{customers:,}")
        st.metric("Avg Unit Price", f"£{avg_price:.2f}")
        st.metric("Total Revenue",  f"£{revenue:,.2f}")

    with col2:
        if rec_btn or True:  # Show recommendations on load
            scores = item_sim_df[selected].drop(selected).sort_values(ascending=False).head(n_recs)
            recs = scores.reset_index()
            recs.columns = ['Product', 'Similarity']

            st.markdown(f"### Customers who bought **{selected}** also bought:")
            for i, row in recs.iterrows():
                pct = int(row['Similarity'] * 100)
                bar_color = '#6366f1' if pct > 70 else '#8b5cf6' if pct > 50 else '#a78bfa'
                st.markdown(f"""
                <div class="rec-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span>🛍️ <strong>{row['Product']}</strong></span>
                        <span style="color:#818cf8; font-size:0.85rem">{pct}% match</span>
                    </div>
                    <div style="background:#334155; border-radius:4px; height:5px; margin-top:0.5rem;">
                        <div style="background:{bar_color}; width:{pct}%; height:5px; border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Similarity heatmap for top 15 products
            st.markdown("#### 🔥 Product Similarity Heatmap (Top 15 Products)")
            top15 = df.groupby('Description')['TotalAmount'].sum().nlargest(15).index.tolist()
            heat_data = item_sim_df.loc[top15, top15]
            fig_heat = px.imshow(heat_data, color_continuous_scale='Purples',
                                  text_auto='.2f', template='plotly_dark', aspect='auto')
            fig_heat.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                   plot_bgcolor='#1e293b', height=450,
                                   margin=dict(l=0,r=0,t=0,b=0),
                                   coloraxis_showscale=False)
            fig_heat.update_xaxes(tickangle=45, tickfont=dict(size=9))
            fig_heat.update_yaxes(tickfont=dict(size=9))
            st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 EDA Insights":
    st.markdown('<div class="hero-title">📈 EDA Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Exploratory Data Analysis — Patterns, Trends & Distributions</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🧮 RFM Analysis","📦 Product Analysis","🌍 Geographic"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(rfm, x='Recency', nbins=40, color_discrete_sequence=['#6366f1'],
                               title='Recency Distribution', template='plotly_dark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)')
            st.plotly_chart(fig, use_container_width=True)

            fig = px.histogram(rfm, x='Monetary', nbins=40, color_discrete_sequence=['#8b5cf6'],
                               title='Monetary Distribution', template='plotly_dark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(rfm, x='Frequency', nbins=30, color_discrete_sequence=['#ec4899'],
                               title='Frequency Distribution', template='plotly_dark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)')
            st.plotly_chart(fig, use_container_width=True)

            # Elbow curve
            k_range = list(range(2, 8))
            inertias = []
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            sc = StandardScaler()
            Xs = sc.fit_transform(rfm[['Recency','Frequency','Monetary']])
            for k in k_range:
                km_t = KMeans(n_clusters=k, random_state=42, n_init=10)
                km_t.fit(Xs)
                inertias.append(km_t.inertia_)
            elbow_df = pd.DataFrame({'K': k_range, 'Inertia': inertias})
            fig_e = px.line(elbow_df, x='K', y='Inertia', markers=True,
                            color_discrete_sequence=['#f59e0b'],
                            title='Elbow Curve — Optimal K', template='plotly_dark')
            fig_e.add_vline(x=4, line_dash='dash', line_color='#ef4444',
                            annotation_text='k=4 chosen', annotation_font_color='#ef4444')
            fig_e.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)')
            st.plotly_chart(fig_e, use_container_width=True)

        # Box plots
        fig_box = px.box(rfm, x='Segment', y='Monetary',
                         color='Segment',
                         color_discrete_map={k: v['bg'] for k,v in SEGMENT_COLORS.items()},
                         title='Monetary Value by Segment', template='plotly_dark')
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)')
        st.plotly_chart(fig_box, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            top20 = df.groupby('Description')['Quantity'].sum().nlargest(20).reset_index()
            fig = px.bar(top20, x='Quantity', y='Description', orientation='h',
                         color='Quantity', color_continuous_scale='Viridis',
                         title='Top 20 Products by Quantity Sold', template='plotly_dark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)',
                             yaxis={'categoryorder':'total ascending'}, height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            monthly_prod = df.groupby([df['InvoiceDate'].dt.to_period('M').astype(str), 'Description'])['Quantity'].sum().reset_index()
            top5 = df.groupby('Description')['Quantity'].sum().nlargest(5).index.tolist()
            m5 = monthly_prod[monthly_prod['Description'].isin(top5)]
            fig2 = px.line(m5, x='InvoiceDate', y='Quantity', color='Description',
                          title='Top 5 Products — Monthly Sales', template='plotly_dark')
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)', height=500)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            country_data = df.groupby('Country').agg(
                Revenue=('TotalAmount','sum'),
                Customers=('CustomerID','nunique'),
                Orders=('InvoiceNo','nunique')
            ).reset_index().sort_values('Revenue', ascending=False)

            fig = px.bar(country_data, x='Country', y='Revenue',
                        color='Revenue', color_continuous_scale='Purples',
                        title='Revenue by Country', template='plotly_dark')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(country_data, x='Customers', y='Revenue', size='Orders',
                             color='Country', text='Country',
                             title='Customers vs Revenue by Country', template='plotly_dark')
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,23,42,0.8)')
            st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(country_data, use_container_width=True)
