import random
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from recommender import MovieRecommender

st.set_page_config(page_title="CineMatch — Smart Movie Recommender", page_icon="🍿", layout="wide")

# ---------------------------------------------------------------------- #
# Load engine (cached so the TF-IDF/similarity matrix is built only once)
# ---------------------------------------------------------------------- #
@st.cache_resource
def load_engine():
    return MovieRecommender("data/movies.csv")

rec = load_engine()

# ---------------------------------------------------------------------- #
# Visual identity — dark streaming-platform background + vibrant gradient
# "posters" (since we don't have real poster images) instead of default
# Streamlit white cards. Stored as hex PAIRS so the same colors can drive
# both CSS gradients and the Three.js carousel textures.
# ---------------------------------------------------------------------- #
GRADIENT_PAIRS = [
    ("#FF512F", "#DD2476"),
    ("#1FA2FF", "#12D8FA"),
    ("#FC5C7D", "#6A82FB"),
    ("#F7971E", "#FFD200"),
    ("#7F00FF", "#E100FF"),
    ("#11998E", "#38EF7D"),
    ("#F953C6", "#B91D73"),
    ("#4776E6", "#8E54E9"),
    ("#FF9A9E", "#FAD0C4"),
    ("#00C9FF", "#92FE9D"),
]
GENRE_EMOJI = {
    "Action": "💥", "Adventure": "🗺️", "Animation": "🎨", "Biography": "📖",
    "Comedy": "😂", "Crime": "🕵️", "Drama": "🎭", "Family": "👨‍👩‍👧",
    "Fantasy": "🧙", "History": "🏛️", "Horror": "👻", "Music": "🎵",
    "Musical": "🎶", "Mystery": "🔍", "Romance": "💕", "Sci-Fi": "🚀",
    "Thriller": "⚡", "War": "⚔️", "Western": "🤠",
}

def colors_for(title: str):
    return GRADIENT_PAIRS[hash(title) % len(GRADIENT_PAIRS)]

def gradient_for(title: str) -> str:
    c1, c2 = colors_for(title)
    return f"linear-gradient(135deg,{c1},{c2})"

def emoji_for(genres: str) -> str:
    first = genres.split()[0] if genres else ""
    return GENRE_EMOJI.get(first, "🎬")

def html(s: str) -> str:
    """Strip leading whitespace from every line so Streamlit's markdown
    renderer doesn't mistake indented HTML for a code block (Markdown
    treats 4+ leading spaces as an indented code block)."""
    return "\n".join(line.strip() for line in s.strip().splitlines())

st.markdown(
    """
    <style>
    .stApp { background: #0b0f1a; }
    .hero {
        background: linear-gradient(120deg,#7F00FF 0%,#E100FF 45%,#FF6B6B 100%);
        border-radius: 20px;
        padding: 34px 40px;
        margin-bottom: 26px;
        box-shadow: 0 10px 40px rgba(126,0,255,0.35);
    }
    .hero h1 { color: white; font-size: 2.4rem; margin: 0; }
    .hero p { color: rgba(255,255,255,0.9); font-size: 1.05rem; margin-top: 6px; }
    .pill {
        display: inline-block; padding: 4px 12px; margin: 2px 4px 2px 0;
        border-radius: 999px; font-size: 0.75rem; font-weight: 700;
        background: rgba(255,255,255,0.18); color: white;
    }
    .movie-card {
        border-radius: 16px; padding: 0; margin-bottom: 18px; overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .movie-card:hover { transform: translateY(-6px) scale(1.015); box-shadow: 0 14px 30px rgba(0,0,0,0.45); }
    .poster {
        height: 100px; display: flex; align-items: center; justify-content: center;
        font-size: 2.6rem;
    }
    .card-body { background: #131a2b; padding: 14px 16px 16px 16px; }
    .movie-title { font-size: 1.05rem; font-weight: 800; color: #f1f3f9; margin-bottom: 2px; }
    .movie-meta { color: #93a0b8; font-size: 0.82rem; margin-bottom: 8px; }
    .match-track { background: rgba(255,255,255,0.08); border-radius: 999px; height: 8px; overflow: hidden; margin-bottom: 10px;}
    .match-fill { height: 8px; border-radius: 999px; background: linear-gradient(90deg,#00C9FF,#92FE9D); }
    .why-chip {
        display: inline-block; font-size: 0.72rem; padding: 3px 9px; margin: 2px 4px 0 0;
        border-radius: 8px; background: rgba(255,107,107,0.15); color: #ff8f8f; border: 1px solid rgba(255,107,107,0.35);
    }
    /* --- 3D flip card (hover to reveal overview + why-matched) --- */
    .flip-card { perspective: 1200px; margin-bottom: 20px; height: 280px; }
    .flip-inner {
        position: relative; width: 100%; height: 100%;
        transition: transform 0.6s; transform-style: preserve-3d;
    }
    .flip-card:hover .flip-inner { transform: rotateY(180deg); }
    .flip-front, .flip-back {
        position: absolute; inset: 0; backface-visibility: hidden;
        border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,0.08);
    }
    .flip-back {
        transform: rotateY(180deg); background: #131a2b; padding: 16px;
        display: flex; flex-direction: column; justify-content: flex-start; overflow-y: auto;
    }
    .flip-back .back-title { color: #f1f3f9; font-weight: 800; font-size: 0.95rem; margin-bottom: 6px; }
    .flip-back .back-overview { color: #c7cee0; font-size: 0.8rem; line-height: 1.4; margin-bottom: 10px; }
    .flip-hint { color: #5d6a86; font-size: 0.68rem; text-align: center; margin-top: 4px; }
    /* --- simple 3D tilt for the trending/quiz mini cards --- */
    .tilt-hover { transition: transform 0.3s ease, box-shadow 0.3s ease; }
    .tilt-hover:hover { transform: perspective(700px) rotateX(4deg) rotateY(-8deg) scale(1.04); box-shadow: 0 14px 26px rgba(0,0,0,0.5); }
    .scroll-row { display: flex; overflow-x: auto; gap: 14px; padding-bottom: 10px; }
    .scroll-row::-webkit-scrollbar { height: 6px; }
    .scroll-row::-webkit-scrollbar-thumb { background: #4776E6; border-radius: 999px; }
    .mini-card {
        min-width: 150px; max-width: 150px; border-radius: 12px; overflow: hidden;
        border: 1px solid rgba(255,255,255,0.08); flex-shrink: 0;
        transition: transform 0.15s ease;
    }
    .mini-card:hover { transform: translateY(-4px); }
    .mini-poster { height: 70px; display: flex; align-items: center; justify-content: center; font-size: 1.8rem; }
    .mini-body { background: #131a2b; padding: 8px 10px; }
    .mini-title { color: #f1f3f9; font-size: 0.78rem; font-weight: 700; line-height: 1.1; }
    .mini-meta { color: #93a0b8; font-size: 0.68rem; margin-top: 2px; }
    section[data-testid="stSidebar"] { background: #0f1524; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------- #
# Hero banner
# ---------------------------------------------------------------------- #
stats = rec.model_stats()
st.markdown(
    html(f"""
    <div class="hero">
        <h1>🍿 CineMatch</h1>
        <p>A content-based recommendation engine — TF-IDF vectorization + cosine similarity over {stats['n_movies']} movies.</p>
        <span class="pill">🎞️ {stats['n_movies']} movies</span>
        <span class="pill">🧬 {stats['vocab_size']} vocabulary features</span>
        <span class="pill">📐 {stats['sparsity_pct']}% sparse matrix</span>
    </div>
    """),
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------- #
# 3D Poster Carousel — real Three.js scene, auto-rotating + mouse-reactive
# ---------------------------------------------------------------------- #
def render_3d_carousel(df: pd.DataFrame):
    movies_js = []
    for _, row in df.iterrows():
        c1, c2 = colors_for(row["title"])
        movies_js.append({
            "title": row["title"],
            "rating": float(row["rating"]),
            "c1": c1,
            "c2": c2,
        })
    movies_json = json.dumps(movies_js)

    carousel_html = f"""
    <div id="three-root" style="width:100%;height:420px;border-radius:20px;overflow:hidden;
         background:radial-gradient(circle at 50% 20%,#1a1f3a,#05070d);"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
    (function() {{
        const movies = {movies_json};
        const container = document.getElementById('three-root');
        const width = container.clientWidth;
        const height = 420;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
        camera.position.set(0, 1.1, 9);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        scene.add(new THREE.AmbientLight(0xffffff, 0.9));
        const point = new THREE.PointLight(0xffffff, 0.7);
        point.position.set(0, 5, 6);
        scene.add(point);

        const group = new THREE.Group();
        scene.add(group);

        function wrapText(ctx, text, x, y, maxWidth, lineHeight) {{
            const words = text.split(' ');
            let line = '';
            const lines = [];
            for (let n = 0; n < words.length; n++) {{
                const testLine = line + words[n] + ' ';
                if (ctx.measureText(testLine).width > maxWidth && n > 0) {{
                    lines.push(line);
                    line = words[n] + ' ';
                }} else {{
                    line = testLine;
                }}
            }}
            lines.push(line);
            const startY = y - ((lines.length - 1) * lineHeight) / 2;
            lines.forEach((l, idx) => ctx.fillText(l.trim(), x, startY + idx * lineHeight));
        }}

        const radius = 4.3;
        const count = movies.length;

        movies.forEach((m, i) => {{
            const canvas = document.createElement('canvas');
            canvas.width = 256; canvas.height = 384;
            const ctx = canvas.getContext('2d');
            const grad = ctx.createLinearGradient(0, 0, 256, 384);
            grad.addColorStop(0, m.c1);
            grad.addColorStop(1, m.c2);
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, 256, 384);

            ctx.fillStyle = 'rgba(255,255,255,0.95)';
            ctx.font = 'bold 22px Arial';
            ctx.textAlign = 'center';
            wrapText(ctx, m.title, 128, 300, 220, 26);

            ctx.font = '18px Arial';
            ctx.fillText('★ ' + m.rating, 128, 350);

            const texture = new THREE.CanvasTexture(canvas);
            const geo = new THREE.PlaneGeometry(2.2, 3.3);
            const mat = new THREE.MeshStandardMaterial({{ map: texture, side: THREE.DoubleSide }});
            const mesh = new THREE.Mesh(geo, mat);

            const angle = (i / count) * Math.PI * 2;
            mesh.position.set(Math.sin(angle) * radius, 0, Math.cos(angle) * radius);
            mesh.lookAt(0, 0, 0);
            group.add(mesh);
        }});

        let mouseX = 0, mouseY = 0;
        container.addEventListener('mousemove', (e) => {{
            const rect = container.getBoundingClientRect();
            mouseX = (e.clientX - rect.left) / width - 0.5;
            mouseY = (e.clientY - rect.top) / height - 0.5;
        }});

        function animate() {{
            requestAnimationFrame(animate);
            group.rotation.y += 0.0035;
            camera.position.x += (mouseX * 3 - camera.position.x) * 0.02;
            camera.position.y += (1.1 - mouseY * 2 - camera.position.y) * 0.02;
            camera.lookAt(0, 0, 0);
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            const w = container.clientWidth;
            camera.aspect = w / height;
            camera.updateProjectionMatrix();
            renderer.setSize(w, height);
        }});
    }})();
    </script>
    """
    components.html(carousel_html, height=440)

st.subheader("🎡 Spin the 3D Poster Wall")
st.caption("Move your mouse over it — it's a live rotating 3D scene, not an image.")
render_3d_carousel(rec.top_rated(10))
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------- #
# Trending row — streaming-platform style, shown before the user does anything
# ---------------------------------------------------------------------- #
def render_scroll_row(df: pd.DataFrame):
    row_html = '<div class="scroll-row">'
    for _, row in df.iterrows():
        row_html += html(f"""
        <div class="mini-card tilt-hover">
            <div class="mini-poster" style="background:{gradient_for(row['title'])}">{emoji_for(row['genres'])}</div>
            <div class="mini-body">
                <div class="mini-title">{row['title']}</div>
                <div class="mini-meta">⭐ {row['rating']} · {row['year']}</div>
            </div>
        </div>
        """)
    row_html += "</div>"
    st.markdown(row_html, unsafe_allow_html=True)

st.subheader("🔥 Trending on CineMatch")
render_scroll_row(rec.top_rated(10))
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------- #
# Sidebar — mode selection
# ---------------------------------------------------------------------- #
st.sidebar.header("🎯 How do you want recommendations?")
mode = st.sidebar.radio(
    "Mode",
    ["🎥 By a movie I like", "🎲 Quick Taste Quiz", "🎯 By my preferences"],
    label_visibility="collapsed",
)
top_n = st.sidebar.slider("Number of recommendations", min_value=3, max_value=15, value=8)
st.sidebar.markdown("---")

with st.sidebar.expander("🧠 How the engine thinks", expanded=False):
    st.latex(r"\text{sim}(A,B)=\frac{A \cdot B}{\|A\|\,\|B\|}")
    st.caption(
        "Every movie's genres, director, cast, keywords and overview are combined into "
        "one text profile and converted into a TF-IDF vector. Cosine similarity then "
        "measures the angle between two vectors — the smaller the angle, the more similar "
        "the movies."
    )
    st.metric("Vocabulary size", stats["vocab_size"])
    st.metric("Matrix sparsity", f"{stats['sparsity_pct']}%")

# ---------------------------------------------------------------------- #
def render_why_chips(explanation: dict):
    chips = ""
    for g in explanation.get("genres", []):
        chips += f'<span class="why-chip">🎭 {g}</span>'
    for k in explanation.get("keywords", []):
        chips += f'<span class="why-chip">🔑 {k}</span>'
    if explanation.get("director"):
        chips += f'<span class="why-chip">🎬 dir. {explanation["director"]}</span>'
    return chips or '<span class="why-chip">Overall thematic similarity</span>'

def render_results(results: pd.DataFrame, explain_fn=None):
    if results.empty:
        st.warning("No recommendations found — try different inputs.")
        return
    cols = st.columns(2)
    for i, (_, row) in enumerate(results.iterrows()):
        chips_html = render_why_chips(explain_fn(row["title"])) if explain_fn else ""
        overview = row.get("overview", "") if hasattr(row, "get") else row["overview"]
        with cols[i % 2]:
            st.markdown(
                html(f"""
                <div class="flip-card">
                    <div class="flip-inner">
                        <div class="flip-front">
                            <div class="poster" style="background:{gradient_for(row['title'])}">{emoji_for(row['genres'])}</div>
                            <div class="card-body">
                                <div class="movie-title">{row['title']} ({row['year']})</div>
                                <div class="movie-meta">{row['genres']} · dir. {row['director']} · ⭐ {row['rating']}</div>
                                <div class="match-track"><div class="match-fill" style="width:{row['similarity']}%"></div></div>
                                <div style="color:#00e0a1;font-size:0.78rem;font-weight:700;">{row['similarity']}% match</div>
                                <div class="flip-hint">↻ hover to see why</div>
                            </div>
                        </div>
                        <div class="flip-back">
                            <div class="back-title">{row['title']}</div>
                            <div class="back-overview">{overview}</div>
                            {chips_html}
                        </div>
                    </div>
                </div>
                """),
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------- #
# Mode 1: single favorite movie
# ---------------------------------------------------------------------- #
if mode == "🎥 By a movie I like":
    st.subheader("Tell us a movie you enjoyed")
    title = st.selectbox("Pick a movie", sorted(rec.all_titles()))
    if st.button("✨ Get Recommendations", type="primary"):
        results = rec.recommend_by_movie(title, top_n=top_n)
        st.subheader(f"Because you liked *{title}*")
        render_results(results, explain_fn=lambda rt: rec.explain_movie_match(title, rt))

# ---------------------------------------------------------------------- #
# Mode 2: Quick taste quiz — like/dislike a shuffled set of movies
# ---------------------------------------------------------------------- #
elif mode == "🎲 Quick Taste Quiz":
    st.subheader("Like a few movies below to build your Taste DNA 🧬")

    if "quiz_sample" not in st.session_state:
        st.session_state.quiz_sample = rec.random_sample(6, seed=random.randint(0, 9999))

    if st.button("🔀 Shuffle picks"):
        st.session_state.quiz_sample = rec.random_sample(6, seed=random.randint(0, 9999))

    liked = []
    cols = st.columns(3)
    for i, (_, row) in enumerate(st.session_state.quiz_sample.iterrows()):
        with cols[i % 3]:
            st.markdown(
                html(f"""
                <div class="movie-card tilt-hover">
                    <div class="poster" style="background:{gradient_for(row['title'])}">{emoji_for(row['genres'])}</div>
                    <div class="card-body">
                        <div class="movie-title">{row['title']}</div>
                        <div class="movie-meta">{row['genres']} · ⭐ {row['rating']}</div>
                    </div>
                </div>
                """),
                unsafe_allow_html=True,
            )
            if st.checkbox("❤️ Like this", key=f"quiz_{row['title']}"):
                liked.append(row["title"])

    if st.button("🧬 Reveal My Taste DNA", type="primary"):
        if not liked:
            st.warning("Like at least one movie first.")
        else:
            results = rec.recommend_by_multiple_movies(liked, top_n=top_n)
            st.subheader(f"Built from your taste for: {', '.join(liked)}")
            render_results(results, explain_fn=lambda rt: rec.explain_movie_match(liked[0], rt))

# ---------------------------------------------------------------------- #
# Mode 3: explicit preferences (genres / people / keywords)
# ---------------------------------------------------------------------- #
else:
    st.subheader("Tell us what you're in the mood for")
    col1, col2 = st.columns(2)
    with col1:
        genres = st.multiselect("Favorite genres", rec.all_genres())
        people = st.text_input("Favorite director or actor (optional)", placeholder="e.g. Christopher Nolan")
    with col2:
        keywords = st.text_input("Themes / keywords (comma-separated, optional)", placeholder="e.g. heist, revenge")

    if st.button("✨ Get Recommendations", type="primary"):
        people_list = [p.strip() for p in people.split(",") if p.strip()]
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()]
        if not genres and not people_list and not keyword_list:
            st.warning("Give us at least a genre, person, or keyword to work with.")
        else:
            results = rec.recommend_by_preferences(
                genres=genres, people=people_list, keywords=keyword_list, top_n=top_n
            )
            st.subheader("Matched to your preferences")
            render_results(
                results,
                explain_fn=lambda rt: rec.explain_preference_match(genres, keyword_list, rt),
            )

# ---------------------------------------------------------------------- #
# Insights panel — genre distribution across the catalog
# ---------------------------------------------------------------------- #
st.markdown("---")
with st.expander("📊 Dataset Insights"):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.caption("Genre distribution across the catalog")
        st.bar_chart(rec.genre_distribution())
    with c2:
        st.metric("Total movies", stats["n_movies"])
        st.metric("Vocabulary size", stats["vocab_size"])
        st.metric("Matrix density", f"{stats['matrix_density_pct']}%")

with st.expander("🎞️ Browse the full movie catalog"):
    st.dataframe(
        rec.df[["title", "year", "genres", "director", "cast", "rating"]],
        use_container_width=True,
        hide_index=True,
    )
