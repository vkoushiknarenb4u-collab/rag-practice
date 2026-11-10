"""Streamlit storefront chat for the existing ShopRAG pipeline."""

import streamlit as st

from rag_pipeline import RAGPipelineError, answer_question


WELCOME_MESSAGE = "Hi! I'm your Shop AI Assistant. Ask me anything about our products."

QUICK_PROMPTS = [
    ("♧", "Find Outfits", "Style picks from the catalogue", "Which products are available under ₹1000?"),
    ("◇", "Best Offers", "Promotions and current deals", "What offers are available?"),
    ("▱", "Product Help", "Sizes, colours, and materials", "What T-shirts do you have?"),
    ("↗", "Store Policies", "Shipping, returns, and refunds", "What is your return policy?"),
]


def add_styles():
    """Set the dark editorial look and orange streetwear accents."""
    st.markdown(
        """
        <style>
        :root {
            --orange: #ff6b0b;
            --orange-bright: #ff790e;
            --ink: #0a0b0d;
            --panel: #141518;
            --muted: #a8a6a2;
            --paper: #f3eee6;
        }

        [data-testid="stAppViewContainer"] {
            color: var(--paper);
            background:
                radial-gradient(ellipse at 73% 4%, rgba(255, 98, 0, 0.12), transparent 31%),
                radial-gradient(ellipse at 84% 95%, rgba(255, 98, 0, 0.07), transparent 27%),
                repeating-linear-gradient(90deg, rgba(255,255,255,.012) 0 1px, transparent 1px 5px),
                #0a0b0d;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1500px;
            padding: 1.25rem 2.5rem 6.5rem;
        }

        [data-testid="stSidebar"] {
            min-width: 285px;
            max-width: 310px;
            background:
                radial-gradient(ellipse at 15% 0%, rgba(255, 107, 11, .08), transparent 40%),
                #101113;
            border-right: 1px solid rgba(255,255,255,.08);
        }

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 1.6rem 1.15rem 1.2rem;
        }

        .brand-lockup {
            margin: .4rem .2rem 1.6rem;
        }

        .brand-lockup .threadline {
            color: var(--orange);
            font-size: .76rem;
            font-weight: 900;
            letter-spacing: .24em;
        }

        .brand-lockup .streetwear {
            color: var(--paper);
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 2rem;
            font-style: italic;
            font-weight: 700;
            letter-spacing: -.07em;
            line-height: 1.05;
        }

        .brand-lockup .assistant-label {
            color: #969493;
            font-size: .9rem;
            margin-top: .5rem;
        }

        .side-section-title {
            color: var(--paper);
            font-size: .9rem;
            font-weight: 750;
            margin: 1.3rem 0 .7rem;
        }

        .active-nav {
            align-items: center;
            background: linear-gradient(100deg, rgba(255,107,11,.2), rgba(255,107,11,.05));
            border: 1px solid rgba(255,107,11,.78);
            border-radius: 12px;
            box-shadow: 0 0 22px rgba(255,107,11,.09), inset 0 0 16px rgba(255,107,11,.04);
            color: #ffd4b2;
            display: flex;
            font-weight: 700;
            gap: .7rem;
            margin: .2rem 0 1.25rem;
            padding: .72rem .85rem;
        }

        .active-nav .nav-icon {
            color: var(--orange-bright);
            font-size: 1.1rem;
        }

        .category-pills {
            display: flex;
            flex-wrap: wrap;
            gap: .48rem;
        }

        .category-pill {
            background: linear-gradient(140deg, #1b1c1f, #141517);
            border: 1px solid rgba(255,255,255,.1);
            border-radius: 999px;
            color: #dfddda;
            font-size: .79rem;
            padding: .45rem .72rem;
            white-space: nowrap;
        }

        .hero {
            background:
                radial-gradient(circle at 75% 48%, rgba(255, 107, 11, .1), transparent 30%),
                repeating-linear-gradient(0deg, transparent 0 28px, rgba(255,255,255,.025) 29px 30px),
                repeating-linear-gradient(90deg, transparent 0 28px, rgba(255,255,255,.025) 29px 30px),
                radial-gradient(ellipse at 22% 100%, rgba(255, 107, 11, .12), transparent 38%),
                linear-gradient(115deg, #171615, #0e0f11 66%);
            border-bottom: 1px solid rgba(255,255,255,.08);
            height: 345px;
            margin: -.25rem 0 .8rem;
            overflow: hidden;
            position: relative;
        }

        .hero::before {
            background: linear-gradient(135deg, transparent 46%, rgba(255,107,11,.45) 46.2%, transparent 46.7%);
            bottom: -28%;
            content: '';
            height: 65%;
            left: 0;
            opacity: .55;
            pointer-events: none;
            position: absolute;
            transform: rotate(-7deg);
            width: 56%;
        }

        .hero-copy {
            left: 4%;
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            width: 54%;
            z-index: 3;
        }

        .hero-kicker {
            color: #b9b5b0;
            font-size: .83rem;
            font-weight: 700;
            letter-spacing: .22em;
            text-transform: uppercase;
        }

        .hero-kicker span {
            color: var(--orange);
        }

        .hero-title {
            color: var(--paper);
            font-size: clamp(3.2rem, 6.4vw, 6.5rem);
            font-weight: 900;
            letter-spacing: -.075em;
            line-height: .98;
            margin: 1.25rem 0 .85rem;
            white-space: nowrap;
        }

        .hero-title em {
            color: var(--orange-bright);
            display: inline-block;
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 1.06em;
            font-weight: 700;
            letter-spacing: -.08em;
            transform: rotate(-7deg);
        }

        .hero-subtitle {
            color: #aaa7a2;
            font-size: 1.03rem;
            margin: 0;
        }

        .hero-art {
            height: 100%;
            position: absolute;
            right: 0;
            top: 0;
            width: 53%;
            z-index: 1;
        }

        .hero-art svg {
            height: 100%;
            width: 100%;
        }

        .hero-sticker {
            background: #e85e08;
            bottom: 1.3rem;
            color: #1a120c;
            font-family: 'Courier New', monospace;
            font-size: .68rem;
            font-weight: 800;
            padding: .45rem .6rem;
            position: absolute;
            right: 2.3rem;
            transform: rotate(-7deg);
            z-index: 2;
        }

        .welcome-message {
            align-items: center;
            background: linear-gradient(100deg, #1b1b1a, #111214);
            border: 1px solid rgba(255,107,11,.72);
            border-radius: 17px;
            box-shadow: 0 0 26px rgba(255,107,11,.08);
            color: #f2eee8;
            display: flex;
            gap: .9rem;
            margin: .8rem .1rem 1.7rem;
            padding: .95rem 1.15rem;
        }

        .welcome-badge {
            align-items: center;
            background: var(--orange);
            border-radius: 10px;
            color: #211106;
            display: flex;
            flex: 0 0 38px;
            font-size: 1.15rem;
            height: 38px;
            justify-content: center;
        }

        [data-testid="stHorizontalBlock"] {
            gap: .9rem;
        }

        .stButton > button {
            background: linear-gradient(145deg, #18191b, #101113);
            border: 1px solid rgba(255,255,255,.1);
            border-radius: 17px;
            color: #eeeae4;
            min-height: 132px;
            padding: .8rem .75rem;
            transition: border-color .16s ease, transform .16s ease, box-shadow .16s ease;
            white-space: pre-wrap;
        }

        .stButton > button:hover {
            border-color: rgba(255,107,11,.72);
            box-shadow: 0 8px 28px rgba(255,107,11,.1);
            color: #fff;
            transform: translateY(-2px);
        }

        .stButton > button:focus:not(:active) {
            border-color: var(--orange);
            color: #fff;
        }

        [data-testid="stChatInput"] textarea {
            background: #141517;
            border: 1px solid rgba(255,107,11,.7);
            border-radius: 999px;
            box-shadow: 0 0 22px rgba(255,107,11,.08);
            color: var(--paper);
            min-height: 58px;
            padding: .9rem 1.25rem;
        }

        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--orange-bright);
            box-shadow: 0 0 0 1px var(--orange-bright), 0 0 25px rgba(255,107,11,.12);
        }

        [data-testid="stChatMessage"] {
            background: rgba(21,22,24,.92);
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 16px;
            margin-bottom: .7rem;
            padding: .75rem 1rem;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.65;
        }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] { padding: 1rem 1rem 5.5rem; }
            .hero { height: 320px; }
            .hero-copy { left: 5%; width: 62%; }
            .hero-art { opacity: .78; width: 54%; }
            .hero-title { font-size: clamp(2.8rem, 8vw, 4.7rem); }
        }

        @media (max-width: 620px) {
            .hero { height: 285px; }
            .hero-copy { left: 6%; width: 75%; }
            .hero-art { opacity: .42; width: 63%; }
            .hero-title { font-size: clamp(2.7rem, 11vw, 4rem); }
            .hero-kicker { font-size: .68rem; letter-spacing: .16em; }
            .hero-subtitle { font-size: .88rem; max-width: 240px; }
            .hero-sticker { right: 1rem; }
            .welcome-message { font-size: .9rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_sidebar():
    """Render the compact brand, chat navigation, and category list."""
    with st.sidebar:
        st.markdown(
            """
            <div class="brand-lockup">
                <div class="threadline">THREADLINE</div>
                <div class="streetwear">Streetwear</div>
                <div class="assistant-label">AI Shop Assistant</div>
            </div>
            <div class="active-nav"><span class="nav-icon">▣</span> Chat</div>
            <div class="side-section-title">Categories</div>
            <div class="category-pills">
                <span class="category-pill">T-shirts</span>
                <span class="category-pill">Jackets</span>
                <span class="category-pill">Hoodies</span>
                <span class="category-pill">Footwear</span>
                <span class="category-pill">Accessories</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        if st.button("＋   New chat", use_container_width=True):
            st.session_state.messages = [
                {"role": "assistant", "content": WELCOME_MESSAGE}
            ]
            st.rerun()


def show_hero():
    """Render the editorial hero and a code-drawn streetwear illustration."""
    st.markdown(
        """
        <section class="hero">
            <div class="hero-copy">
                <div class="hero-kicker"><span>THREADLINE</span> / AI SHOP ASSISTANT</div>
                <h1 class="hero-title">Find your <em>fit.</em></h1>
                <p class="hero-subtitle">Product help, offers, and store policies. Ask away.</p>
            </div>
            <div class="hero-art" aria-hidden="true">
                <svg viewBox="0 0 620 380" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <radialGradient id="halo" cx="50%" cy="42%" r="60%">
                            <stop offset="0" stop-color="#ff7a0a"/>
                            <stop offset=".78" stop-color="#ee5b00"/>
                            <stop offset="1" stop-color="#aa3500"/>
                        </radialGradient>
                        <linearGradient id="jacket" x1="0" y1="0" x2="1" y2="1">
                            <stop offset="0" stop-color="#333536"/>
                            <stop offset=".5" stop-color="#111315"/>
                            <stop offset="1" stop-color="#050607"/>
                        </linearGradient>
                        <linearGradient id="paper" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0" stop-color="#c8c2b8"/>
                            <stop offset="1" stop-color="#827d75"/>
                        </linearGradient>
                    </defs>
                    <circle cx="330" cy="170" r="139" fill="url(#halo)" opacity=".96"/>
                    <g opacity=".46" stroke="#d0cbc2" stroke-width="3" fill="none">
                        <path d="M380 72h205M397 72v205M580 72v205M390 277h198"/>
                        <path d="M425 72l18 18 18-18M475 72l18 18 18-18M525 72l18 18 18-18"/>
                    </g>
                    <g opacity=".73">
                        <path d="M420 103h54l-27 23zM476 103h52l-26 22zM531 103h48l-24 21z" fill="none" stroke="#c4c0b8" stroke-width="3"/>
                        <path d="M428 128h43l11 116h-65zM483 127h43l13 125h-69zM535 126h40l14 115h-65z" fill="#17191b" stroke="#aaa59c" stroke-width="2"/>
                        <path d="M438 140l13 20-7 77M495 140l12 20-7 82M546 139l11 18-7 80" stroke="#76736e" stroke-width="2" fill="none"/>
                    </g>
                    <path d="M256 79c-11-30 12-53 43-51 30 2 44 22 39 49l-8 37-52 5z" fill="#08090a"/>
                    <ellipse cx="301" cy="97" rx="34" ry="43" fill="#17191a"/>
                    <path d="M270 88c8-36 49-45 69-15-10-3-18-10-24-18-4 16-22 24-45 33z" fill="#050607"/>
                    <path d="M282 102c10 8 23 10 34 6" stroke="#6d6c68" stroke-width="2" fill="none"/>
                    <path d="M279 139l-29 21-44 31-23 139h265l-18-141-46-31-25-18z" fill="url(#jacket)"/>
                    <path d="M279 140l20 31 22-31 40 22-19 122h-96l-11-122z" fill="#202224" stroke="#4b4c4b" stroke-width="2"/>
                    <path d="M299 171l-9 109M321 171l14 109M252 163l-37 37M361 163l39 36" stroke="#65645f" stroke-width="2" opacity=".7"/>
                    <path d="M305 167v109" stroke="#959087" stroke-width="2" opacity=".65"/>
                    <text x="342" y="218" fill="#ff6b0b" font-size="20" font-weight="900" font-family="Arial, sans-serif">TL.</text>
                    <path d="M154 328l28-25 60 11 45-9 36 12 45-8 41 16-5 25H158z" fill="url(#paper)" opacity=".88"/>
                    <path d="M165 336l94-13 53 9 87-8" stroke="#4a4540" stroke-width="2" fill="none" opacity=".7"/>
                    <path d="M438 314h111v34H438z" fill="#e9e3d8" transform="rotate(-7 438 314)"/>
                    <text x="450" y="336" fill="#171717" font-size="13" font-family="Courier New, monospace" font-weight="bold" transform="rotate(-7 450 336)">TL / DROP 01</text>
                </svg>
            </div>
            <div class="hero-sticker">THREADLINE / 001</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def ask_shop(question):
    """Send a question through the existing backend and save the reply."""
    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🟠"):
        try:
            with st.spinner("Searching the shop catalogue..."):
                result = answer_question(question)
            answer = result["answer"]
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
        except RAGPipelineError as error:
            error_message = f"I couldn't complete that request: {error}"
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "is_error": True}
            )
        except Exception:
            error_message = "Something went wrong. Please try again in a moment."
            st.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message, "is_error": True}
            )


def main():
    st.set_page_config(
        page_title="Shop AI | Threadline",
        page_icon="🟠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    add_styles()

    # Start each new browser session with the requested welcome message.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]

    show_sidebar()
    show_hero()

    # Use a distinctive welcome card before the customer starts chatting.
    if (
        len(st.session_state.messages) == 1
        and st.session_state.messages[0]["role"] == "assistant"
        and st.session_state.messages[0]["content"] == WELCOME_MESSAGE
    ):
        st.markdown(
            f'<div class="welcome-message"><span class="welcome-badge">✦</span>'
            f'<span>{WELCOME_MESSAGE}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        # Keep later customer and assistant messages in chat bubbles.
        for message in st.session_state.messages:
            avatar = "🟠" if message["role"] == "assistant" else None
            with st.chat_message(message["role"], avatar=avatar):
                if message.get("is_error"):
                    st.error(message["content"])
                else:
                    st.markdown(message["content"])

    selected_prompt = None
    if len(st.session_state.messages) == 1:
        columns = st.columns(4)
        for index, (icon, title, description, prompt) in enumerate(QUICK_PROMPTS):
            with columns[index]:
                label = f"{icon}   {title}\n\n{description}"
                if st.button(
                    label,
                    key=f"quick_prompt_{index}",
                    help=description,
                    use_container_width=True,
                ):
                    selected_prompt = prompt

    customer_question = st.chat_input("Message Shop AI...")
    question = selected_prompt or customer_question
    if question:
        ask_shop(question)
        # Rerender to hide the quick-start cards once the conversation begins.
        st.rerun()


if __name__ == "__main__":
    main()
