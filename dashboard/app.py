import streamlit as st, json, os
st.title('Crypto AI Hybrid v13 Dashboard - Placeholder')
cfg = json.load(open('config.json'))
st.write('Symbols:', cfg.get('symbols',[]))
