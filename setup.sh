mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"patelsapan530@example.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
" > ~/.streamlit/config.toml
