cat > README.md <<'EOF'
# Optivion

This repository contains the Optivion Streamlit app — interactive analog & light-based simulations and ML visualization.

## Quick setup (local)

\`\`\`bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\\Scripts\\activate    # Windows (PowerShell)
pip install -r requirements.txt
streamlit run app.py
\`\`\`

## Deploy to Streamlit Community Cloud
1. Push repo to GitHub (see commands below).
2. Go to https://share.streamlit.io, sign in with GitHub, create a new app using \`main\` branch and \`app.py\` as the entrypoint. Streamlit will install dependencies from \`requirements.txt\`.

## Deploy to Render.com
1. Create a Render Web Service connected to this repository.
2. Use the above \`Procfile\` or set the start command to:
\`\`\`
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
\`\`\`

## Useful git commands
\`\`\`bash
git init
git add .
git commit -m "Initial commit - Optivion"
gh repo create optivion --public --source=. --remote=origin --push  # optional if you have GitHub CLI
# OR
# create repo on GitHub web, then:
# git remote add origin https://github.com/<your-username>/optivion.git
# git branch -M main
# git push -u origin main
\`\`\`

## Notes
- If deployment fails due to missing packages, pin versions in \`requirements.txt\`.
- Add secrets via Streamlit Cloud app settings if your app needs them.
EOF