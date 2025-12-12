ssh root@100.89.11.82
cd ~/AI-Traffic-detector
git checkout futuristic-dashboard
git pull
docker compose up -d --build


interface:

http://100.89.11.82:8001/static/dashboard.html
