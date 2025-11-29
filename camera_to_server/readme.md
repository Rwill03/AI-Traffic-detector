ssh radxa@192.168.0.160
cd ~/traffic_counter
git checkout futuristic-dashboard
git pull
cd camera_to_server
docker build -t traffic-counter-rock5 .   # alleen na code changes
docker run --rm -it --device=/dev/video1:/dev/video1 --network=host traffic-counter-rock5
