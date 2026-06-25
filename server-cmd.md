Full CLI reference (SSH in first)

ssh -i ~/Downloads/acb_processor.pem ubuntu@98.86.63.69
Service auto-start (systemd)

sudo systemctl status forensic-audio      # is it enabled/running
sudo systemctl start forensic-audio       # start (if stopped)
sudo systemctl stop forensic-audio        # stop all containers
sudo systemctl restart forensic-audio     # restart everything
sudo systemctl enable forensic-audio      # enable auto-start on boot
sudo systemctl disable forensic-audio     # disable auto-start
Docker containers

cd /app

docker ps                                  # running containers
docker ps -a                               # all including stopped
docker compose up -d                       # start all
docker compose down                        # stop all
docker compose restart                     # restart all
docker compose restart worker-gpu          # restart specific service
docker compose logs -f worker-gpu          # live logs GPU worker
docker compose logs -f api                 # live logs API
docker compose logs --tail=50 api          # last 50 lines
docker stats                               # live CPU/RAM/GPU usage
Deploy latest code from git

cd /app
git pull origin IndicConformer             # pull latest
docker compose build --no-cache           # rebuild images
docker compose down && docker compose up -d  # redeploy
Queue / jobs

docker compose exec redis redis-cli FLUSHDB    # clear all queued jobs
docker compose exec redis redis-cli DBSIZE     # jobs in queue
curl -s http://localhost:5555                  # Flower dashboard (Celery monitor)
API test

# Health check
curl -s -o /dev/null -w '%{http_code}' \
  -H 'X-API-Key: f379241418da1092837aaa6b7138e850e4b99b1b2f88ba90d527e6a0b4b4a600' \
  http://localhost/cases

# Check job status
curl -s -H 'X-API-Key: f379241418da1092837aaa6b7138e850e4b99b1b2f88ba90d527e6a0b4b4a600' \
  http://localhost/jobs/<JOB_ID>

# Get result
curl -s -H 'X-API-Key: f379241418da1092837aaa6b7138e850e4b99b1b2f88ba90d527e6a0b4b4a600' \
  http://localhost/jobs/<JOB_ID>/result
Disk / GPU

df -h                                      # disk usage
nvidia-smi                                 # GPU VRAM usage
docker system df                           # docker disk usage
docker system prune -af --volumes          # nuke all unused images/volumes
Database (Postgres)

docker compose exec postgres psql -U pipeline -d forensic   # psql shell
# inside psql:
\dt                                        # list tables
SELECT id, status, stage FROM jobs ORDER BY created_at DESC LIMIT 10;
\q                                         # quit