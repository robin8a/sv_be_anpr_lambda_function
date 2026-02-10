# Install Docker for `sv_be_anpr_lambda_function`

This project builds and pushes a Docker image to AWS ECR and then deploys it to AWS Lambda.  
You need Docker installed **locally** before running the commands in `DEPLOYMENT.md`.

If you already have Docker installed and `docker run hello-world` works, you can skip this file.

---

## 1. Linux (Ubuntu / Debian)

The commands below follow the official Docker Engine installation steps for Ubuntu/Debian-based systems.

```sh
# 1) Remove any old Docker versions (safe if none are installed)
sudo apt-get remove docker docker-engine docker.io containerd runc || true

# 2) Install dependencies
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

# 3) Add Docker’s official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/$(. /etc/os-release && echo "$ID")/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 4) Add the Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$(. /etc/os-release && echo "$ID") \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 5) Install Docker Engine, CLI, containerd, and plugins
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 6) (Optional) Allow your user to run Docker without sudo
sudo groupadd docker || true
sudo usermod -aG docker "$USER"

# You may need to log out and back in, or run:
newgrp docker

# 7) Verify installation
docker run --rm hello-world
```

If the final command prints a "Hello from Docker!" message, Docker is installed correctly.

---

## 2. macOS

On macOS you normally use **Docker Desktop**.

1. Go to the Docker Desktop download page:  
   `https://www.docker.com/products/docker-desktop/`
2. Download **Docker Desktop for Mac** (Intel or Apple Silicon, matching your machine).
3. Install it by dragging the Docker app into `Applications`.
4. Launch Docker Desktop and wait until it reports that Docker is running (whale icon in the menu bar).
5. Open **Terminal** and verify:

```sh
docker run --rm hello-world
```

If you see the "Hello from Docker!" message, you are ready to build images for this project.

---

## 3. Windows 10 / 11

On Windows you also use **Docker Desktop**, normally with the WSL2 backend.

1. Go to:  
   `https://www.docker.com/products/docker-desktop/`
2. Download **Docker Desktop for Windows**.
3. Run the installer and:
   - Accept using **WSL2 based engine** (recommended).
   - Ensure **"Use WSL 2 instead of Hyper-V"** is checked if available.
4. After installation, start Docker Desktop and wait until it reports that Docker is running.
5. Open **PowerShell** or **Windows Terminal** and verify:

```powershell
docker run --rm hello-world
```

If the test container runs successfully, Docker is installed correctly.

---

## 4. Next steps

Once Docker is installed and verified:

1. Go back to `DEPLOYMENT.md`.
2. Follow the **Docker build and push to ECR** section.
3. Then continue with the **Create / Update Lambda function** steps.

