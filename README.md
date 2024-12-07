# LLMs
Learning LLM

https://github.com/DataTalksClub/llm-zoomcamp

# Check if docker is installed
``` which docker ```

# If installed, remove it
``` sudo apt-get remove docker docker-engine docker.io containerd runc ```

# Set up the repository
```
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg lsb-release
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

# Install Docker CE
```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

# verify that Docker CE is installed correctly by running the hello-world image
``` sudo docker run hello-world```

# to use docker without sudo
```
sudo usermod -aG docker $USER
newgrp docker
```

# Elastic search

# Pull the image
``` docker pull docker.elastic.co/elasticsearch/elasticsearch:8.12.0 ```

# Run the image
```
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.12.0
```
