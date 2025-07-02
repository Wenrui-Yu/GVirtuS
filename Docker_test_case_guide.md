## 📦 Installation

```bash
# Clone the GVirtuS main repository:
git clone https://github.com/ecn-aau/GVirtuS

# Change into the repo directory:
cd GVirtuS
```

# 🧪 Docker-Based Testcase Development with GVirtuS

This document outlines the **step-by-step process** for setting up and managing a Docker-based development and testing workflow for GVirtuS, with best practices for sharing and maintaining your environment via Docker Hub.

---

## ✅ 1. Create a Dockerfile and Push to Docker Hub

First, build a Docker image that includes all dependencies needed for compiling and testing GVirtuS.

### 📄 Sample Dockerfile (save as `docker/dev/Dockerfile`):

```Dockerfile
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl wget unzip nano vim pkg-config \
    python3 python3-pip ca-certificates libgtest-dev libprotobuf-dev librdmacm-dev \
    libibverbs-dev libmesa-dev \
    protobuf-compiler libgoogle-glog-dev libgflags-dev libssl-dev && \
    rm -rf /var/lib/apt/lists/*

RUN cd /usr/src/gtest && cmake CMakeLists.txt && make && cp lib/*.a /usr/lib

ENV GVIRTUS_HOME=/usr/local/gvirtus
ENV GVIRTUS_LOGLEVEL=0
ENV LD_LIBRARY_PATH=${GVIRTUS_HOME}/lib:${LD_LIBRARY_PATH}
```

### 🚀 Build and Push:

```bash
docker buildx build \
  --platform linux/amd64 \
  --push \
  --no-cache \
  -f docker/dev/Dockerfile \
  -t yourusername/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04 \
  .
```

This pushes the image to Docker Hub under your account.

---

## ✅ 2. Keep the Dockerfile Locally in `docker/dev/`

Even after pushing the image, always **keep the Dockerfile versioned in your repo**.

### ❓Why keep it locally?

* 🔁 **Reproducibility** – Anyone cloning your repo can rebuild the image exactly.
* ⚙️ **Modifiability** – You or your teammates can easily update dependencies.
* 📖 **Transparency** – View what’s installed without pulling or inspecting the image.
* 🚧 **Disaster Recovery** – If Docker Hub image is deleted or corrupted, you can rebuild it.

### ❌ What happens if you *don’t* keep it?

* You lose the ability to rebuild or modify the image confidently.
* Your environment becomes a "black box."
* Collaboration or CI/CD automation becomes much harder.

### ✅ Advantages of Docker Hub

* ⚡ Fast startup – avoids building dependencies every time.
* 🌐 Shareable – others can pull and use your environment instantly.
* 📦 Portable – use the same setup on other machines or in CI/CD pipelines.

---

## ✅ 3. Create a Makefile

This simplifies running your environment and tests with one-liners.

### 📄 Example `Makefile`:

```makefile
run-gvirtus-backend-dev:
	docker run \
		--rm \
		-it \
		--gpus all \
		-v ./cmake:/gvirtus/cmake/ \
		-v ./etc:/gvirtus/etc/ \
		-v ./include:/gvirtus/include/ \
		-v ./plugins:/gvirtus/plugins/ \
		-v ./src:/gvirtus/src/ \
		-v ./tools:/gvirtus/tools/ \
		-v ./tests:/gvirtus/tests/ \
		-v ./CMakeLists.txt:/gvirtus/CMakeLists.txt \
		-v ./docker/dev/entrypoint.sh:/entrypoint.sh \
		--entrypoint /entrypoint.sh \
		--name gvirtus \
		yourusername/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04

docker-build-push-dev:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/dev/Dockerfile \
		-t yourusername/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04 \
		.

run-gvirtus-tests:
	docker exec -it gvirtus bash -c "cd /gvirtus/build && ctest --output-on-failure"

stop-gvirtus:
	docker stop gvirtus || true

```

---

## ✅ 4. What Happens When You Run the Makefile?

### ▶️ `make run-gvirtus-backend-dev`:

* Pulls and runs the Docker image from Docker Hub.
* Mounts your local GVirtuS source folders into the container.
* Uses `/entrypoint.sh` to configure, build, and start a shell inside the build environment.
* Keeps your workflow **isolated and reproducible**.

### ▶️ `make docker-build-push-dev`:

* Rebuilds the Docker image from your **local Dockerfile**.
* Skips cached layers to ensure a full rebuild.
* Pushes the new image version to Docker Hub for shared usage.

---



## 🧠 Summary

| Step                 | Purpose                                                   |
| -------------------- | --------------------------------------------------------- |
| **Dockerfile**       | Blueprint for the development environment                 |
| **Docker Hub Image** | Portable, sharable image used by Makefile                 |
| **Makefile**         | Simple interface to run and manage dev/test               |
| **Keep Dockerfile**  | So you can rebuild, version, share, and recover the image |


| Clone GVirtuS | `git clone https://github.com/tgasla/GVirtuS.git` |
| Build and run dev container | `make run-gvirtus-backend-dev` |
| Run CUDA test cases | `make run-gvirtus-tests` |
| Stop the container | `make stop-gvirtus` |
| Add More testing functions | Re-run the dev container, tests and lastly stop it |
| If make any changes localy at Dockerfile Rebuild and push image | `make docker-build-push-dev` |

docker/dev/Dockerfile = The recipe
Docker Hub image       = The cooked dish
Makefile               = The waiter serving it


+---------------------+        Mount source code         +---------------------+
|  Local Machine       |  <---------------------------->  |  Docker Container   |
|  (Your GVirtuS repo) |                                   |  (GVirtuS Backend)  |
|                      |                                   |                     |
|  - GVirtuS source    |                                   |  - CUDA 12 & cuDNN  |
|    code (local files)|                                   |  - All dependencies |
|  - Tests (.cu files) |                                   |  - Mounted source   |
+---------------------+                                   +---------------------+
         |                                                        ^
         |                                                        |
         |                                                        |
         |                  Build & Run Docker Image              |
         |------------------------------------------------------->|
         |                                                        |
         |               Run backend & tests inside container    |
         |<-------------------------------------------------------|
         |                                                        |
         |                   Run test commands (ctest)           |
         |------------------------------------------------------->|
         |                                                        |
         |                Receive test results/logs               |
         |<-------------------------------------------------------|


---


