# 📄 NVIDIA Driver & CUDA Compatibility for GVirtuS (Docker-Based)

## ✅ Summary

This document outlines compatibility between CUDA 12.6 Docker images, host NVIDIA drivers, and Ubuntu versions — particularly in the context of running **GVirtuS** within a Docker environment.

---

## 🔧 Key Compatibility Observations

### 1. **CUDA 12.6 Docker Image on NVIDIA Driver 570** (or 560+ onwards as mentioned as in main Readme file)

* ✅ Works **perfectly** on hosts with **NVIDIA driver 570**.
* ✅ Even if the host has a **different CUDA version installed (e.g., 11.5)**, the Docker container will work fine.
* ✅ This setup is suitable for running GVirtuS with modern CUDA workloads.

> 📝 **Why it works:** NVIDIA driver 570 supports CUDA 12.6 at the kernel level, which is sufficient for containerized workloads.

---

### 2. **CUDA 12.6 Docker Image on NVIDIA Driver 535**

* ❌ **Fails to run**, even if the host has **CUDA 12.2 installed**.
* ❌ `nvidia-container-cli` will throw a compatibility error:
  `unsatisfied condition: cuda>=12.6`

> 🛑 **Why it fails:** NVIDIA driver 535 **does not support CUDA 12.6**, and containers require driver support at runtime.

---

### 3. **Ubuntu 20.04 LTS Driver Support**

* ✅ Fully supports **NVIDIA driver 535**.
* ❌ Does **not officially support NVIDIA driver 570** — manual installation may break DKMS, kernel modules, or require kernel upgrades.

> ⚠️ **Recommendation:** Avoid using NVIDIA 570 with Ubuntu 20.04 in production.

---

### 4. **Ubuntu 22.04 LTS Driver Support**

* ✅ Fully supports **NVIDIA driver 570** and newer.
* ✅ Compatible with **CUDA 12.6 Docker images**.
* ✅ Recommended base system for **GVirtuS + Docker + CUDA 12.6** environments.

---

## 📦 GVirtuS Docker Compatibility

In our **GVirtuS system**, we rely on:

* **CUDA 12.6 Docker image**
* **NVIDIA driver 570**
* **Ubuntu 22.04 LTS host system**

This combination ensures:

* ✅ Full support for GPU acceleration in containers
* ✅ Compatibility with CUDA and cuDNN via GVirtuS
* ✅ Stability for deep learning and virtualized GPU workloads

---

## ✅ Recommended Setup for GVirtuS

| Component        | Version/Recommendation                    |
| ---------------- | ----------------------------------------- |
| Ubuntu Host      | 22.04 LTS                                 |
| NVIDIA Driver    | 570                                       |
| Docker Image     | CUDA 12.6 (e.g., `nvidia/cuda:12.6-base`) |
| GVirtuS Frontend | Compatible with CUDA 12.6                 |

---

