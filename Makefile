.PHONY: docker-build-push-dev docker-build-push-prod run-gvirtus-backend-dev run-gvirtus-tests stop-gvirtus docker-build-push-openpose run-openpose-frontend

docker-build-push-dev:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/dev/Dockerfile \
		-t darsh916/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04 \
		.

docker-build-push-prod:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/prod/Dockerfile \
		-t darsh916/gvirtus:cuda12.6.3-cudnn-ubuntu22.04 \
		.

run-gvirtus-backend-dev:
	docker run \
		--rm \
		-it \
		--network host \
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
		-v ./examples:/gvirtus/examples/ \
		--entrypoint /entrypoint.sh \
		--name gvirtus \
		--runtime=nvidia \
		darsh916/gvirtus-dependencies:cuda12.6.3-cudnn-ubuntu22.04


run-gvirtus-tests:
	docker exec \
		-it gvirtus \
		bash -c \
		'export LD_LIBRARY_PATH=$$GVIRTUS_HOME/lib/frontend:$$LD_LIBRARY_PATH && \
		 cd /gvirtus/build/tests && ./test_cudnn'                  

stop-gvirtus:
	docker stop gvirtus


docker-build-push-openpose:
	docker buildx build \
		--platform linux/amd64 \
		--push \
		--no-cache \
		-f docker/openpose/Dockerfile \
		-t darsh916/openpose_gvirtus:cuda12.6 \
		docker/openpose

run-openpose-frontend:
	xhost +local:root
	docker run \
		--rm \
		-it \
		--gpus all \
		--network host \
		--env DISPLAY=$$DISPLAY \
		--env XAUTHORITY=/root/.Xauthority \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v $$HOME/.Xauthority:/root/.Xauthority \
		-v ./examples/openpose-gvirtus:/home/openpose/examples/openpose-gvirtus \
		--name openpose-frontend \
		darsh916/openpose_gvirtus:cuda12.6