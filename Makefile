CURRENT_VERSION=$(shell cat shared/version.py | cut -d '"' -f 2)
IMAGE_NAME=llm-finetuning
DOCKER_IMAGE_NAME=dataesr/llm
GHCR_IMAGE_NAME=ghcr.io/$(DOCKER_IMAGE_NAME)


# Build Docker image for finetuning
docker-build-finetuning:
	@echo "Building finetuning docker image..."
	docker build -f finetuning/Dockerfile -t $(GHCR_IMAGE_NAME)-finetuning:$(CURRENT_VERSION) -t $(GHCR_IMAGE_NAME)-finetuning:latest .
	@echo "Finetuning Docker image built."

# Build Docker image for inference
docker-build-inference:
	@echo "Building inference docker image..."
	docker build -f inference/Dockerfile -t $(GHCR_IMAGE_NAME)-inference:$(CURRENT_VERSION) -t $(GHCR_IMAGE_NAME)-inference:latest .
	@echo "Inference Docker image built."

# Build Docker image for inference
docker-build-inference-app:
	@echo "Building inference docker image..."
	docker build -f inference_app/Dockerfile -t $(GHCR_IMAGE_NAME)-inference-app:$(CURRENT_VERSION) -t $(GHCR_IMAGE_NAME)-inference-app:latest .
	@echo "Inference Dock

# Push Docker image for finetuning
docker-push-finetuning:
	@echo Pushing a new docker image
	docker push $(GHCR_IMAGE_NAME)-finetuning --all-tags
	@echo Docker image pushed

# Push Docker image for inference
docker-push-inference:
	@echo Pushing a new docker image
	docker push $(GHCR_IMAGE_NAME)-inference --all-tags
	@echo Docker image pushed

# Push Docker image for inference app
docker-push-inference-app:
	@echo Pushing a new docker image
	docker push $(GHCR_IMAGE_NAME)-inference-app --all-tags
	@echo Docker image pushed