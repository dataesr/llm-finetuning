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

# Push Docker image for finetuning
docker-push-finetuning:
	@echo Pushing a new docker image
	docker push $(GHCR_IMAGE_NAME)-finetuning:$(CURRENT_VERSION)
	docker push $(GHCR_IMAGE_NAME)-finetuning:latest
	@echo Docker image pushed

# Push Docker image for inference
docker-push-inference:
	@echo Pushing a new docker image
	docker push $(GHCR_IMAGE_NAME)-inference:$(CURRENT_VERSION)
	docker push $(GHCR_IMAGE_NAME)-inference:latest
	@echo Docker image pushed

release:
	echo 'VERSION = "$(VERSION)"'' > project/version.py
	git commit -am '[release] version $(VERSION)'
	git tag $(VERSION)
	@echo If everything is OK, you can push with tags i.e. git push origin main --tags