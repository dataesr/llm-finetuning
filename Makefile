CURRENT_VERSION=$(shell cat server/version.py | cut -d '"' -f 2)
DOCKER_IMAGE_NAME=dataesr/llm_finetuning
GHCR_IMAGE_NAME=ghcr.io/$(DOCKER_IMAGE_NAME)

docker-build:
	@echo Building a new docker image
	docker build -t $(GHCR_IMAGE_NAME):$(CURRENT_VERSION) -t $(GHCR_IMAGE_NAME):latest ./server
	@echo Docker image built

docker-run:
	@echo Running a new docker image
	docker run -p 5000:5000 -t $(GHCR_IMAGE_NAME):latest