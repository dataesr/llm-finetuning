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

docker-push:
	@echo Pushing a new docker image
	docker push $(GHCR_IMAGE_NAME):$(CURRENT_VERSION)
	docker push $(GHCR_IMAGE_NAME):latest
	@echo Docker image pushed

release:
	echo 'VERSION = "$(VERSION)"'' > server/version.py
	git commit -am '[release] version $(VERSION)'
	git tag $(VERSION)
	@echo If everything is OK, you can push with tags i.e. git push origin main --tags