VENV_NAME = .venv
PYTHON = python3
REQS = numpy tqdm
VERSION = v0.0.1

$(VENV_NAME)/bin/activate:
	$(PYTHON) -m venv $(VENV_NAME)
	$(VENV_NAME)/bin/pip install --upgrade pip
	$(VENV_NAME)/bin/pip install $(REQS)

setupEnvironment: $(VENV_NAME)/bin/activate
	@echo -e "\033[0;32mSetup complete. Virtual environment created and dependencies installed.\033[0m"
	@echo -e "\033[0;36mActivate venv with: source $(VENV_NAME)/bin/activate\033[0m"

cleanEnvironment:
	@rm -rf $(VENV_NAME)
	@echo -e "\033[0;31mVirtual environment removed.\033[0m"

showVersion:
	@echo -e "\033[0;33mSynapse $(VERSION)\033[0m"
