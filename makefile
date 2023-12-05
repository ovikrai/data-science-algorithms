#####################################################
# MAIN PROJECT COMMANDS
#####################################################
update-pip:
	pip3 install --upgrade pip

update-build:
	python3 -m pip install --upgrade build

install-all: update-pip install-dev install

install:
	pip3 install -r requirements-dev.txt
	pip3 install -r requirements.txt

install-dev:
	pip3 install --upgrade pip
	pip3 install -r requirements-dev.txt

upgrade: update-pip
	pip3 install -r requirements-dev.txt --upgrade
	pip3 install -r requirements.txt --upgrade

clean:
	rm -rf ./log/
	rm -rf ./build/
	rm -rf ./dist/
	rm -rf ./images/*.pinavault
	rm -rf ./src/*.pyo
	rm -rf ./src/__pycache__
	rm -rf ./src/*.egg-info/
	rm -rf ./src/pinavault/*.c
	rm -rf ./src/pinavault/*.so
	rm -rf ./src/pinavault/__pycache__

run:
	python3 ./main.py

publish: clean pack upload

debug:
	python3 -m pdb ./main.py

doc:
	pydoc -w ./src

pack:
	python3 -m build

upload:
	scp ./dist/*.whl ovikrai@192.168.0.9:~/packages/
	scp ./dist/*.tar.gz ovikrai@192.168.0.9:~/packages/



#####################################################
# TESTING AND LINTING
#####################################################
test-unit-resources:
	coverage run -m pytest -vv

test-unit:
	coverage run -m pytest -k $(TEST_NAME) -vv

test-cover:
	coverage report

test-clean:
	coverage erase