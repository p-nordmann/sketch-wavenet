
eqx-wavenet:
	git clone https://github.com/p-nordmann/eqx-wavenet
	cd eqx-wavenet && pip3 install -e .

drawsvg:
	apt install libcairo2-dev -y
	pip3 install "drawsvg[all]~=2.0"


tensorboard:
	pip3 install tensorboard

install:
	pip3 install -e .

clean:
	rm -rf ./eqx-wavenet

all: clean drawsvg eqx-wavenet tensorboard install
