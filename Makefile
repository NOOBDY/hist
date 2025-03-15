.PHONY: all build clean

all: build shader/test.vert.spv shader/test.frag.spv

build:
	cargo build

shader/test.vert.spv: shader/test.vert.glsl
	glslang -o shader/test.vert.spv -V shader/test.vert.glsl

shader/test.frag.spv: shader/test.frag.glsl
	glslang -o shader/test.frag.spv -V shader/test.frag.glsl

clean:
	cargo clean
	rm shader/*.spv

