.PHONY: all clean

all:
	mkdir -p build
	cmake -S . -B build
	cmake --build build
	mv build/libsample.* .

clean:
	rm -rf build libsample.*
